package router

import (
	"strings"
	"sync"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// deviceAssigner places the members of a group onto the devices the group
// declared in `gpus`: one member per device, chosen when the member starts.
//
// Freeing a device is not this type's job. A group with N devices gets a
// recent-model pool of N (resolved at config load), so the scheduler already
// holds N-1 members loaded and evicts the least recently used one when an
// N+1th member is requested. By the time assign runs, that eviction is in the
// swap's evict set, so its device counts as free here.
//
// Occupancy is read from live process state rather than from bookkeeping of
// its own: a member that exits on its own — a TTL expiry, a crash — releases
// its device with nobody having to notice.
type deviceAssigner struct {
	mu             sync.RWMutex
	modelToGroup   map[string]string
	groupMembers   map[string][]string
	groupDevices   map[string][]string
	groupDeviceEnv map[string]string

	// lastDevice is the device each model was last placed on. It is only a
	// preference: reusing the same GPU across reloads is friendlier to
	// anything caching per device (compile caches, NUMA placement). A stale
	// entry can never cause a double placement, because occupancy comes from
	// live process state.
	lastDevice map[string]string
}

// newDeviceAssigner returns nil when no group declares devices, which is the
// common case and leaves the router's swap path untouched.
func newDeviceAssigner(conf config.Config) *deviceAssigner {
	d := &deviceAssigner{
		modelToGroup:   make(map[string]string),
		groupMembers:   make(map[string][]string),
		groupDevices:   make(map[string][]string),
		groupDeviceEnv: make(map[string]string),
		lastDevice:     make(map[string]string),
	}
	if !d.load(conf) {
		return nil
	}
	return d
}

// load fills the maps from conf and reports whether any group manages devices.
func (d *deviceAssigner) load(conf config.Config) bool {
	for groupID, groupConfig := range conf.Routing.Router.Settings.Groups {
		if len(groupConfig.GPUs) == 0 {
			continue
		}
		d.groupDevices[groupID] = groupConfig.GPUs
		d.groupDeviceEnv[groupID] = groupConfig.Device()
		d.groupMembers[groupID] = groupConfig.Members
		for _, member := range groupConfig.Members {
			d.modelToGroup[member] = groupID
		}
	}
	return len(d.groupDevices) > 0
}

// update re-reads the group topology after a surgical config reload. A reload
// that changes the groups themselves is not surgical (DiffModels reports a
// structural change and forces a full reload), so in practice this only ever
// re-reads the same topology — but holding a stale copy would be a bug waiting
// for the day that stops being true.
func (d *deviceAssigner) update(conf config.Config) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.modelToGroup = make(map[string]string)
	d.groupMembers = make(map[string][]string)
	d.groupDevices = make(map[string][]string)
	d.groupDeviceEnv = make(map[string]string)
	d.load(conf)
}

// manages reports whether modelID belongs to a group that places its members.
func (d *deviceAssigner) manages(modelID string) bool {
	if d == nil {
		return false
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	_, ok := d.modelToGroup[modelID]
	return ok
}

// deviceEnvFor returns the environment variable modelID's group hands the
// assigned device to the process as.
func (d *deviceAssigner) deviceEnvFor(modelID string) string {
	if d == nil {
		return config.DefaultDeviceEnv
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	groupID, ok := d.modelToGroup[modelID]
	if !ok {
		return config.DefaultDeviceEnv
	}
	return d.groupDeviceEnv[groupID]
}

// assign picks the device modelID should start on. evict is the set of models
// this swap is about to stop, so the devices they hold count as free.
// deviceOf reports the device a running model currently occupies ("" when it
// is not running).
//
// ok is false when the model's group does not manage devices, and when every
// device is taken — which the pool sizing is supposed to prevent, so the caller
// says so loudly rather than papering over it.
func (d *deviceAssigner) assign(modelID string, evict []string, deviceOf func(string) string) (device string, ok bool) {
	if d == nil {
		return "", false
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	device, ok = d.choose(modelID, evict, deviceOf)
	if ok {
		d.lastDevice[modelID] = device
	}
	return device, ok
}

// peek is assign without committing: it answers which device the model would
// be placed on, for callers that need to know before the swap starts (the VRAM
// check has to look at the device the model will actually land on, not the one
// its env names). Safe to call at any time; it records nothing.
func (d *deviceAssigner) peek(modelID string, evict []string, deviceOf func(string) string) (string, bool) {
	if d == nil {
		return "", false
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.choose(modelID, evict, deviceOf)
}

// choose is the placement decision. The caller holds d.mu.
func (d *deviceAssigner) choose(modelID string, evict []string, deviceOf func(string) string) (device string, ok bool) {
	groupID, managed := d.modelToGroup[modelID]
	if !managed {
		return "", false
	}

	evicting := make(map[string]struct{}, len(evict))
	for _, id := range evict {
		evicting[id] = struct{}{}
	}

	// A device is taken when a sibling that is not being evicted still holds
	// it. A model that is starting or stopping still holds its memory, and
	// deviceOf reports a device for exactly those states.
	taken := make(map[string]struct{})
	for _, sibling := range d.groupMembers[groupID] {
		if sibling == modelID {
			continue
		}
		if _, evicted := evicting[sibling]; evicted {
			continue
		}
		for _, dev := range splitDevices(deviceOf(sibling)) {
			taken[dev] = struct{}{}
		}
	}

	devices := d.groupDevices[groupID]

	// Prefer the device this model last ran on when it is free.
	if previous, had := d.lastDevice[modelID]; had {
		if _, busy := taken[previous]; !busy && containsDevice(devices, previous) {
			return previous, true
		}
	}

	for _, dev := range devices {
		if _, busy := taken[dev]; busy {
			continue
		}
		return dev, true
	}

	return "", false
}

// splitDevices turns a CUDA_VISIBLE_DEVICES value into the devices it names. A
// model given "0,1" occupies both.
func splitDevices(value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func containsDevice(devices []string, target string) bool {
	for _, device := range devices {
		if device == target {
			return true
		}
	}
	return false
}

// processDeviceFunc returns a lookup for the device a running model occupies,
// read from live process state. A stopped or shut-down process reports "".
func (b *baseRouter) processDeviceFunc() func(string) string {
	procs := b.processesAt()
	return func(modelID string) string {
		p, ok := procs[modelID]
		if !ok {
			return ""
		}
		switch p.State() {
		case process.StateStopped, process.StateShutdown:
			return ""
		}
		if gpu, ok := p.(interface{ GPU() string }); ok {
			return gpu.GPU()
		}
		return ""
	}
}
