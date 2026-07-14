package router

import (
	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// deviceAssigner places the members of a group onto the devices the group
// declared in `gpus`. One member per device, assigned when it starts: the model
// is handed its device through the group's device env var (CUDA_VISIBLE_DEVICES
// by default), which the process applies over its own env.
//
// Freeing a device is the scheduler's job, not this one's. A group with N
// devices gets a recent-model pool of N (resolved at config load), so the
// scheduler already holds N-1 members loaded and evicts the least recently used
// one when an N+1th member is requested. By the time assign runs, that eviction
// is in the swap's evict set, so its device counts as free here.
type deviceAssigner struct {
	modelToGroup   map[string]string
	groupMembers   map[string][]string
	groupDevices   map[string][]string
	groupDeviceEnv map[string]string

	// assigned is the device each model was last given. Entries for models that
	// are no longer running are stale and ignored — occupancy is always derived
	// from live process state, so a model that stopped on its own (a TTL expiry)
	// releases its device without anyone having to notice.
	assigned map[string]string
}

// newDeviceAssigner returns nil when no group declares devices, which is the
// common case and keeps the router's swap path untouched.
func newDeviceAssigner(conf config.Config) *deviceAssigner {
	d := &deviceAssigner{
		modelToGroup:   make(map[string]string),
		groupMembers:   make(map[string][]string),
		groupDevices:   make(map[string][]string),
		groupDeviceEnv: make(map[string]string),
		assigned:       make(map[string]string),
	}

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

	if len(d.groupDevices) == 0 {
		return nil
	}
	return d
}

// assign picks the device modelID starts on and reports it as a "KEY=VALUE" env
// entry. evict is the set of models this swap is about to stop, so the devices
// they hold count as free. state reports a model's current process state.
//
// It returns ok=false when the model's group does not manage devices, and when
// every device is taken — which the pool sizing is supposed to prevent, so the
// caller logs it rather than papering over it.
func (d *deviceAssigner) assign(modelID string, evict []string, state func(string) (process.ProcessState, bool)) (env string, ok bool) {
	groupID, managed := d.modelToGroup[modelID]
	if !managed {
		return "", false
	}

	evicting := make(map[string]struct{}, len(evict))
	for _, id := range evict {
		evicting[id] = struct{}{}
	}

	// A device is taken when a sibling that is not being evicted still holds it.
	// Anything that is not Stopped counts — a model that is starting or stopping
	// is still holding its memory.
	taken := make(map[string]struct{})
	for _, sibling := range d.groupMembers[groupID] {
		if sibling == modelID {
			continue
		}
		if _, evicted := evicting[sibling]; evicted {
			continue
		}
		device, hasDevice := d.assigned[sibling]
		if !hasDevice {
			continue
		}
		if s, known := state(sibling); known && s != process.StateStopped {
			taken[device] = struct{}{}
		}
	}

	devices := d.groupDevices[groupID]

	// Prefer the device this model last ran on: when it is free, reusing it
	// keeps a model pinned to one GPU across reloads, which is friendlier to
	// anything caching per-device (compile caches, NUMA placement).
	if previous, had := d.assigned[modelID]; had {
		if _, busy := taken[previous]; !busy && containsDevice(devices, previous) {
			return d.bind(modelID, groupID, previous), true
		}
	}

	for _, device := range devices {
		if _, busy := taken[device]; busy {
			continue
		}
		return d.bind(modelID, groupID, device), true
	}

	return "", false
}

func (d *deviceAssigner) bind(modelID, groupID, device string) string {
	d.assigned[modelID] = device
	return d.groupDeviceEnv[groupID] + "=" + device
}

// deviceOf reports the device a model is currently assigned, for tests and logs.
func (d *deviceAssigner) deviceOf(modelID string) (string, bool) {
	device, ok := d.assigned[modelID]
	return device, ok
}

func containsDevice(devices []string, target string) bool {
	for _, device := range devices {
		if device == target {
			return true
		}
	}
	return false
}
