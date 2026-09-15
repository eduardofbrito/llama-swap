package router

import (
	"sync"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// DeviceMemory is one GPU's memory reading, in MB, as reported by the host.
type DeviceMemory struct {
	UsedMB  int
	TotalMB int
}

// VRAMOracle is the router's window into GPU memory. The server implements it
// over the performance monitor (live per-device readings) and the store
// (measurements persisted across restarts); tests supply a fake.
//
// It is deliberately narrow: the router must not know about perf rings or SQL,
// and the admission check runs on the scheduler's run loop, so every method
// here has to be cheap and non-blocking.
type VRAMOracle interface {
	// DeviceMemory reports the most recent reading for each GPU, keyed by the
	// device index as CUDA_VISIBLE_DEVICES names it ("0", "1", ...). An empty
	// map means the host reported nothing, which disables the check.
	DeviceMemory() map[string]DeviceMemory

	// MeasuredVRAMMB reports the VRAM a model needed on its last successful
	// load, used when the model does not declare vramMB.
	MeasuredVRAMMB(modelID string) (int, bool)

	// RecordVRAMMB persists a fresh measurement. It must not block the caller.
	RecordVRAMMB(modelID string, vramMB int)
}

// vramGuard answers one question for the scheduler: after this swap stops the
// models it plans to, will the target model's GPU have room for it?
//
// The question is worth asking because llama-swap does not necessarily own the
// GPU. Another process — a training job, a second llama-swap, a desktop
// session — can hold memory the router knows nothing about, so "everything I
// evicted is stopped" is not the same as "the memory is free". Without the
// check that shows up as a load failing deep inside the runtime, minutes later.
//
// The guard is conservative on purpose: every path where it does not have a
// trustworthy answer admits the model. Refusing to load on ignorance would be
// worse than the behaviour it replaces.
type vramGuard struct {
	oracle VRAMOracle

	// runningDeviceOf reports the GPU a model is on RIGHT NOW, read from live
	// process state. The router wires it in; a nil func (tests, a guard built
	// standalone) falls back to the configured device.
	//
	// It exists because deviceOf below can only see what the config pins, and
	// a model in a group with `gpus:` pins nothing — the device assigner picks
	// one at load time. Without this, every evictee in such a group resolved
	// to "", the memory it was about to give back was credited to a device
	// that does not exist, and the target was refused for the full size of a
	// model that was already on its way out.
	runningDeviceOf func(string) string

	mu sync.RWMutex
	// cfg and models are swapped on a surgical reload, so they are guarded
	// rather than captured once.
	cfg    config.FifoConfig
	models map[string]config.ModelConfig
}

func newVRAMGuard(conf config.Config, oracle VRAMOracle) *vramGuard {
	g := &vramGuard{oracle: oracle}
	g.update(conf)
	return g
}

// update re-reads the config after a reload.
func (g *vramGuard) update(conf config.Config) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.cfg = conf.Routing.Scheduler.Settings.Fifo
	g.models = conf.Models
}

// enabled reports whether the check should run at all.
func (g *vramGuard) enabled() bool {
	if g == nil || g.oracle == nil {
		return false
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.cfg.VramCheck
}

// requirementMB is how much VRAM modelID needs: its declared vramMB, or the
// value measured on its last successful load. ok is false when neither is
// known, which admits the model.
func (g *vramGuard) requirementMB(modelID string) (int, bool) {
	g.mu.RLock()
	mc, known := g.models[modelID]
	g.mu.RUnlock()

	if known && mc.VramMB > 0 {
		return mc.VramMB, true
	}
	if measured, ok := g.oracle.MeasuredVRAMMB(modelID); ok && measured > 0 {
		return measured, true
	}
	return 0, false
}

// deviceOf returns the GPU a model is pinned to by its config, or "" when it
// pins none.
func (g *vramGuard) deviceOf(modelID string) string {
	g.mu.RLock()
	mc, ok := g.models[modelID]
	g.mu.RUnlock()
	if !ok {
		return ""
	}
	return process.DefaultGPU(mc.Env)
}

// evicteeDevice resolves the GPU an about-to-be-evicted model occupies: where
// it actually is, falling back to where the config pins it. ok is false when
// neither answers, which admits the target — the same rule requirementMB
// follows, and for the same reason: a device the guard cannot name makes the
// whole sum meaningless, and refusing on ignorance turns a gap in the guard's
// knowledge into an outage.
func (g *vramGuard) evicteeDevice(modelID string) (string, bool) {
	if g.runningDeviceOf != nil {
		if dev := g.runningDeviceOf(modelID); dev != "" {
			return dev, true
		}
	}
	if dev := g.deviceOf(modelID); dev != "" {
		return dev, true
	}
	return "", false
}

// shortfallMB reports how much VRAM is still missing for modelID to load after
// the models in evict are stopped, on the device it would load onto. Zero means
// there is room — or that the guard cannot answer, in which case the model is
// admitted.
//
// device is the GPU the request pinned (a per-request override), or "" to use
// the model's configured GPU.
func (g *vramGuard) shortfallMB(modelID, device string, evict []string) int {
	if !g.enabled() {
		return 0
	}
	need, known := g.requirementMB(modelID)
	if !known {
		return 0
	}

	readings := g.oracle.DeviceMemory()
	if len(readings) == 0 {
		// The host reports no GPU memory; there is nothing to check against.
		return 0
	}

	if device == "" {
		device = g.deviceOf(modelID)
	}

	g.mu.RLock()
	margin := g.cfg.VramMargin()
	g.mu.RUnlock()
	want := need + need*margin/100

	// Memory the evicted models are about to give back on each device.
	//
	// An evictee with no known requirement makes the whole sum meaningless: the
	// device's free memory after the swap is then unknowable, not zero. This
	// used to credit such a model with nothing and carry on, which reads as
	// conservative but is really the guard answering a question it cannot
	// answer — and it refuses loads that would have succeeded. It is how a
	// model sitting alone on a GPU, with a resident sibling about to be evicted
	// off that same GPU, got told it was short by the sibling's entire
	// footprint.
	//
	// So: admit, the same as every other path where the guard does not know.
	// Refusing on ignorance is the one behaviour this check must never have —
	// it turns a missing measurement into an outage. Declaring models.*.vramMB
	// is what buys the protection back.
	freed := make(map[string]int)
	for _, id := range evict {
		if id == modelID {
			continue
		}
		mb, ok := g.requirementMB(id)
		if !ok {
			return 0
		}
		dev, ok := g.evicteeDevice(id)
		if !ok {
			return 0
		}
		freed[dev] += mb
	}

	available := func(dev string) int {
		r, ok := readings[dev]
		if !ok || r.TotalMB <= 0 {
			return -1
		}
		return r.TotalMB - r.UsedMB + freed[dev]
	}

	if device != "" {
		free := available(device)
		if free < 0 {
			// The model names a device the host did not report (a multi-GPU
			// value like "0,1", or an index that does not exist here). Not
			// something to guess about.
			return 0
		}
		if free >= want {
			return 0
		}
		return want - free
	}

	// The model pins no device: it could land on any of them, so check against
	// the roomiest one. If even that cannot hold the model, no device can.
	best := -1
	for dev := range readings {
		if free := available(dev); free > best {
			best = free
		}
	}
	if best < 0 || best >= want {
		return 0
	}
	return want - best
}

// deviceSnapshot returns the current per-device readings, or nil when the
// guard has no oracle. Used as the baseline for a load measurement.
func (g *vramGuard) deviceSnapshot() map[string]DeviceMemory {
	if g == nil || g.oracle == nil {
		return nil
	}
	return g.oracle.DeviceMemory()
}

// observeLoad records the VRAM a model actually used, as the rise in its
// device's used memory across the load. before is the reading captured just
// before the process started.
//
// A measurement is only kept when it is positive and the device is the same in
// both readings; anything else means something other than this model moved the
// number, and a wrong measurement is worse than none.
// It reports why nothing was recorded, so a model that never acquires a
// measurement is diagnosable. That matters more than it looks: the readings
// come from the performance monitor's ring, sampled on an interval, and a swap
// that finishes inside one sampling period sees the same sample at both ends —
// a zero rise, no measurement, silently, forever. A model with no measurement
// and no declared vramMB is one the guard cannot reason about at all.
func (g *vramGuard) observeLoad(modelID, device string, before map[string]DeviceMemory) (recorded bool, why string) {
	if g == nil || g.oracle == nil || len(before) == 0 {
		return false, "no baseline reading"
	}
	if device == "" {
		device = g.deviceOf(modelID)
	}
	if device == "" {
		// Without a device there is no single number to attribute to this
		// model; a multi-GPU or unpinned load is not measured.
		return false, "model pins no single device"
	}
	start, hadStart := before[device]
	end, hadEnd := g.oracle.DeviceMemory()[device]
	if !hadStart || !hadEnd {
		return false, "device " + device + " missing from a reading"
	}
	used := end.UsedMB - start.UsedMB
	if used <= 0 {
		return false, "used memory did not rise across the load; the GPU sample likely did not refresh in time (see performance.every) — declare vramMB for this model"
	}
	g.oracle.RecordVRAMMB(modelID, used)
	return true, ""
}
