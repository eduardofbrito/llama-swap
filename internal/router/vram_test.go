package router

import (
	"sync"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
)

// fakeOracle is a scripted VRAMOracle: fixed device readings, a measurement
// table, and a record of what was written back.
type fakeOracle struct {
	mu       sync.Mutex
	devices  map[string]DeviceMemory
	measured map[string]int
	recorded map[string]int
}

func newFakeOracle(devices map[string]DeviceMemory) *fakeOracle {
	return &fakeOracle{
		devices:  devices,
		measured: map[string]int{},
		recorded: map[string]int{},
	}
}

func (f *fakeOracle) DeviceMemory() map[string]DeviceMemory {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make(map[string]DeviceMemory, len(f.devices))
	for k, v := range f.devices {
		out[k] = v
	}
	return out
}

func (f *fakeOracle) MeasuredVRAMMB(modelID string) (int, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	mb, ok := f.measured[modelID]
	return mb, ok
}

func (f *fakeOracle) RecordVRAMMB(modelID string, vramMB int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.recorded[modelID] = vramMB
}

// vramConfig builds a config with the check on and models pinned to devices.
func vramConfig(models map[string]config.ModelConfig) config.Config {
	conf := config.Config{Models: models}
	conf.Routing.Scheduler.Settings.Fifo.VramCheck = true
	conf.Routing.Scheduler.Settings.Fifo.VramMarginPct = -1 // no margin, exact math
	return conf
}

func pinned(device string, vramMB int) config.ModelConfig {
	return config.ModelConfig{Env: []string{"CUDA_VISIBLE_DEVICES=" + device}, VramMB: vramMB}
}

func TestVRAMGuard_AdmitsWhenDeviceHasRoom(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 1000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"big": pinned("0", 20000),
	}), oracle)

	if got := g.shortfallMB("big", "", nil); got != 0 {
		t.Errorf("shortfall = %d, want 0 (23000 MB free, model needs 20000)", got)
	}
}

func TestVRAMGuard_ReportsShortfallWhenGPUHeldOutside(t *testing.T) {
	// 24 GB card with 20 GB held by something llama-swap did not start.
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 20000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"big": pinned("0", 20000),
	}), oracle)

	got := g.shortfallMB("big", "", nil)
	if want := 16000; got != want {
		t.Errorf("shortfall = %d, want %d (needs 20000, only 4000 free)", got, want)
	}
}

func TestVRAMGuard_CountsMemoryTheEvictionWillFree(t *testing.T) {
	// Both models on device 0; the resident one is about to be evicted, so its
	// memory has to count as free for the incoming one.
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 20000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"incoming": pinned("0", 18000),
		"resident": pinned("0", 18000),
	}), oracle)

	if got := g.shortfallMB("incoming", "", nil); got == 0 {
		t.Error("expected a shortfall before accounting for the eviction")
	}
	if got := g.shortfallMB("incoming", "", []string{"resident"}); got != 0 {
		t.Errorf("shortfall = %d, want 0 once resident's 18000 MB is freed", got)
	}
}

func TestVRAMGuard_RequestOverrideWinsOverConfiguredDevice(t *testing.T) {
	// The model is configured for device 0 (full), but the request pinned
	// device 1 (empty). The explicit override decides which GPU is checked.
	oracle := newFakeOracle(map[string]DeviceMemory{
		"0": {UsedMB: 23000, TotalMB: 24000},
		"1": {UsedMB: 0, TotalMB: 24000},
	})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"m": pinned("0", 20000),
	}), oracle)

	if got := g.shortfallMB("m", "", nil); got == 0 {
		t.Error("configured device 0 is full; expected a shortfall")
	}
	if got := g.shortfallMB("m", "1", nil); got != 0 {
		t.Errorf("shortfall = %d on the overridden device 1, want 0", got)
	}
}

func TestVRAMGuard_DeclaredVramWinsOverMeasured(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 0, TotalMB: 24000}})
	oracle.measured["m"] = 1000 // a stale, far too small measurement
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"m": pinned("0", 30000), // declared: does not fit at all
	}), oracle)

	if got := g.shortfallMB("m", "", nil); got != 6000 {
		t.Errorf("shortfall = %d, want 6000; the declared vramMB must win over the measured value", got)
	}
}

func TestVRAMGuard_FallsBackToMeasuredWhenUndeclared(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 20000, TotalMB: 24000}})
	oracle.measured["m"] = 20000
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"m": pinned("0", 0), // nothing declared
	}), oracle)

	if got := g.shortfallMB("m", "", nil); got != 16000 {
		t.Errorf("shortfall = %d, want 16000 from the measured value", got)
	}
}

// TestVRAMGuard_AdmitsWhenNothingIsKnown is the conservative rule that keeps
// the feature from breaking working setups: a model with no declared and no
// measured requirement must load exactly as it does today.
func TestVRAMGuard_AdmitsWhenNothingIsKnown(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 24000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"m": pinned("0", 0),
	}), oracle)

	if got := g.shortfallMB("m", "", nil); got != 0 {
		t.Errorf("shortfall = %d; a model with no known requirement must be admitted", got)
	}
}

func TestVRAMGuard_DisabledAndNoOracleAdmitEverything(t *testing.T) {
	models := map[string]config.ModelConfig{"m": pinned("0", 30000)}
	full := map[string]DeviceMemory{"0": {UsedMB: 24000, TotalMB: 24000}}

	off := config.Config{Models: models} // VramCheck defaults to false
	if got := newVRAMGuard(off, newFakeOracle(full)).shortfallMB("m", "", nil); got != 0 {
		t.Errorf("shortfall = %d with the check off, want 0", got)
	}
	if got := newVRAMGuard(vramConfig(models), nil).shortfallMB("m", "", nil); got != 0 {
		t.Errorf("shortfall = %d with no oracle, want 0", got)
	}
	var nilGuard *vramGuard
	if got := nilGuard.shortfallMB("m", "", nil); got != 0 {
		t.Errorf("shortfall = %d on a nil guard, want 0", got)
	}
}

// TestVRAMGuard_UnknownDeviceAdmits covers a model pinned to something the
// host did not report, such as a multi-GPU "0,1" value.
func TestVRAMGuard_UnknownDeviceAdmits(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 24000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{
		"m": pinned("0,1", 20000),
	}), oracle)

	if got := g.shortfallMB("m", "", nil); got != 0 {
		t.Errorf("shortfall = %d; a device the host did not report must not be guessed at", got)
	}
}

// TestVRAMGuard_UnpinnedModelChecksRoomiestDevice: a model that pins no GPU
// could land on any of them, so only a shortfall on every device is a real one.
func TestVRAMGuard_UnpinnedModelChecksRoomiestDevice(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{
		"0": {UsedMB: 23000, TotalMB: 24000},
		"1": {UsedMB: 2000, TotalMB: 24000},
	})
	models := map[string]config.ModelConfig{"m": {VramMB: 20000}} // no env, no device
	g := newVRAMGuard(vramConfig(models), oracle)

	if got := g.shortfallMB("m", "", nil); got != 0 {
		t.Errorf("shortfall = %d; device 1 has room, so the model must be admitted", got)
	}

	// Now fill device 1 too: no device can hold it.
	oracle.devices["1"] = DeviceMemory{UsedMB: 23000, TotalMB: 24000}
	if got := g.shortfallMB("m", "", nil); got != 19000 {
		t.Errorf("shortfall = %d, want 19000 once every device is full", got)
	}
}

func TestVRAMGuard_MarginIsAppliedOnTopOfRequirement(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 0, TotalMB: 21000}})
	conf := config.Config{Models: map[string]config.ModelConfig{"m": pinned("0", 20000)}}
	conf.Routing.Scheduler.Settings.Fifo.VramCheck = true
	conf.Routing.Scheduler.Settings.Fifo.VramMarginPct = 10 // wants 22000

	g := newVRAMGuard(conf, oracle)
	if got := g.shortfallMB("m", "", nil); got != 1000 {
		t.Errorf("shortfall = %d, want 1000 (20000 + 10%% margin vs 21000 free)", got)
	}
}

func TestVRAMGuard_ObserveLoadRecordsTheRise(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 1000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{"m": pinned("0", 0)}), oracle)

	before := g.deviceSnapshot()
	oracle.devices["0"] = DeviceMemory{UsedMB: 19000, TotalMB: 24000}
	g.observeLoad("m", "", before)

	if got := oracle.recorded["m"]; got != 18000 {
		t.Errorf("recorded %d MB, want 18000 (the rise across the load)", got)
	}
}

// TestVRAMGuard_ObserveLoadIgnoresNonRises: memory going down or staying flat
// across a load means something other than this model moved the number, and a
// wrong measurement is worse than none.
func TestVRAMGuard_ObserveLoadIgnoresNonRises(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 19000, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{"m": pinned("0", 0)}), oracle)

	before := g.deviceSnapshot()
	oracle.devices["0"] = DeviceMemory{UsedMB: 8000, TotalMB: 24000} // went down
	g.observeLoad("m", "", before)

	if mb, ok := oracle.recorded["m"]; ok {
		t.Errorf("recorded %d MB; a non-positive delta must not be stored", mb)
	}
}

// TestVRAMGuard_UpdateTracksConfigReload makes sure a surgical reload is seen:
// the guard holds the config, and a stale copy would check the old numbers.
func TestVRAMGuard_UpdateTracksConfigReload(t *testing.T) {
	oracle := newFakeOracle(map[string]DeviceMemory{"0": {UsedMB: 0, TotalMB: 24000}})
	g := newVRAMGuard(vramConfig(map[string]config.ModelConfig{"m": pinned("0", 1000)}), oracle)
	if got := g.shortfallMB("m", "", nil); got != 0 {
		t.Fatalf("shortfall = %d before the reload, want 0", got)
	}

	g.update(vramConfig(map[string]config.ModelConfig{"m": pinned("0", 30000)}))
	if got := g.shortfallMB("m", "", nil); got != 6000 {
		t.Errorf("shortfall = %d after the reload, want 6000; the guard must see the new config", got)
	}
}
