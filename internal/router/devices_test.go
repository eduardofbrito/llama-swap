package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// gpuGroupConfig builds a config with one device-managing group, the way
// config load leaves it: members' pool size resolved to the device count.
func gpuGroupConfig(devices []string, members []string, deviceEnv string) config.Config {
	conf := config.Config{HealthCheckTimeout: 5, Models: map[string]config.ModelConfig{}}
	for _, m := range members {
		conf.Models[m] = config.ModelConfig{}
	}
	conf.Routing.Router.Settings.Groups = map[string]config.GroupConfig{
		"g": {Members: members, GPUs: devices, DeviceEnv: deviceEnv},
	}
	conf.Routing.Scheduler.Settings.Fifo.PoolSize = map[string]int{}
	for _, m := range members {
		conf.Routing.Scheduler.Settings.Fifo.PoolSize[m] = len(devices)
	}
	return conf
}

// staticDevices is a deviceOf lookup backed by a plain map, for the unit tests
// that drive the assigner directly.
func staticDevices(m map[string]string) func(string) string {
	return func(id string) string { return m[id] }
}

func TestDeviceAssigner_PlacesMembersOnFreeDevices(t *testing.T) {
	d := newDeviceAssigner(gpuGroupConfig([]string{"0", "1"}, []string{"a", "b"}, ""))
	if d == nil {
		t.Fatal("assigner is nil; a group declaring gpus must produce one")
	}

	// Nothing running: the first member takes the first device.
	got, ok := d.assign("a", nil, staticDevices(nil))
	if !ok || got != "0" {
		t.Fatalf("assign(a) = %q, %v; want 0, true", got, ok)
	}
	// a now holds device 0, so b must land on 1.
	got, ok = d.assign("b", nil, staticDevices(map[string]string{"a": "0"}))
	if !ok || got != "1" {
		t.Errorf("assign(b) = %q, %v; want 1, true", got, ok)
	}
}

// TestDeviceAssigner_EvictedMemberReleasesItsDevice: the eviction set is the
// whole reason a device is free at placement time.
func TestDeviceAssigner_EvictedMemberReleasesItsDevice(t *testing.T) {
	d := newDeviceAssigner(gpuGroupConfig([]string{"0", "1"}, []string{"a", "b", "c"}, ""))
	live := staticDevices(map[string]string{"a": "0", "b": "1"})

	// Both devices busy and nothing evicted: no placement is possible.
	if _, ok := d.assign("c", nil, live); ok {
		t.Error("assign(c) succeeded with every device busy; want ok=false")
	}
	// Evicting a frees device 0.
	got, ok := d.assign("c", []string{"a"}, live)
	if !ok || got != "0" {
		t.Errorf("assign(c, evict=[a]) = %q, %v; want 0, true", got, ok)
	}
}

// TestDeviceAssigner_PrefersPreviousDevice keeps a model pinned to one GPU
// across reloads, which is friendlier to per-device caches.
func TestDeviceAssigner_PrefersPreviousDevice(t *testing.T) {
	d := newDeviceAssigner(gpuGroupConfig([]string{"0", "1"}, []string{"a", "b"}, ""))

	// a is placed on 1 because b holds 0.
	got, _ := d.assign("a", nil, staticDevices(map[string]string{"b": "0"}))
	if got != "1" {
		t.Fatalf("assign(a) = %q, want 1", got)
	}
	// With everything free, a goes back to 1 rather than to the first device.
	got, _ = d.assign("a", nil, staticDevices(nil))
	if got != "1" {
		t.Errorf("assign(a) = %q, want 1 (the device it last ran on)", got)
	}
}

// TestDeviceAssigner_PeekDoesNotCommit: the VRAM check asks where a model
// would land, and must not change the placement state by asking.
func TestDeviceAssigner_PeekDoesNotCommit(t *testing.T) {
	d := newDeviceAssigner(gpuGroupConfig([]string{"0", "1"}, []string{"a", "b"}, ""))

	got, ok := d.peek("a", nil, staticDevices(map[string]string{"b": "0"}))
	if !ok || got != "1" {
		t.Fatalf("peek(a) = %q, %v; want 1, true", got, ok)
	}
	// Nothing was recorded, so with every device free a takes the first one.
	got, _ = d.assign("a", nil, staticDevices(nil))
	if got != "0" {
		t.Errorf("assign(a) = %q after a peek, want 0; peek must not record a preference", got)
	}
}

// TestDeviceAssigner_MultiDeviceSiblingOccupiesBoth: a member running with
// "0,1" holds both devices.
func TestDeviceAssigner_MultiDeviceSiblingOccupiesBoth(t *testing.T) {
	d := newDeviceAssigner(gpuGroupConfig([]string{"0", "1"}, []string{"a", "b"}, ""))
	if _, ok := d.assign("b", nil, staticDevices(map[string]string{"a": "0,1"})); ok {
		t.Error("assign(b) succeeded while a held both devices; want ok=false")
	}
}

func TestDeviceAssigner_NilWithoutDeviceGroups(t *testing.T) {
	conf := config.Config{Routing: config.RoutingConfig{}}
	if d := newDeviceAssigner(conf); d != nil {
		t.Error("assigner must be nil when no group declares gpus")
	}
	// Every method has to tolerate the nil receiver: it is the common case.
	var d *deviceAssigner
	if d.manages("a") {
		t.Error("nil assigner claims to manage a model")
	}
	if _, ok := d.assign("a", nil, staticDevices(nil)); ok {
		t.Error("nil assigner assigned a device")
	}
	if _, ok := d.peek("a", nil, staticDevices(nil)); ok {
		t.Error("nil assigner peeked a device")
	}
	if got := d.deviceEnvFor("a"); got != config.DefaultDeviceEnv {
		t.Errorf("deviceEnvFor = %q, want the default", got)
	}
	d.update(config.Config{}) // must not panic
}

// TestBaseRouter_GroupGPUsPlaceMembersAcrossDevices drives the whole router:
// two members of a two-GPU group must end up on different devices, with the
// device reaching the process through its start options.
func TestBaseRouter_GroupGPUsPlaceMembersAcrossDevices(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true

	conf := gpuGroupConfig([]string{"0", "1"}, []string{"a", "b"}, "")
	r := newTestBaseWithConfig(t, conf,
		map[string]process.Process{"a": a, "b": bProc}, &stubPlanner{})

	serve(t, r, "a")
	serve(t, r, "b")

	devA, devB := a.lastOpts().GpuOverride, bProc.lastOpts().GpuOverride
	if devA == "" || devB == "" {
		t.Fatalf("devices = %q and %q; both members must be placed", devA, devB)
	}
	if devA == devB {
		t.Errorf("both members placed on device %q; a group places one member per device", devA)
	}
}

// TestBaseRouter_GroupGPUsRequestOverrideWins is the precedence rule: an
// explicit per-request GPU beats the group's placement. Someone who picked a
// GPU asked for that GPU.
func TestBaseRouter_GroupGPUsRequestOverrideWins(t *testing.T) {
	a := newFakeProcess("a")
	a.autoReady = true

	conf := gpuGroupConfig([]string{"0", "1"}, []string{"a", "b"}, "")
	r := newTestBaseWithConfig(t, conf, map[string]process.Process{"a": a}, &stubPlanner{})

	req := newRequest("a")
	req = req.WithContext(swaputil.SetContext(req.Context(), swaputil.ReqContextData{
		Model: "a", ModelID: "a", GpuOverride: "7",
	}))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}

	if got := a.lastOpts().GpuOverride; got != "7" {
		t.Errorf("GpuOverride = %q, want 7; an explicit request override must win over the group", got)
	}
}

// TestBaseRouter_GroupGPUsUseConfiguredDeviceEnv checks that a group's
// deviceEnv reaches the process, so a non-CUDA runtime can be placed with the
// variable it actually reads.
func TestBaseRouter_GroupGPUsUseConfiguredDeviceEnv(t *testing.T) {
	a := newFakeProcess("a")
	a.autoReady = true

	conf := gpuGroupConfig([]string{"0"}, []string{"a"}, "HIP_VISIBLE_DEVICES")
	r := newTestBaseWithConfig(t, conf, map[string]process.Process{"a": a}, &stubPlanner{})

	serve(t, r, "a")

	if got := a.lastOpts().GpuEnvVar; got != "HIP_VISIBLE_DEVICES" {
		t.Errorf("GpuEnvVar = %q, want HIP_VISIBLE_DEVICES", got)
	}
}

// TestBaseRouter_NoDeviceGroupLeavesOptionsAlone pins the default: a model
// outside any device-managing group is started exactly as before.
func TestBaseRouter_NoDeviceGroupLeavesOptionsAlone(t *testing.T) {
	a := newFakeProcess("a")
	a.autoReady = true
	r := newTestBase(t, map[string]process.Process{"a": a}, &stubPlanner{})

	serve(t, r, "a")

	if got := a.lastOpts().GpuOverride; got != "" {
		t.Errorf("GpuOverride = %q, want empty for a model no group places", got)
	}
	if got := a.lastOpts().GpuEnvVar; got != "" {
		t.Errorf("GpuEnvVar = %q, want empty for a model no group places", got)
	}
}
