package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/hw"
	"github.com/mostlygeek/llama-swap/internal/perf"
)

func gpuTestStrPtr(s string) *string { return &s }
func gpuTestIntPtr(i int) *int       { return &i }

func gpuTestSnapshot(accelerators ...hw.Accelerator) *hw.HardwareSnapshot {
	return &hw.HardwareSnapshot{
		SchemaVersion: hw.SchemaVersion,
		Accelerators:  accelerators,
	}
}

// TestServer_APIGpus_ListsSelectableGpus verifies /api/gpus returns exactly
// the accelerators a model can be loaded onto: kind "gpu" with a known
// DeviceIndex and a selectable vendor/model. Entries carry the real device
// index (nvidia-smi style), sorted ascending.
func TestServer_APIGpus_ListsSelectableGpus(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.hardware = gpuTestSnapshot(
		// display index 0, real device index 1 — must be offered
		hw.Accelerator{Index: 0, DeviceIndex: gpuTestIntPtr(1), Kind: "gpu", Vendor: gpuTestStrPtr("NVIDIA"), Model: gpuTestStrPtr("RTX 4090")},
		// unknown device index — must be skipped
		hw.Accelerator{Index: 1, Kind: "gpu", Vendor: gpuTestStrPtr("NVIDIA"), Model: gpuTestStrPtr("RTX 4090")},
		// non-gpu kind — must be skipped
		hw.Accelerator{Index: 2, DeviceIndex: gpuTestIntPtr(0), Kind: "cpu", Vendor: gpuTestStrPtr("Intel"), Model: gpuTestStrPtr("Xeon")},
		// integrated GPU without vendor/model info — must be skipped
		hw.Accelerator{Index: 3, DeviceIndex: gpuTestIntPtr(2), Kind: "gpu"},
		// AMD discrete GPU, real device index 0 — must be offered, and sort first
		hw.Accelerator{Index: 4, DeviceIndex: gpuTestIntPtr(0), Kind: "gpu", Vendor: gpuTestStrPtr("AMD"), Model: gpuTestStrPtr("Radeon RX 7900")},
	)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/gpus", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var got []apiGpu
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v, body = %s", err, w.Body.String())
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2, body = %s", len(got), w.Body.String())
	}
	if got[0].Index != 0 || got[1].Index != 1 {
		t.Errorf("indexes = [%d %d], want [0 1] (sorted, real device indexes)", got[0].Index, got[1].Index)
	}
	if !regexp.MustCompile(`^GPU 0 AMD Radeon RX 7900$`).MatchString(got[0].Label) {
		t.Errorf("label[0] = %q", got[0].Label)
	}
	if !regexp.MustCompile(`^GPU 1 NVIDIA RTX 4090$`).MatchString(got[1].Label) {
		t.Errorf("label[1] = %q", got[1].Label)
	}
}

// TestServer_APIGpus_EmptyWhenNoHardware: with no hardware snapshot the
// endpoint returns an empty (non-nil) list so the UI can hide the selector.
func TestServer_APIGpus_EmptyWhenNoHardware(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/gpus", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var got []apiGpu
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v, body = %s", err, w.Body.String())
	}
	if len(got) != 0 {
		t.Errorf("got %d gpus, want 0: %+v", len(got), got)
	}
}

// TestServer_APIGpus_HardwareUnavailable still lists nothing (200 + empty),
// not 503: the selector can simply stay hidden when detection fails.
func TestServer_APIGpus_HardwareUnavailable(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/gpus", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}
}

// TestGpuList_MergesLiveMemory checks that each GPU carries the memory reading
// for its own device index, which is what lets the GPUs page show memory held
// by processes llama-swap did not start.
func TestGpuList_MergesLiveMemory(t *testing.T) {
	hardware := gpuTestSnapshot(
		hw.Accelerator{
			Index: 0, Kind: "gpu", DeviceIndex: gpuTestIntPtr(0),
			Vendor: gpuTestStrPtr("NVIDIA"), Model: gpuTestStrPtr("RTX 4090"),
		},
		hw.Accelerator{
			Index: 1, Kind: "gpu", DeviceIndex: gpuTestIntPtr(1),
			Vendor: gpuTestStrPtr("NVIDIA"), Model: gpuTestStrPtr("RTX 4090"),
		},
	)
	memory := map[int]perf.GpuStat{
		0: {ID: 0, MemUsedMB: 18000, MemTotalMB: 24000},
		// device 1 has no sample at all
	}

	gpus := gpuList(hardware, memory)
	if len(gpus) != 2 {
		t.Fatalf("got %d gpus, want 2", len(gpus))
	}
	if gpus[0].UsedMB != 18000 || gpus[0].TotalMB != 24000 {
		t.Errorf("gpu 0 memory = %d/%d, want 18000/24000", gpus[0].UsedMB, gpus[0].TotalMB)
	}
	// No reading must stay zero so the JSON omits it: the UI has to tell
	// "unknown" apart from "empty GPU".
	if gpus[1].UsedMB != 0 || gpus[1].TotalMB != 0 {
		t.Errorf("gpu 1 memory = %d/%d, want 0/0 for a device with no sample", gpus[1].UsedMB, gpus[1].TotalMB)
	}
}

// TestGpuList_MemoryKeyedByDeviceIndexNotDisplayIndex is the bug this merge
// could easily have: the accelerator's display position and its real device
// index differ, and the memory sample is keyed by the device index.
func TestGpuList_MemoryKeyedByDeviceIndexNotDisplayIndex(t *testing.T) {
	hardware := gpuTestSnapshot(hw.Accelerator{
		Index: 0, DeviceIndex: gpuTestIntPtr(3), Kind: "gpu",
		Vendor: gpuTestStrPtr("NVIDIA"), Model: gpuTestStrPtr("RTX 4090"),
	})
	memory := map[int]perf.GpuStat{
		0: {ID: 0, MemUsedMB: 1, MemTotalMB: 2},         // display index — wrong
		3: {ID: 3, MemUsedMB: 12000, MemTotalMB: 24000}, // device index — right
	}

	gpus := gpuList(hardware, memory)
	if len(gpus) != 1 {
		t.Fatalf("got %d gpus, want 1", len(gpus))
	}
	if gpus[0].UsedMB != 12000 {
		t.Errorf("UsedMB = %d, want 12000; memory must be keyed by the device index", gpus[0].UsedMB)
	}
}

// TestGpuList_NoPerfMonitorOmitsMemory: with monitoring off the fields stay
// zero and omitempty drops them from the JSON entirely.
func TestGpuList_NoPerfMonitorOmitsMemory(t *testing.T) {
	hardware := gpuTestSnapshot(hw.Accelerator{
		Index: 0, DeviceIndex: gpuTestIntPtr(0), Kind: "gpu",
		Vendor: gpuTestStrPtr("NVIDIA"), Model: gpuTestStrPtr("RTX 4090"),
	})

	gpus := gpuList(hardware, newestGPUSamples(nil))
	body, err := json.Marshal(gpus)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got := string(body); strings.Contains(got, "usedMB") || strings.Contains(got, "totalMB") {
		t.Errorf("payload = %s; memory keys must be omitted when there is no reading", got)
	}
}

// TestNewestPerGPU keeps the latest sample per device, whatever order the ring
// hands them over in.
func TestNewestPerGPU(t *testing.T) {
	older := time.Date(2026, 9, 13, 10, 0, 0, 0, time.UTC)
	newer := older.Add(time.Minute)

	got := newestPerGPU([]perf.GpuStat{
		{ID: 0, Timestamp: older, MemUsedMB: 100},
		{ID: 1, Timestamp: newer, MemUsedMB: 900},
		{ID: 0, Timestamp: newer, MemUsedMB: 500},
		{ID: 1, Timestamp: older, MemUsedMB: 111}, // out of order: must not win
	})

	if got[0].MemUsedMB != 500 {
		t.Errorf("gpu 0 = %d MB, want the newest sample (500)", got[0].MemUsedMB)
	}
	if got[1].MemUsedMB != 900 {
		t.Errorf("gpu 1 = %d MB, want the newest sample (900) despite arriving first", got[1].MemUsedMB)
	}
	if n := len(newestPerGPU(nil)); n != 0 {
		t.Errorf("newestPerGPU(nil) has %d entries, want 0", n)
	}
}
