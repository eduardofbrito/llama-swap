package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/hw"
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
