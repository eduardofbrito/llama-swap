package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// TestBaseRouter_ServeHTTP_FlipsGpuOverrideToProcess proves the router
// forwards a request-scoped GPU override from the request context into the
// start options: a load triggered by a request carrying GpuOverride must
// start the process through EnsureReadyWithOptions with that value.
func TestBaseRouter_ServeHTTP_FlipsGpuOverrideToProcess(t *testing.T) {
	a := newFakeProcess("a")
	a.autoReady = true

	b := newTestBase(t, map[string]process.Process{"a": a}, &stubPlanner{})

	r := newRequest("a")
	r = r.WithContext(swaputil.SetContext(r.Context(), swaputil.ReqContextData{
		Model:       "a",
		ModelID:     "a",
		GpuOverride: "3",
	}))

	w := httptest.NewRecorder()
	b.ServeHTTP(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if got := a.optsCalls.Load(); got != 1 {
		t.Fatalf("optsCalls=%d want 1 (a GPU override must start the process via WithOptions)", got)
	}
	if got := a.lastOpts().GpuOverride; got != "3" {
		t.Errorf("GpuOverride=%q want %q", got, "3")
	}
}

// TestBaseRouter_ServeHTTP_NoGpuOverridePassesEmptyOptions proves the
// default path: without a GPU override the start options carry an empty
// GpuOverride, so the model's configured GPU applies. (The router still
// uses the WithOptions variant when the process implements it; empty
// Options is a no-op there.)
func TestBaseRouter_ServeHTTP_NoGpuOverridePassesEmptyOptions(t *testing.T) {
	a := newFakeProcess("a")
	a.autoReady = true

	b := newTestBase(t, map[string]process.Process{"a": a}, &stubPlanner{})

	w := httptest.NewRecorder()
	b.ServeHTTP(w, newRequest("a"))

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if got := a.lastOpts().GpuOverride; got != "" {
		t.Errorf("GpuOverride=%q want empty (no override: config default expected)", got)
	}
	if got := a.runCalls.Load(); got != 1 {
		t.Errorf("runCalls=%d want 1", got)
	}
}
