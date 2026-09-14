package router

import (
	"fmt"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// sleepConfig builds a router config where the named models are configured
// with sleepMode.
func sleepConfig(sleepers ...string) config.Config {
	conf := config.Config{HealthCheckTimeout: 5, Models: map[string]config.ModelConfig{}}
	for _, id := range []string{"a", "b"} {
		conf.Models[id] = config.ModelConfig{}
	}
	for _, id := range sleepers {
		conf.Models[id] = config.ModelConfig{SleepMode: true}
	}
	return conf
}

// exclusivePlanner makes loading either model evict the other, so every swap
// is a real eviction.
func exclusivePlanner() *stubPlanner {
	return &stubPlanner{evict: map[string][]string{"a": {"b"}, "b": {"a"}}}
}

// TestBaseRouter_SleepModeEvictsBySleeping is the feature: a model evicted to
// make room for another is put to sleep rather than killed, so the next time
// it is asked for it wakes instead of cold-starting.
func TestBaseRouter_SleepModeEvictsBySleeping(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	b := newTestBaseWithConfig(t, sleepConfig("a", "b"),
		map[string]process.Process{"a": a, "b": bProc}, exclusivePlanner())

	serve(t, b, "a")
	serve(t, b, "b") // evicts a

	if got := a.sleepCalls.Load(); got != 1 {
		t.Errorf("a slept %d times, want 1", got)
	}
	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a was stopped %d times; a sleepMode model must not be killed to make room", got)
	}
	if got := a.State(); got != process.StateSleeping {
		t.Errorf("a state = %s, want %s", got, process.StateSleeping)
	}
}

// TestBaseRouter_SleepModeWakesOnNextRequest closes the loop: the model that
// was slept comes back into service, and the model that displaced it is the
// one that sleeps now.
func TestBaseRouter_SleepModeWakesOnNextRequest(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	b := newTestBaseWithConfig(t, sleepConfig("a", "b"),
		map[string]process.Process{"a": a, "b": bProc}, exclusivePlanner())

	serve(t, b, "a")
	serve(t, b, "b") // a sleeps
	serve(t, b, "a") // a must wake, b must sleep

	if got := a.State(); got != process.StateReady {
		t.Errorf("a state = %s, want %s — the sleeping model must wake to serve", got, process.StateReady)
	}
	if got := bProc.State(); got != process.StateSleeping {
		t.Errorf("b state = %s, want %s", got, process.StateSleeping)
	}
	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a was stopped %d times; waking must never involve a restart", got)
	}
}

// TestBaseRouter_SleepModeOffStopsAsBefore pins that this is opt-in: without
// sleepMode the eviction path is exactly what it was.
func TestBaseRouter_SleepModeOffStopsAsBefore(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	b := newTestBaseWithConfig(t, sleepConfig(), // neither model sleeps
		map[string]process.Process{"a": a, "b": bProc}, exclusivePlanner())

	serve(t, b, "a")
	serve(t, b, "b")

	if got := a.sleepCalls.Load(); got != 0 {
		t.Errorf("a slept %d times; sleeping must be opt-in", got)
	}
	if got := a.stopCalls.Load(); got == 0 {
		t.Error("a was never stopped; eviction without sleepMode must still stop the model")
	}
}

// TestBaseRouter_SleepFailureFallsBackToStop is the safety property. The swap
// that triggered the eviction is about to load another model onto the same
// device, so a model whose sleep failed must not be left holding its VRAM —
// better a slow swap than a failed one.
func TestBaseRouter_SleepFailureFallsBackToStop(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	// The realistic cause: the upstream is not a vLLM with sleep mode enabled,
	// so /sleep answers 404.
	a.sleepErr = fmt.Errorf("sleep returned 404")
	b := newTestBaseWithConfig(t, sleepConfig("a", "b"),
		map[string]process.Process{"a": a, "b": bProc}, exclusivePlanner())

	serve(t, b, "a")
	serve(t, b, "b")

	if got := a.sleepCalls.Load(); got != 1 {
		t.Errorf("a slept %d times, want 1 attempt", got)
	}
	if got := a.stopCalls.Load(); got == 0 {
		t.Error("a was never stopped after its sleep failed; its VRAM would never be released")
	}
}

// TestBaseRouter_UnloadStopsSleepModeModel separates eviction from an explicit
// unload. Unload means the operator wants the model gone; a process left alive
// holding its weights in host RAM would not be gone.
func TestBaseRouter_UnloadStopsSleepModeModel(t *testing.T) {
	a := newFakeProcess("a")
	a.autoReady = true
	b := newTestBaseWithConfig(t, sleepConfig("a"),
		map[string]process.Process{"a": a}, &stubPlanner{})

	serve(t, b, "a")
	b.Unload(time.Second, "a")

	if got := a.sleepCalls.Load(); got != 0 {
		t.Errorf("a slept %d times; an explicit unload must not sleep the model", got)
	}
	if got := a.stopCalls.Load(); got == 0 {
		t.Error("a was never stopped by Unload")
	}
	if got := a.State(); got != process.StateStopped {
		t.Errorf("a state = %s, want %s", got, process.StateStopped)
	}
}
