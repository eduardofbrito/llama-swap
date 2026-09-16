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

// capConfig builds a config where every model sleeps on eviction, pinned to
// one device, with the per-GPU sleeper cap set to n.
func capConfig(n int, device string, models ...string) config.Config {
	conf := config.Config{HealthCheckTimeout: 5, Models: map[string]config.ModelConfig{}}
	for _, id := range models {
		conf.Models[id] = config.ModelConfig{
			SleepMode: true,
			Env:       []string{"CUDA_VISIBLE_DEVICES=" + device},
		}
	}
	conf.Routing.Scheduler.Settings.Fifo.MaxSleepingPerGPU = n
	return conf
}

// TestBaseRouter_SleeperCapStopsTheLongestAsleep is the point of the cap:
// sleeping does not free the whole card (the process keeps its CUDA context),
// and a sleeping model is never evicted again, so without a bound the residue
// accumulates until the resident model can no longer get the fraction of the
// card it asks for.
func TestBaseRouter_SleeperCapStopsTheLongestAsleep(t *testing.T) {
	a, bProc, c := newFakeProcess("a"), newFakeProcess("b"), newFakeProcess("c")
	for _, p := range []*fakeProcess{a, bProc, c} {
		p.autoReady = true
	}
	// Everything evicts everything, so each request parks the previous model.
	planner := &stubPlanner{evict: map[string][]string{
		"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"},
	}}
	r := newTestBaseWithConfig(t, capConfig(1, "0", "a", "b", "c"),
		map[string]process.Process{"a": a, "b": bProc, "c": c}, planner)

	serve(t, r, "a")
	serve(t, r, "b") // a sleeps: device 0 now has one sleeper, at the cap
	if got := a.State(); got != process.StateSleeping {
		t.Fatalf("a state = %s, want %s", got, process.StateSleeping)
	}

	serve(t, r, "c") // b wants to sleep too; a is the longest asleep and goes

	if got := a.State(); got != process.StateStopped {
		t.Errorf("a state = %s, want %s; the longest-asleep model must be stopped to stay within the cap", got, process.StateStopped)
	}
	if got := bProc.State(); got != process.StateSleeping {
		t.Errorf("b state = %s, want %s; the most recent sleeper must be kept", got, process.StateSleeping)
	}
}

// TestBaseRouter_SleeperCapOffKeepsEveryoneAsleep pins that the cap is opt-in:
// 0 (the default) means no bound, which is how sleepMode behaved before it
// existed.
func TestBaseRouter_SleeperCapOffKeepsEveryoneAsleep(t *testing.T) {
	a, bProc, c := newFakeProcess("a"), newFakeProcess("b"), newFakeProcess("c")
	for _, p := range []*fakeProcess{a, bProc, c} {
		p.autoReady = true
	}
	planner := &stubPlanner{evict: map[string][]string{
		"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"},
	}}
	r := newTestBaseWithConfig(t, capConfig(0, "0", "a", "b", "c"),
		map[string]process.Process{"a": a, "b": bProc, "c": c}, planner)

	serve(t, r, "a")
	serve(t, r, "b")
	serve(t, r, "c")

	if got := a.State(); got != process.StateSleeping {
		t.Errorf("a state = %s, want %s; with no cap every evicted model stays asleep", got, process.StateSleeping)
	}
	if got := bProc.State(); got != process.StateSleeping {
		t.Errorf("b state = %s, want %s", got, process.StateSleeping)
	}
}

// TestBaseRouter_SleeperCapIsPerDevice: the cap counts sleepers on ONE card.
// Models parked on a different GPU are not competing for the same memory and
// must not be stopped to make room.
func TestBaseRouter_SleeperCapIsPerDevice(t *testing.T) {
	a, bProc, c := newFakeProcess("a"), newFakeProcess("b"), newFakeProcess("c")
	for _, p := range []*fakeProcess{a, bProc, c} {
		p.autoReady = true
	}
	conf := capConfig(1, "0", "b", "c")    // b and c on device 0
	conf.Models["a"] = config.ModelConfig{ // a on device 1
		SleepMode: true,
		Env:       []string{"CUDA_VISIBLE_DEVICES=1"},
	}
	planner := &stubPlanner{evict: map[string][]string{
		"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"},
	}}
	r := newTestBaseWithConfig(t, conf,
		map[string]process.Process{"a": a, "b": bProc, "c": c}, planner)

	serve(t, r, "a")
	serve(t, r, "b") // a sleeps on device 1
	serve(t, r, "c") // b sleeps on device 0 — a different card, so a stays

	if got := a.State(); got != process.StateSleeping {
		t.Errorf("a state = %s, want %s; a sleeper on another GPU must not be stopped", got, process.StateSleeping)
	}
	if got := bProc.State(); got != process.StateSleeping {
		t.Errorf("b state = %s, want %s; it is the only sleeper on device 0", got, process.StateSleeping)
	}
}

// TestBaseRouter_SleeperCapSparesTheModelBeingWoken reproduces what production
// did on the first swap back to a sleeping model:
//
//	<qwen3.6-35b-a3b> slept in 21.311s — GPU memory released, process kept alive
//	group: device 0 is over the sleeper cap (1 max); stopping soberano-v1.1,
//	       asleep the longest, to free its share of the card
//	group: evicted [qwen3.6-35b-a3b] in 26.8s, now loading soberano-v1.1
//	group: swapped to soberano-v1.1 in 1m11.098s (evictions 26.8s, load 44.297s)
//
// The cap stopped the very model the swap was waking. It is the natural victim
// — waking it is what freed the card the newly evicted model just took, so it
// is reliably the longest-asleep sleeper there — and it is also the one model
// that is about to stop being a sleeper at all. Counting it made the swap
// throw away a one-second wake and pay a 44-second cold start instead, every
// single time the group came back around.
func TestBaseRouter_SleeperCapSparesTheModelBeingWoken(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	planner := &stubPlanner{evict: map[string][]string{"a": {"b"}, "b": {"a"}}}
	r := newTestBaseWithConfig(t, capConfig(1, "0", "a", "b"),
		map[string]process.Process{"a": a, "b": bProc}, planner)

	serve(t, r, "a")
	serve(t, r, "b") // a sleeps on device 0 — one sleeper, at the cap
	if got := a.State(); got != process.StateSleeping {
		t.Fatalf("a state = %s, want %s", got, process.StateSleeping)
	}
	stopsBefore := a.stopCalls.Load()

	serve(t, r, "a") // b sleeps to make room; a must wake, not be stopped

	if got := a.State(); got != process.StateReady {
		t.Errorf("a state = %s, want %s", got, process.StateReady)
	}
	if got := a.stopCalls.Load(); got != stopsBefore {
		t.Errorf("a was stopped %d more time(s) while being woken; the cap must "+
			"never pick the model the swap is loading — that turns a wake into "+
			"a cold start", got-stopsBefore)
	}
	if got := bProc.State(); got != process.StateSleeping {
		t.Errorf("b state = %s, want %s; it is within the cap once a is awake", got, process.StateSleeping)
	}
}
