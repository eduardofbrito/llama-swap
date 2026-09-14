package process

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

// sleepTestProcess starts a sleep-capable upstream and returns it ready to
// serve. extraArgs is appended to the responder's command line.
func sleepTestProcess(t *testing.T, extraArgs ...string) (*ProcessCommand, int) {
	t.Helper()
	skipIfNoSimpleResponder(t)

	args := append([]string{"-silent"}, extraArgs...)
	cmd, port := simpleResponderCmd(t, args...)
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	ctx, cancel := context.WithTimeout(context.Background(), testStartTimeout)
	defer cancel()
	if err := p.EnsureReady(ctx, testStartTimeout); err != nil {
		t.Fatalf("EnsureReady: %v", err)
	}
	return p, port
}

// upstreamIsSleeping asks the upstream itself, rather than trusting
// llama-swap's own state, whether it was actually put to sleep.
func upstreamIsSleeping(t *testing.T, port int) bool {
	t.Helper()
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/is_sleeping", port))
	if err != nil {
		t.Fatalf("GET /is_sleeping: %v", err)
	}
	defer resp.Body.Close()
	var body struct {
		IsSleeping bool `json:"is_sleeping"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decoding /is_sleeping: %v", err)
	}
	return body.IsSleeping
}

// TestProcessCommand_SleepAndWake is the round trip: a ready process sleeps
// (upstream told, state reflects it, traffic refused), and the next
// EnsureReady wakes it back into service — all without the process exiting,
// which is the entire point of sleeping instead of stopping.
func TestProcessCommand_SleepAndWake(t *testing.T) {
	p, port := sleepTestProcess(t)

	if err := p.Sleep(testStopTimeout); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if got := p.State(); got != StateSleeping {
		t.Errorf("state = %s, want %s", got, StateSleeping)
	}
	if !upstreamIsSleeping(t, port) {
		t.Error("upstream was never asked to sleep")
	}

	// A sleeping model holds no weights on the GPU, so forwarding to it would
	// be forwarding into a model that cannot answer.
	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("ServeHTTP while sleeping = %d, want 503", w.Code)
	}

	ctx, cancel := context.WithTimeout(context.Background(), testStartTimeout)
	defer cancel()
	if err := p.EnsureReady(ctx, testStartTimeout); err != nil {
		t.Fatalf("EnsureReady (wake): %v", err)
	}
	if got := p.State(); got != StateReady {
		t.Errorf("state after wake = %s, want %s", got, StateReady)
	}
	if upstreamIsSleeping(t, port) {
		t.Error("upstream was never woken")
	}

	// The handler has to come back too — a woken model that 503s is no better
	// than a sleeping one.
	w = httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/test", nil))
	if w.Code != http.StatusOK {
		t.Errorf("ServeHTTP after wake = %d, want 200", w.Code)
	}
}

// TestProcessCommand_SleepKeepsProcessAlive pins the property the whole
// feature rests on: sleeping must not restart anything. If the upstream were
// killed and relaunched, the wake would cost a cold start and the port would
// have been given up in between.
func TestProcessCommand_SleepKeepsProcessAlive(t *testing.T) {
	p, port := sleepTestProcess(t)

	if err := p.Sleep(testStopTimeout); err != nil {
		t.Fatalf("Sleep: %v", err)
	}

	// The upstream answering at all proves the same process is still listening.
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
	if err != nil {
		t.Fatalf("upstream stopped listening while asleep: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("upstream /health while asleep = %d, want 200", resp.StatusCode)
	}
}

// TestProcessCommand_SleepIsIdempotent covers the router evicting from a
// stale snapshot: a model can be nominated for eviction while already asleep,
// and answering with an error there would send the router down its "sleep
// failed, stop it instead" fallback and kill a model holding no GPU memory.
func TestProcessCommand_SleepIsIdempotent(t *testing.T) {
	p, _ := sleepTestProcess(t)

	if err := p.Sleep(testStopTimeout); err != nil {
		t.Fatalf("first Sleep: %v", err)
	}
	if err := p.Sleep(testStopTimeout); err != nil {
		t.Errorf("second Sleep: %v, want nil (already sleeping is a no-op)", err)
	}
	if got := p.State(); got != StateSleeping {
		t.Errorf("state = %s, want %s", got, StateSleeping)
	}
}

// TestProcessCommand_SleepRejectedWhenStopped: there is nothing to release
// when no process is running, and claiming success would let the router
// believe it had freed memory it had not.
func TestProcessCommand_SleepRejectedWhenStopped(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	if err := p.Sleep(testStopTimeout); err == nil {
		t.Error("Sleep on a stopped process returned nil, want an error")
	}
	if got := p.State(); got != StateStopped {
		t.Errorf("state = %s, want %s", got, StateStopped)
	}
}

// TestProcessCommand_SleepWithoutUpstreamSupport is the misconfiguration that
// will actually happen: sleepMode set on a model whose upstream is not a vLLM
// with sleep mode enabled. Sleep must fail and leave the model serving, so the
// router's fallback to stopping it is the only thing that changes.
func TestProcessCommand_SleepWithoutUpstreamSupport(t *testing.T) {
	p, _ := sleepTestProcess(t, "-no-sleep-mode")

	err := p.Sleep(testStopTimeout)
	if err == nil {
		t.Fatal("Sleep against an upstream without sleep mode returned nil, want an error")
	}

	if got := p.State(); got != StateReady {
		t.Errorf("state = %s, want %s — a failed sleep must not change state", got, StateReady)
	}
	// The handler must be back: a failed sleep that left the model unable to
	// answer would take it out of service for no reason.
	w := httptest.NewRecorder()
	p.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/test", nil))
	if w.Code != http.StatusOK {
		t.Errorf("ServeHTTP after a failed sleep = %d, want 200", w.Code)
	}
}

// TestProcessCommand_StopWhileSleeping: an explicit unload has to terminate a
// sleeping process, otherwise a model nobody asked for keeps its weights in
// host RAM forever.
func TestProcessCommand_StopWhileSleeping(t *testing.T) {
	p, port := sleepTestProcess(t)

	if err := p.Sleep(testStopTimeout); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if err := p.Stop(testStopTimeout); err != nil {
		t.Fatalf("Stop while sleeping: %v", err)
	}
	if got := p.State(); got != StateStopped {
		t.Errorf("state = %s, want %s", got, StateStopped)
	}

	// The listener must be gone — a surviving process would still be holding
	// the RAM (and the port).
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err != nil {
			return // refused: the process is gone, which is what we want
		}
		resp.Body.Close()
		time.Sleep(testPollInterval)
	}
	t.Error("upstream still listening after Stop")
}
