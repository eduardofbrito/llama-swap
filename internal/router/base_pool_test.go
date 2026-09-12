package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// poolConfig builds a router config with the recent-model pool sized to n.
func poolConfig(n int) config.Config {
	conf := config.Config{HealthCheckTimeout: 5}
	conf.Routing.Scheduler.Settings.Fifo.RecentPoolSize = n
	return conf
}

// serve drives one request through the router and fails the test when it does
// not succeed, so a test that means to exercise eviction cannot pass by
// accident on an error path.
func serve(t *testing.T, b *baseRouter, model string) {
	t.Helper()
	w := httptest.NewRecorder()
	b.ServeHTTP(w, newRequest(model))
	if w.Code != http.StatusOK {
		t.Fatalf("serving %s: status=%d body=%q", model, w.Code, w.Body.String())
	}
}

// TestBaseRouter_RecentPoolKeepsMostRecentModels is the whole point of the
// feature: with a pool of 2, loading a third model unloads only the least
// recently used one instead of every other model.
func TestBaseRouter_RecentPoolKeepsMostRecentModels(t *testing.T) {
	a, bProc, c := newFakeProcess("a"), newFakeProcess("b"), newFakeProcess("c")
	for _, p := range []*fakeProcess{a, bProc, c} {
		p.autoReady = true
	}
	// The swapper wants exclusive use: loading any model evicts both others.
	planner := &stubPlanner{evict: map[string][]string{
		"a": {"b", "c"},
		"b": {"a", "c"},
		"c": {"a", "b"},
	}}
	b := newTestBaseWithConfig(t, poolConfig(2),
		map[string]process.Process{"a": a, "b": bProc, "c": c}, planner)

	serve(t, b, "a")
	serve(t, b, "b") // a is now the pool's other member: it must survive
	serve(t, b, "c") // pool is {c, b}: a is the LRU and is the one to go

	if got := a.stopCalls.Load(); got == 0 {
		t.Error("a was never stopped; the least recently used model must be evicted")
	}
	// b was the most recent before c, so the pool held it back on the last swap.
	if got := bProc.stopCalls.Load(); got > 1 {
		t.Errorf("b stopped %d times; the pool must hold back the most recent model", got)
	}
}

// TestBaseRouter_RecentPoolNeverEvictsBeyondSwapper pins the safety property:
// poolEviction only ever removes IDs from the swapper's evict set. A model the
// swapper never nominated must never be stopped because of the pool.
func TestBaseRouter_RecentPoolNeverEvictsBeyondSwapper(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	// The swapper nominates nothing: both models coexist.
	b := newTestBaseWithConfig(t, poolConfig(1),
		map[string]process.Process{"a": a, "b": bProc}, &stubPlanner{})

	serve(t, b, "a")
	serve(t, b, "b")
	serve(t, b, "a")

	if got := bProc.stopCalls.Load(); got != 0 {
		t.Errorf("b stopped %d times; the pool must not evict what the swapper kept", got)
	}
	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a stopped %d times; the pool must not evict what the swapper kept", got)
	}
}

// TestBaseRouter_RecentPoolDisabledLeavesSwapperAlone checks the default: with
// the pool off, every eviction the swapper asks for still happens.
func TestBaseRouter_RecentPoolDisabledLeavesSwapperAlone(t *testing.T) {
	a, bProc := newFakeProcess("a"), newFakeProcess("b")
	a.autoReady, bProc.autoReady = true, true
	planner := &stubPlanner{evict: map[string][]string{"b": {"a"}}}

	for _, size := range []int{0, 1} {
		a.stopCalls.Store(0)
		b := newTestBaseWithConfig(t, poolConfig(size),
			map[string]process.Process{"a": a, "b": bProc}, planner)

		serve(t, b, "a")
		serve(t, b, "b")

		if got := a.stopCalls.Load(); got == 0 {
			t.Errorf("recentPoolSize=%d: a was not evicted; sizes 0 and 1 must disable the pool", size)
		}
	}
}
