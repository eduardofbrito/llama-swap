package scheduler

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// FIFO methods all run on the router's single run-loop goroutine, so these
// tests drive them directly and synchronously. A swap is "completed" by calling
// OnSwapDone, a served request "finishes" by calling OnServeDone — exactly the
// events the run loop would deliver. fakeEffects records every side-effect and
// stubPlanner supplies a fixed eviction set per target.

// stubPlanner returns a fixed eviction list per target.
type stubPlanner struct {
	evict map[string][]string
}

func (s *stubPlanner) EvictionFor(target string, _ []string) []string {
	if s.evict == nil {
		return nil
	}
	return s.evict[target]
}

func (s *stubPlanner) OnSwapStart(string, []string) {}

// grantRec is one GrantError / GrantServe call. err!=nil marks an error grant;
// otherwise it is a serve grant and serve reports whether the caller received it.
type grantRec struct {
	model string
	err   error
	serve bool
}

type startRec struct {
	model string
	evict []string
	gpu   string
}

type stopRec struct {
	timeout time.Duration
	ids     []string
}

// fakeEffects is an in-memory scheduler.Effects. Tests program process states
// and GrantServe outcomes, then assert on the recorded calls.
type fakeEffects struct {
	states       map[string]process.ProcessState // model -> state; missing => not handled
	manual       map[string]bool                 // model -> manualOnly (default false)
	serveResult  map[string]bool                 // GrantServe return per model (default true)
	lastServeReq HandlerReq

	// vramShortfall is the MB the fake reports missing per model; absent means
	// there is room, which is the default so existing tests are unaffected.
	vramShortfall map[string]int
	// vramShortfallFor, when set, overrides the table and lets a test answer
	// differently depending on the eviction set it is handed.
	vramShortfallFor func(modelID string, evict []string) int
	vramCalls        []string

	starts []startRec
	grants []grantRec
	stops  []stopRec
}

func newFakeEffects() *fakeEffects {
	return &fakeEffects{
		states:      map[string]process.ProcessState{},
		manual:      map[string]bool{},
		serveResult: map[string]bool{},

		vramShortfall: map[string]int{},
	}
}

func (f *fakeEffects) VRAMShortfall(modelID, device string, evict []string) int {
	f.vramCalls = append(f.vramCalls, modelID)
	if f.vramShortfallFor != nil {
		return f.vramShortfallFor(modelID, evict)
	}
	return f.vramShortfall[modelID]
}

func (f *fakeEffects) ModelState(modelID string) (process.ProcessState, bool) {
	st, ok := f.states[modelID]
	return st, ok
}

func (f *fakeEffects) ModelManual(modelID string) (bool, bool) {
	if f.manual == nil {
		return false, false
	}
	m, ok := f.manual[modelID]
	return m, ok
}

func (f *fakeEffects) RunningModels() map[string]process.ProcessState {
	out := make(map[string]process.ProcessState)
	for id, st := range f.states {
		if st == process.StateStopped || st == process.StateShutdown {
			continue
		}
		out[id] = st
	}
	return out
}

func (f *fakeEffects) StartSwap(modelID string, evict []string, opts process.Options) {
	f.starts = append(f.starts, startRec{model: modelID, evict: evict, gpu: opts.GpuOverride})
}

func (f *fakeEffects) GrantError(req HandlerReq, err error) {
	f.grants = append(f.grants, grantRec{model: req.Model, err: err})
}

func (f *fakeEffects) GrantServe(req HandlerReq, modelID string) bool {
	ok := true
	if v, set := f.serveResult[modelID]; set {
		ok = v
	}
	f.lastServeReq = req
	f.grants = append(f.grants, grantRec{model: modelID, serve: ok})
	return ok
}

func (f *fakeEffects) StopProcesses(timeout time.Duration, ids []string) {
	f.stops = append(f.stops, stopRec{timeout: timeout, ids: ids})
}

// served counts grants that handed modelID a handler and were received.
func (f *fakeEffects) served(modelID string) int {
	n := 0
	for _, g := range f.grants {
		if g.err == nil && g.serve && g.model == modelID {
			n++
		}
	}
	return n
}

// errored counts error grants, optionally filtered by model ("" = any).
func (f *fakeEffects) errored(model string) int {
	n := 0
	for _, g := range f.grants {
		if g.err != nil && (model == "" || g.model == model) {
			n++
		}
	}
	return n
}

// startsFor counts StartSwap calls for modelID.
func (f *fakeEffects) startsFor(modelID string) int {
	n := 0
	for _, s := range f.starts {
		if s.model == modelID {
			n++
		}
	}
	return n
}

func newFIFO(planner Swapper, eff Effects) *FIFO {
	return NewFIFO("test", logmon.NewWriter(io.Discard), planner, config.FifoConfig{}, nil, eff)
}

func req(model string) HandlerReq {
	return HandlerReq{
		Model: model,
		Ctx:   context.Background(),
		Admit: make(chan error, 1),
	}
}

// reqCh creates a HandlerReq with a unique Respond channel so OnCancel can
// identify it among queued requests and swap waiters.
func reqCh(model string) HandlerReq {
	r := req(model)
	r.Respond = make(chan HandlerResp, 1)
	return r
}

func admitErr(t *testing.T, req HandlerReq) error {
	t.Helper()
	select {
	case err := <-req.Admit:
		return err
	default:
		t.Fatal("admission result not sent")
		return nil
	}
}

func assertAdmitted(t *testing.T, req HandlerReq) {
	t.Helper()
	if err := admitErr(t, req); err != nil {
		t.Fatalf("admission err=%v want nil", err)
	}
}

func assertAdmission429(t *testing.T, req HandlerReq) {
	t.Helper()
	var httpErr swaputil.HTTPError
	err := admitErr(t, req)
	if !errors.As(err, &httpErr) {
		t.Fatalf("admission err=%v want HTTPError", err)
	}
	if httpErr.StatusCode() != http.StatusTooManyRequests {
		t.Fatalf("StatusCode()=%d want 429", httpErr.StatusCode())
	}
	if httpErr.Header().Get("Retry-After") == "" {
		t.Fatal("missing Retry-After header")
	}
}

func TestFIFO_SendAdmission_CancelledContextWins(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := HandlerReq{
		Ctx:   ctx,
		Admit: make(chan error, 1),
	}

	if sendAdmission(r, nil) {
		t.Fatal("sendAdmission returned true for cancelled request")
	}
	select {
	case err := <-r.Admit:
		t.Fatalf("admission sent after cancellation: %v", err)
	default:
	}
}

func TestFIFO_SendAdmission_NilAdmitPreservesExistingBehavior(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if !sendAdmission(HandlerReq{Ctx: ctx}, nil) {
		t.Fatal("nil Admit should preserve existing accepted behavior")
	}
}

func TestFIFO_ReleaseWithoutReservationPanics(t *testing.T) {
	s := newFIFO(&stubPlanner{}, newFakeEffects())

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("release without reservation did not panic")
		}
	}()
	s.release("a")
}

func TestFIFO_FastPath(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateReady
	s := newFIFO(&stubPlanner{}, eff)

	s.OnRequest(req("a"))

	if got := eff.startsFor("a"); got != 0 {
		t.Errorf("StartSwap calls=%d want 0 (fast path should not swap)", got)
	}
	if got := eff.served("a"); got != 1 {
		t.Errorf("served(a)=%d want 1", got)
	}
}

func TestFIFO_GrantSetsPriorityMetadata(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateReady
	cfg := config.FifoConfig{Priority: map[string]int{"a": 7}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{}, cfg, nil, eff)

	ctx := swaputil.SetContext(context.Background(), swaputil.ReqContextData{ModelID: "a", Metadata: make(map[string]string)})
	s.OnRequest(HandlerReq{Model: "a", Ctx: ctx})

	if got := eff.served("a"); got != 1 {
		t.Fatalf("served(a)=%d want 1", got)
	}
	data, ok := swaputil.ReadContext(eff.lastServeReq.Ctx)
	if !ok {
		t.Fatal("context data missing from granted request")
	}
	if data.Metadata["fifo_priority"] != "7" {
		t.Errorf("fifo_priority = %q, want 7", data.Metadata["fifo_priority"])
	}
}

func TestFIFO_ModelNotFound(t *testing.T) {
	eff := newFakeEffects() // no states => model unknown
	s := newFIFO(&stubPlanner{}, eff)

	r := req("ghost")
	s.OnRequest(r)

	if got := len(eff.starts); got != 0 {
		t.Errorf("StartSwap calls=%d want 0", got)
	}
	if got := eff.errored("ghost"); got != 0 {
		t.Fatalf("error grants=%d want 0 for admission rejection", got)
	}
	if err := admitErr(t, r); !errors.Is(err, ErrModelNotFound) {
		t.Errorf("admission err=%v want ErrModelNotFound", err)
	}
}

func TestFIFO_OnDemandStartThenServe(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	s := newFIFO(&stubPlanner{}, eff)

	s.OnRequest(req("a"))
	if got := eff.startsFor("a"); got != 1 {
		t.Fatalf("StartSwap(a)=%d want 1", got)
	}
	if got := eff.served("a"); got != 0 {
		t.Errorf("served(a)=%d want 0 before swap completes", got)
	}

	// Swap finishes, model is now ready.
	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})

	if got := eff.served("a"); got != 1 {
		t.Errorf("served(a)=%d want 1 after swap done", got)
	}
}

func TestFIFO_JoinInFlightSwap(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	s := newFIFO(&stubPlanner{}, eff)

	s.OnRequest(req("a")) // starts swap
	s.OnRequest(req("a")) // joins
	s.OnRequest(req("a")) // joins

	if got := eff.startsFor("a"); got != 1 {
		t.Fatalf("StartSwap(a)=%d want 1 (all three share one swap)", got)
	}

	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})

	if got := eff.served("a"); got != 3 {
		t.Errorf("served(a)=%d want 3 (one swap serves all waiters)", got)
	}
}

func TestFIFO_SwapDoneError_FailsAllWaiters(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	s := newFIFO(&stubPlanner{}, eff)

	s.OnRequest(req("a"))
	s.OnRequest(req("a"))

	s.OnSwapDone(SwapDone{ModelID: "a", Err: errors.New("boom")})

	if eff.served("a") != 0 {
		t.Errorf("served(a)=%d want 0 on swap error", eff.served("a"))
	}
	if eff.errored("a") != 2 {
		t.Errorf("errored(a)=%d want 2 (both waiters fail)", eff.errored("a"))
	}
}

// TestFIFO_QueueOnEvictionCollision covers a request whose target evicts the
// model currently being swapped: it must queue until that swap finishes AND its
// served request drains, because starting it would stop a busy process.
func TestFIFO_QueueOnEvictionCollision(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	// Loading b evicts a.
	s := newFIFO(&stubPlanner{evict: map[string][]string{"b": {"a"}}}, eff)

	s.OnRequest(req("a")) // StartSwap(a)
	s.OnRequest(req("b")) // collides with a's in-flight swap -> queue
	if got := eff.startsFor("b"); got != 0 {
		t.Fatalf("b started early: StartSwap(b)=%d want 0", got)
	}

	// a becomes ready and is granted (now serving, inFlight[a]=1).
	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})
	if got := eff.startsFor("b"); got != 0 {
		t.Fatalf("b started while a is serving: StartSwap(b)=%d want 0", got)
	}

	// a's request finishes -> a no longer in-flight -> b may now swap.
	s.OnServeDone(ServeDoneEvent{ModelID: "a"})
	if got := eff.startsFor("b"); got != 1 {
		t.Fatalf("StartSwap(b)=%d want 1 after a drained", got)
	}
	if got := eff.starts[len(eff.starts)-1].evict; len(got) != 1 || got[0] != "a" {
		t.Errorf("b swap evict=%v want [a]", got)
	}
}

// TestFIFO_DisjointSwapsRunInParallel verifies two requests with
// non-conflicting evict sets both start without waiting for each other.
func TestFIFO_DisjointSwapsRunInParallel(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	s := newFIFO(&stubPlanner{}, eff) // empty evicts

	s.OnRequest(req("a"))
	s.OnRequest(req("b"))

	if eff.startsFor("a") != 1 || eff.startsFor("b") != 1 {
		t.Fatalf("StartSwap a=%d b=%d want 1 each (parallel)", eff.startsFor("a"), eff.startsFor("b"))
	}
}

// TestFIFO_OverlappingEvictSetsDoNotRunInParallel verifies two swaps with
// different targets that evict the *same* model do not run concurrently: the
// second must queue rather than double-evict the shared model. Neither target is
// in the other's evict set, so this is only caught by the evict-set overlap
// check in collidesWith.
func TestFIFO_OverlappingEvictSetsDoNotRunInParallel(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	eff.states["x"] = process.StateReady // shared eviction target, running
	// Loading a or b both require evicting x.
	s := newFIFO(&stubPlanner{evict: map[string][]string{"a": {"x"}, "b": {"x"}}}, eff)

	s.OnRequest(req("a")) // StartSwap(a, [x])
	s.OnRequest(req("b")) // overlaps a's evict set ([x]) -> queue
	if eff.startsFor("a") != 1 {
		t.Fatalf("StartSwap(a)=%d want 1", eff.startsFor("a"))
	}
	if got := eff.startsFor("b"); got != 0 {
		t.Fatalf("b started in parallel while a evicts x: StartSwap(b)=%d want 0", got)
	}

	// a's swap completes and x is gone; b can now evict nothing and start.
	eff.states["a"] = process.StateReady
	eff.states["x"] = process.StateStopped
	s.OnSwapDone(SwapDone{ModelID: "a"})
	if got := eff.startsFor("b"); got != 1 {
		t.Fatalf("StartSwap(b)=%d want 1 after a's swap drained", got)
	}
}

// TestFIFO_QueueDrainPromotesMultiple verifies completing one swap unblocks
// every queued request that no longer collides — they all start together.
func TestFIFO_QueueDrainPromotesMultiple(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	eff.states["c"] = process.StateStopped
	// a's swap evicts both b and c; b and c evict nothing.
	s := newFIFO(&stubPlanner{evict: map[string][]string{"a": {"b", "c"}}}, eff)

	s.OnRequest(req("a")) // StartSwap(a, [b,c])
	s.OnRequest(req("b")) // collides (in a's evict set) -> queue
	s.OnRequest(req("c")) // collides -> queue
	if eff.startsFor("b") != 0 || eff.startsFor("c") != 0 {
		t.Fatalf("b/c started early")
	}

	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})

	// b and c have empty evict sets and don't evict a, so both start now.
	if eff.startsFor("b") != 1 || eff.startsFor("c") != 1 {
		t.Fatalf("StartSwap b=%d c=%d want 1 each after a done", eff.startsFor("b"), eff.startsFor("c"))
	}
	if eff.served("a") != 1 {
		t.Errorf("served(a)=%d want 1", eff.served("a"))
	}
}

// TestFIFO_QueueCollation verifies duplicate requests collapse into one swap
// per model: the second request for each model joins the active swap (at arrival
// or at drain time) rather than triggering its own swap.
func TestFIFO_QueueCollation(t *testing.T) {
	eff := newFakeEffects()
	for _, id := range []string{"a", "b", "c"} {
		eff.states[id] = process.StateStopped
	}
	// Each model evicts the other two: all swaps are mutually exclusive.
	s := newFIFO(&stubPlanner{evict: map[string][]string{
		"a": {"b", "c"},
		"b": {"a", "c"},
		"c": {"a", "b"},
	}}, eff)

	for _, id := range []string{"a", "b", "c", "a", "b", "c"} {
		s.OnRequest(req(id))
	}

	// Drain a, then its served requests, which promotes b; repeat for b -> c.
	drain := func(model string, waiters int) {
		eff.states[model] = process.StateReady
		s.OnSwapDone(SwapDone{ModelID: model})
		for i := 0; i < waiters; i++ {
			s.OnServeDone(ServeDoneEvent{ModelID: model})
		}
	}
	drain("a", 2)
	drain("b", 2)
	drain("c", 2)

	for _, id := range []string{"a", "b", "c"} {
		if got := eff.startsFor(id); got != 1 {
			t.Errorf("StartSwap(%s)=%d want 1 (collation)", id, got)
		}
		if got := eff.served(id); got != 2 {
			t.Errorf("served(%s)=%d want 2", id, got)
		}
	}
}

// TestFIFO_NoSwapWhileServing verifies a model still handling requests is not
// evicted: the evicting request waits until every in-flight request drains.
func TestFIFO_NoSwapWhileServing(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateReady
	eff.states["b"] = process.StateStopped
	s := newFIFO(&stubPlanner{evict: map[string][]string{"b": {"a"}}}, eff)

	s.OnRequest(req("a")) // fast path, inFlight[a]=1
	s.OnRequest(req("a")) // fast path, inFlight[a]=2
	s.OnRequest(req("b")) // would evict busy a -> queue
	if eff.startsFor("b") != 0 {
		t.Fatalf("b started while a serving")
	}

	s.OnServeDone(ServeDoneEvent{ModelID: "a"}) // inFlight[a]=1
	if eff.startsFor("b") != 0 {
		t.Fatalf("b started while a still serving one request")
	}

	s.OnServeDone(ServeDoneEvent{ModelID: "a"}) // inFlight[a]=0
	if eff.startsFor("b") != 1 {
		t.Fatalf("StartSwap(b)=%d want 1 after a fully drained", eff.startsFor("b"))
	}
}

// TestFIFO_GrantServeFalseDoesNotLeakInFlight verifies that when a caller has
// walked away (GrantServe returns false) the in-flight count is not bumped, so a
// later evicting request is not blocked forever.
func TestFIFO_GrantServeFalseDoesNotLeakInFlight(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	eff.serveResult["a"] = false // a's waiter is gone by grant time
	s := newFIFO(&stubPlanner{evict: map[string][]string{"b": {"a"}}}, eff)

	s.OnRequest(req("a"))
	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"}) // grant fails, inFlight[a] stays 0

	// b evicts a; since a is not in-flight, b should start immediately.
	s.OnRequest(req("b"))
	if eff.startsFor("b") != 1 {
		t.Fatalf("StartSwap(b)=%d want 1 (no leaked in-flight on a)", eff.startsFor("b"))
	}
}

// TestFIFO_OnShutdown_FailsAllWaiters verifies shutdown errors every waiter the
// scheduler holds: active-swap waiters and queued requests alike.
func TestFIFO_OnShutdown_FailsAllWaiters(t *testing.T) {
	eff := newFakeEffects()
	for _, id := range []string{"a", "b", "c"} {
		eff.states[id] = process.StateStopped
	}
	// a and b load in parallel; c collides with both and queues.
	s := newFIFO(&stubPlanner{evict: map[string][]string{"c": {"a", "b"}}}, eff)

	s.OnRequest(req("a")) // StartSwap(a)
	s.OnRequest(req("a")) // join a
	s.OnRequest(req("b")) // StartSwap(b)
	s.OnRequest(req("b")) // join b
	s.OnRequest(req("c")) // queued

	s.OnShutdown(errors.New("shutting down"))

	if got := eff.errored(""); got != 5 {
		t.Errorf("error grants=%d want 5 (2 a + 2 b + 1 c)", got)
	}
}

func TestFIFO_OnUnload_ReleasesActiveWaiters(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	s := newFIFO(&stubPlanner{}, eff)

	s.OnRequest(req("a")) // active swap a with one waiter
	s.OnRequest(req("a")) // join

	s.OnUnload([]string{"a"}, time.Second)

	if got := eff.errored("a"); got != 2 {
		t.Errorf("errored(a)=%d want 2 (active swap waiters released)", got)
	}
	if len(eff.stops) != 1 || len(eff.stops[0].ids) != 1 || eff.stops[0].ids[0] != "a" {
		t.Errorf("StopProcesses=%+v want one call stopping [a]", eff.stops)
	}
	if eff.stops[0].timeout != time.Second {
		t.Errorf("StopProcesses timeout=%v want 1s", eff.stops[0].timeout)
	}
}

func TestFIFO_OnUnload_DropsQueuedRequests(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	// b evicts a, so a request for b queues while a is loading.
	s := newFIFO(&stubPlanner{evict: map[string][]string{"b": {"a"}}}, eff)

	s.OnRequest(req("a")) // StartSwap(a)
	s.OnRequest(req("b")) // queued

	s.OnUnload([]string{"b"}, time.Second)

	if got := eff.errored("b"); got != 1 {
		t.Errorf("errored(b)=%d want 1 (queued request dropped)", got)
	}
	if got := eff.startsFor("b"); got != 0 {
		t.Errorf("StartSwap(b)=%d want 0 (b should never start)", got)
	}
	// a's swap is untouched: its waiter is neither served nor errored yet.
	if eff.served("a") != 0 || eff.errored("a") != 0 {
		t.Errorf("a swap should be untouched: served=%d errored=%d", eff.served("a"), eff.errored("a"))
	}
}

// TestFIFO_PriorityQueueOrder verifies queued requests are ordered by descending
// priority, with arrival (FIFO) order preserved among equal-priority models.
func TestFIFO_PriorityQueueOrder(t *testing.T) {
	eff := newFakeEffects()
	for _, m := range []string{"z", "A", "B", "C", "D"} {
		eff.states[m] = process.StateStopped
	}
	// z's swap evicts every other model, so any request that arrives while z is
	// loading collides with z's in-flight swap and parks in the queue.
	planner := &stubPlanner{evict: map[string][]string{"z": {"A", "B", "C", "D"}}}
	cfg := config.FifoConfig{Priority: map[string]int{"A": 10, "B": 5, "C": 5, "D": 1}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), planner, cfg, nil, eff)

	s.OnRequest(req("z")) // StartSwap(z, [A,B,C,D])

	// Arrive out of priority order; B before C exercises FIFO tie-breaking.
	for _, m := range []string{"B", "D", "C", "A"} {
		s.OnRequest(req(m))
	}

	got := make([]string, len(s.queued))
	for i, q := range s.queued {
		got[i] = q.Model
	}
	want := []string{"A", "B", "C", "D"}
	if len(got) != len(want) {
		t.Fatalf("queue=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("queue=%v want %v", got, want)
		}
	}
}

// TestFIFO_OnCancel_QueuedRequest verifies that cancelling a queued request
// prevents drainQueue from ever starting a model load for it. Without OnCancel
// the dead request would sit in the queue until a drain triggers a wasted swap.
func TestFIFO_OnCancel_QueuedRequest(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	// b evicts a, so a request for b queues while a is loading.
	s := newFIFO(&stubPlanner{evict: map[string][]string{"b": {"a"}}}, eff)

	s.OnRequest(req("a")) // StartSwap(a)

	cancelledReq := reqCh("b")
	s.OnRequest(cancelledReq) // queued (collides with a's in-flight swap)
	if len(s.queued) != 1 {
		t.Fatalf("queue len=%d want 1 before cancel", len(s.queued))
	}

	// Client disconnects.
	s.OnCancel(cancelledReq)

	if len(s.queued) != 0 {
		t.Fatalf("queue len=%d want 0 after cancel", len(s.queued))
	}

	// a's swap finishes; drainQueue runs but b is gone — no swap for b.
	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})

	if got := eff.startsFor("b"); got != 0 {
		t.Errorf("StartSwap(b)=%d want 0 (cancelled request should not trigger a load)", got)
	}
}

// TestFIFO_OnCancel_SwapWaiter verifies that cancelling a request that joined an
// in-flight swap removes it from the waiter list. When the swap completes, the
// cancelled waiter receives no grant and does not bump the in-flight count.
func TestFIFO_OnCancel_SwapWaiter(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	s := newFIFO(&stubPlanner{}, eff)

	liveReq := reqCh("a")
	cancelledReq := reqCh("a")
	s.OnRequest(liveReq)      // starts swap
	s.OnRequest(cancelledReq) // joins

	if sw := s.active["a"]; len(sw.waiters) != 2 {
		t.Fatalf("waiters=%d want 2", len(sw.waiters))
	}

	s.OnCancel(cancelledReq)

	if sw := s.active["a"]; len(sw.waiters) != 1 {
		t.Fatalf("waiters=%d want 1 after cancel", len(sw.waiters))
	}

	// Swap finishes: only the live waiter is granted.
	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})

	if got := eff.served("a"); got != 1 {
		t.Errorf("served(a)=%d want 1 (only the non-cancelled waiter)", got)
	}
}

// TestFIFO_OnCancel_NotPresent is a no-op: cancelling a request that was already
// granted (and is no longer queued or waiting) must not affect anything.
func TestFIFO_OnCancel_NotPresent(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateReady
	s := newFIFO(&stubPlanner{}, eff)

	r := reqCh("a")
	s.OnRequest(r) // fast-path served immediately

	// Cancel after grant — should be a harmless no-op.
	s.OnCancel(r)

	if got := eff.served("a"); got != 1 {
		t.Errorf("served(a)=%d want 1 (cancel of granted request is a no-op)", got)
	}
	if len(s.queued) != 0 {
		t.Errorf("queue should be empty, len=%d", len(s.queued))
	}
}

// newFIFOWithLimit builds a FIFO whose single model has the given concurrency
// limit, already in StateReady so every request exercises the fast path.
func newFIFOWithLimit(t *testing.T, model string, limit int) (*FIFO, *fakeEffects) {
	t.Helper()
	eff := newFakeEffects()
	eff.states[model] = process.StateReady
	models := map[string]config.ModelConfig{
		model: {ConcurrencyLimit: limit},
	}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{}, config.FifoConfig{}, models, eff)
	return s, eff
}

// TestFIFO_ConcurrencyLimit_RejectsOverLimit verifies that a request arriving
// while the model is at capacity gets rejected during admission, before it can
// be queued or served, and that a new request succeeds once capacity returns.
func TestFIFO_ConcurrencyLimit_RejectsOverLimit(t *testing.T) {
	s, eff := newFIFOWithLimit(t, "a", 1)

	// First request: served (inFlight 0 → 1).
	r1 := req("a")
	s.OnRequest(r1)
	assertAdmitted(t, r1)
	if got := eff.served("a"); got != 1 {
		t.Fatalf("served(a)=%d want 1", got)
	}

	// Second request while slot is occupied: rejected at admission with 429.
	r2 := req("a")
	s.OnRequest(r2)
	assertAdmission429(t, r2)
	if got := eff.errored("a"); got != 0 {
		t.Fatalf("errored(a)=%d want 0 (over-limit rejects before grant)", got)
	}

	// After the in-flight request finishes, a new request succeeds.
	s.OnServeDone(ServeDoneEvent{ModelID: "a"})
	r3 := req("a")
	s.OnRequest(r3)
	assertAdmitted(t, r3)
	if got := eff.served("a"); got != 2 {
		t.Fatalf("served(a)=%d want 2 after drain", got)
	}
}

// TestFIFO_ConcurrencyLimit_DefaultIsTen verifies that a model without an
// explicit ConcurrencyLimit gets the default cap of 10.
func TestFIFO_ConcurrencyLimit_DefaultIsTen(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateReady
	// nil models → every model gets defaultConcurrencyLimit (10).
	s := newFIFO(&stubPlanner{}, eff)

	for i := 0; i < 10; i++ {
		r := req("a")
		s.OnRequest(r)
		assertAdmitted(t, r)
	}
	if got := eff.served("a"); got != 10 {
		t.Fatalf("served(a)=%d want 10 (default limit)", got)
	}

	// 11th request is rejected.
	r := req("a")
	s.OnRequest(r)
	assertAdmission429(t, r)
	if got := eff.errored("a"); got != 0 {
		t.Fatalf("errored(a)=%d want 0 (over default limit rejects before grant)", got)
	}
}

// TestFIFO_ConcurrencyLimit_CustomLimit verifies a ConcurrencyLimit greater
// than zero overrides the default.
func TestFIFO_ConcurrencyLimit_CustomLimit(t *testing.T) {
	s, eff := newFIFOWithLimit(t, "a", 2)

	r1 := req("a")
	r2 := req("a")
	r3 := req("a")
	s.OnRequest(r1)
	s.OnRequest(r2)
	s.OnRequest(r3)
	assertAdmitted(t, r1)
	assertAdmitted(t, r2)
	assertAdmission429(t, r3)

	if got := eff.served("a"); got != 2 {
		t.Fatalf("served(a)=%d want 2 (custom limit)", got)
	}
	if got := eff.errored("a"); got != 0 {
		t.Fatalf("errored(a)=%d want 0 (over custom limit rejects before grant)", got)
	}
}

// TestFIFO_ConcurrencyLimit_SwapWaiters verifies that when more swap waiters
// exist than the concurrency limit, excess waiters are rejected during
// admission rather than after the loading stream has started.
func TestFIFO_ConcurrencyLimit_SwapWaiters(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	models := map[string]config.ModelConfig{
		"a": {ConcurrencyLimit: 2},
	}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{}, config.FifoConfig{}, models, eff)

	// Three requests arrive while model is loading: one starts swap, two join.
	r1 := req("a")
	r2 := req("a")
	r3 := req("a")
	s.OnRequest(r1)
	s.OnRequest(r2)
	s.OnRequest(r3)
	assertAdmitted(t, r1)
	assertAdmitted(t, r2)
	assertAdmission429(t, r3)

	if got := eff.startsFor("a"); got != 1 {
		t.Fatalf("StartSwap(a)=%d want 1", got)
	}
	if sw := s.active["a"]; len(sw.waiters) != 2 {
		t.Fatalf("waiters=%d want 2 (third request must not join)", len(sw.waiters))
	}

	// Swap completes: only the two admitted requests are served.
	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})

	if got := eff.served("a"); got != 2 {
		t.Fatalf("served(a)=%d want 2", got)
	}
	if got := eff.errored("a"); got != 0 {
		t.Fatalf("errored(a)=%d want 0 (excess waiter rejected at admission)", got)
	}
}

func TestFIFO_ConcurrencyLimit_QueuedWaitersReserveCapacity(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	models := map[string]config.ModelConfig{
		"a": {ConcurrencyLimit: 2},
		"b": {},
	}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{evict: map[string][]string{"a": {"b"}}}, config.FifoConfig{}, models, eff)

	bReq := req("b")
	aReq1 := req("a")
	aReq2 := req("a")
	aReq3 := req("a")

	s.OnRequest(bReq)  // StartSwap(b)
	s.OnRequest(aReq1) // queued behind b
	s.OnRequest(aReq2) // queued behind b
	s.OnRequest(aReq3) // rejected before queueing

	assertAdmitted(t, bReq)
	assertAdmitted(t, aReq1)
	assertAdmitted(t, aReq2)
	assertAdmission429(t, aReq3)

	if got := len(s.queued); got != 2 {
		t.Fatalf("queue len=%d want 2", got)
	}
	if got := eff.startsFor("a"); got != 0 {
		t.Fatalf("StartSwap(a)=%d want 0 while b is loading", got)
	}

	eff.states["b"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "b"})
	s.OnServeDone(ServeDoneEvent{ModelID: "b"})
	if got := eff.startsFor("a"); got != 1 {
		t.Fatalf("StartSwap(a)=%d want 1 after b drains", got)
	}

	eff.states["a"] = process.StateReady
	s.OnSwapDone(SwapDone{ModelID: "a"})
	if got := eff.served("a"); got != 2 {
		t.Fatalf("served(a)=%d want 2", got)
	}
}

func TestFIFO_ConcurrencyLimit_CancelledQueuedWaiterReleasesReservation(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped
	models := map[string]config.ModelConfig{
		"a": {ConcurrencyLimit: 1},
		"b": {},
	}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{evict: map[string][]string{"a": {"b"}}}, config.FifoConfig{}, models, eff)

	bReq := req("b")
	cancelledReq := reqCh("a")
	rejectedReq := req("a")
	retryReq := req("a")

	s.OnRequest(bReq)
	s.OnRequest(cancelledReq)
	s.OnRequest(rejectedReq)
	assertAdmitted(t, bReq)
	assertAdmitted(t, cancelledReq)
	assertAdmission429(t, rejectedReq)

	s.OnCancel(cancelledReq)
	s.OnRequest(retryReq)
	assertAdmitted(t, retryReq)

	if got := len(s.queued); got != 1 {
		t.Fatalf("queue len=%d want 1 after cancel and retry", got)
	}
}

// newFIFOManual builds a FIFO whose model a is manualOnly and started in the
// given state, with the planner forcing an eviction so a non-fast-path request
// would normally queue or start a swap.
func newFIFOManual(t *testing.T, state process.ProcessState) (*FIFO, *fakeEffects) {
	t.Helper()
	eff := newFakeEffects()
	eff.states["a"] = state
	eff.states["b"] = process.StateReady
	eff.manual["a"] = true
	models := map[string]config.ModelConfig{"a": {ManualOnly: true}, "b": {}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{evict: map[string][]string{"a": {"b"}}}, config.FifoConfig{}, models, eff)
	return s, eff
}

// TestFIFO_ManualOnly_NotLoadedRejectsWith503 verifies that an inference
// request for an unloaded manual-only model is rejected at admission with a
// 503, starts no swap, and leaks no concurrency reservation (a follow-up load
// probe can still start the model).
func TestFIFO_ManualOnly_NotLoadedRejectsWith503(t *testing.T) {
	s, eff := newFIFOManual(t, process.StateStopped)

	r := req("a")
	s.OnRequest(r)

	var httpErr swaputil.HTTPError
	err := admitErr(t, r)
	if !errors.As(err, &httpErr) {
		t.Fatalf("admission err=%v want HTTPError", err)
	}
	if httpErr.StatusCode() != http.StatusServiceUnavailable {
		t.Fatalf("StatusCode()=%d want 503", httpErr.StatusCode())
	}
	if got := eff.startsFor("a"); got != 0 {
		t.Errorf("StartSwap(a)=%d want 0", got)
	}
	if got := eff.served("a"); got != 0 {
		t.Errorf("served(a)=%d want 0", got)
	}

	// No reservation leak: a load probe for the same model can still start.
	probe := req("a")
	probe.LoadRequest = true
	s.OnRequest(probe)
	assertAdmitted(t, probe)
	if got := eff.startsFor("a"); got != 1 {
		t.Errorf("StartSwap(a)=%d want 1 (load probe honored)", got)
	}
}

// TestFIFO_ManualOnly_ReadyServes verifies that an inference request for a
// ready manual-only model is served normally (the operator loaded it, so it
// works).
func TestFIFO_ManualOnly_ReadyServes(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateReady
	eff.manual["a"] = true
	models := map[string]config.ModelConfig{"a": {ManualOnly: true}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{}, config.FifoConfig{}, models, eff)

	r := req("a")
	s.OnRequest(r)
	assertAdmitted(t, r)
	if got := eff.served("a"); got != 1 {
		t.Errorf("served(a)=%d want 1", got)
	}
	if got := eff.startsFor("a"); got != 0 {
		t.Errorf("StartSwap(a)=%d want 0", got)
	}
}

// TestFIFO_ManualOnly_InferenceDoesNotJoinInFlightSwap verifies that while a
// manual-only model is loading (e.g. an operator pressed the load button), a
// concurrent inference request is rejected instead of waiting on the load.
func TestFIFO_ManualOnly_InferenceDoesNotJoinInFlightSwap(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStarting
	eff.manual["a"] = true
	models := map[string]config.ModelConfig{"a": {ManualOnly: true}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{}, config.FifoConfig{}, models, eff)

	// An in-flight swap to a (as if the load button fired).
	probe := reqCh("a")
	probe.LoadRequest = true
	s.OnRequest(probe)
	assertAdmitted(t, probe)
	if got := eff.startsFor("a"); got != 1 {
		t.Fatalf("StartSwap(a)=%d want 1", got)
	}

	infer := req("a")
	s.OnRequest(infer)
	var httpErr swaputil.HTTPError
	if err := admitErr(t, infer); !errors.As(err, &httpErr) || httpErr.StatusCode() != http.StatusServiceUnavailable {
		t.Fatalf("admission err=%v want 503 HTTPError", err)
	}
	if sw, ok := s.active["a"]; !ok || len(sw.waiters) != 1 {
		t.Errorf("swap waiters=%d want 1 (inference must not join)", len(s.active["a"].waiters))
	}
	if len(s.queued) != 0 {
		t.Errorf("queue len=%d want 0", len(s.queued))
	}
}

// TestFIFO_ManualOnly_DrainQueueErrorsStaleRequests verifies that an
// inference request already in the queue (admitted before manualOnly was
// enabled by a config reload) is rejected by drainQueue instead of being
// granted or starting a load.
func TestFIFO_ManualOnly_DrainQueueErrorsStaleRequests(t *testing.T) {
	eff := newFakeEffects()
	eff.states["a"] = process.StateStopped
	eff.states["b"] = process.StateStopped // b's request below starts its swap
	// b is NOT manual at admission time; a is too. The stale request for a
	// queues because its swap would evict b, whose own swap is in flight.
	models := map[string]config.ModelConfig{"a": {ManualOnly: true}, "b": {}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{evict: map[string][]string{"a": {"b"}}}, config.FifoConfig{}, models, eff)

	// A swap to b is in flight.
	bReq := reqCh("b")
	s.OnRequest(bReq)
	assertAdmitted(t, bReq)
	if got := eff.startsFor("b"); got != 1 {
		t.Fatalf("StartSwap(b)=%d want 1", got)
	}

	// A request for a arrives while a is not manual: it queues, because its
	// swap (evicting b) collides with b's in-flight swap.
	eff.manual["a"] = false
	r := reqCh("a")
	s.OnRequest(r)
	assertAdmitted(t, r)
	if len(s.queued) != 1 {
		t.Fatalf("queue len=%d want 1", len(s.queued))
	}

	// Config reload flips a to manualOnly before the queue drains.
	eff.manual["a"] = true

	// b's swap completes: drainQueue would now start a's swap, but a is
	// manual-only and not ready, so the stale request must be rejected.
	s.OnSwapDone(SwapDone{ModelID: "b"})

	// fakeEffects.GrantError records instead of sending on the Respond
	// channel, so assert on the recorded error grant.
	if got := eff.errored("a"); got != 1 {
		t.Fatalf("errored(a)=%d want 1", got)
	}
	var httpErr swaputil.HTTPError
	found := false
	for _, g := range eff.grants {
		if g.model == "a" && g.err != nil && errors.As(g.err, &httpErr) {
			if httpErr.StatusCode() == http.StatusServiceUnavailable {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("no 503 grant recorded for a: %+v", eff.grants)
	}
	if got := eff.startsFor("a"); got != 0 {
		t.Errorf("StartSwap(a)=%d want 0", got)
	}
	if len(s.queued) != 0 {
		t.Errorf("queue len=%d want 0", len(s.queued))
	}
}

// newPoolFIFO builds a FIFO with the recent-model pool sized to n and a
// pre-seeded recency list (most recently used first).
func newPoolFIFO(n int, recent []string) *FIFO {
	s := NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{},
		config.FifoConfig{RecentPoolSize: n}, nil, newFakeEffects())
	s.recentPool = append([]string(nil), recent...)
	return s
}

// TestFIFO_PoolEvictionOnlyRemoves pins the safety property of the pool for
// every size: whatever it returns must be a subset of what the swapper asked
// for. The pool may hold an eviction back; it may never invent one.
func TestFIFO_PoolEvictionOnlyRemoves(t *testing.T) {
	asked := []string{"m1", "m2", "m3"}
	askedSet := map[string]struct{}{"m1": {}, "m2": {}, "m3": {}}

	for size := 0; size <= 5; size++ {
		s := newPoolFIFO(size, []string{"m3", "m2", "m1", "target"})
		got := s.poolEviction("target", append([]string(nil), asked...))

		for _, id := range got {
			if _, ok := askedSet[id]; !ok {
				t.Fatalf("size=%d: pool added %q to the eviction set", size, id)
			}
		}
		want := len(asked)
		if size > 1 {
			want = max(len(asked)-(size-1), 0)
		}
		if len(got) != want {
			t.Errorf("size=%d: evicts %v (%d), want %d models evicted", size, got, len(got), want)
		}
	}
}

// TestFIFO_PoolEvictionKeepsMostRecentFirst checks the order the pool spends
// its slots in: the most recently used candidates are the ones held back.
func TestFIFO_PoolEvictionKeepsMostRecentFirst(t *testing.T) {
	// Recency: m3 newest, then m2, then m1. Pool of 3 = target + 2 held back,
	// so m3 and m2 survive and m1 (least recent) is evicted.
	s := newPoolFIFO(3, []string{"m3", "m2", "m1"})
	got := s.poolEviction("target", []string{"m1", "m2", "m3"})
	if len(got) != 1 || got[0] != "m1" {
		t.Errorf("evicts %v, want [m1] (the least recently used candidate)", got)
	}
}

// TestFIFO_PoolEvictionIgnoresNonCandidates is the bug the pool's design
// avoids: a model in the recency list that the swapper did NOT nominate (a
// persistent member that served between two swaps) must not consume a pool
// slot, which would otherwise collapse the retention of the real candidates.
func TestFIFO_PoolEvictionIgnoresNonCandidates(t *testing.T) {
	// "persistent" is the most recent, but it is not an eviction candidate.
	s := newPoolFIFO(2, []string{"persistent", "m2", "m1"})
	got := s.poolEviction("target", []string{"m1", "m2"})
	// One slot for the candidates: m2 (most recent candidate) is held back.
	if len(got) != 1 || got[0] != "m1" {
		t.Errorf("evicts %v, want [m1]; a non-candidate must not consume a pool slot", got)
	}
}

// TestFIFO_MarkModelUsedMovesToFrontWithoutDuplicates checks the recency list
// bookkeeping, including that a re-served model moves rather than duplicates.
func TestFIFO_MarkModelUsedMovesToFrontWithoutDuplicates(t *testing.T) {
	s := newPoolFIFO(2, nil)
	for _, id := range []string{"a", "b", "c", "a"} {
		s.markModelUsed(id)
	}
	want := []string{"a", "c", "b"}
	if len(s.recentPool) != len(want) {
		t.Fatalf("recentPool = %v, want %v", s.recentPool, want)
	}
	for i := range want {
		if s.recentPool[i] != want[i] {
			t.Fatalf("recentPool = %v, want %v", s.recentPool, want)
		}
	}
}

// TestFIFO_ForgetModelDropsFromRecency makes sure a surgical config reload
// does not leave a stale ID holding a pool slot for a process that is gone.
func TestFIFO_ForgetModelDropsFromRecency(t *testing.T) {
	s := newPoolFIFO(2, []string{"a", "b", "c"})
	s.forgetModel("b")
	for _, id := range s.recentPool {
		if id == "b" {
			t.Fatalf("b still in recentPool %v after forgetModel", s.recentPool)
		}
	}
	if len(s.recentPool) != 2 {
		t.Errorf("recentPool = %v, want the other two entries kept", s.recentPool)
	}
}

// newVRAMFIFO builds a FIFO with the VRAM check exercised through the fake
// effects: the planner's full eviction set is `full`, and the pool is sized so
// that it holds part of it back.
func newVRAMFIFO(poolSize int, evictBeyond bool, eff *fakeEffects, full map[string][]string) *FIFO {
	cfg := config.FifoConfig{RecentPoolSize: poolSize, EvictBeyondPoolOnPressure: evictBeyond}
	return NewFIFO("test", logmon.NewWriter(io.Discard), &stubPlanner{evict: full}, cfg, nil, eff)
}

// TestFIFO_VRAMShortfallRefusesLoad is behaviour (b): with no room on the GPU
// the request is refused with a 503 rather than starting a load that would
// fail inside the runtime, and no swap is started.
func TestFIFO_VRAMShortfallRefusesLoad(t *testing.T) {
	eff := newFakeEffects()
	eff.states["m"] = process.StateStopped
	eff.vramShortfall["m"] = 4096

	s := newVRAMFIFO(0, false, eff, nil)
	r := req("m")
	s.OnRequest(r)

	if len(eff.starts) != 0 {
		t.Errorf("started %d swaps; a refused load must not start one", len(eff.starts))
	}
	// Past admit(), a refusal is delivered through GrantError, which is what
	// the baseRouter turns into the HTTP response.
	if len(eff.grants) != 1 {
		t.Fatalf("grants = %d, want 1", len(eff.grants))
	}
	var vramErr swaputil.VRAMUnavailableError
	if !errors.As(eff.grants[0].err, &vramErr) {
		t.Fatalf("err = %v, want VRAMUnavailableError", eff.grants[0].err)
	}
	if vramErr.ShortfallMB != 4096 {
		t.Errorf("ShortfallMB = %d, want 4096", vramErr.ShortfallMB)
	}
	if vramErr.StatusCode() != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", vramErr.StatusCode())
	}
	// The reservation taken by admit() must be released on the way out.
	if n := s.reserved["m"]; n != 0 {
		t.Errorf("reserved[m] = %d after a refusal, want 0", n)
	}
}

// TestFIFO_VRAMCheckSkippedWhenModelReady: a model that is already loaded
// holds its memory, so re-checking it would double-count and wrongly refuse.
func TestFIFO_VRAMCheckSkippedWhenModelReady(t *testing.T) {
	eff := newFakeEffects()
	eff.states["m"] = process.StateReady
	eff.vramShortfall["m"] = 99999

	s := newVRAMFIFO(0, false, eff, nil)
	r := req("m")
	s.OnRequest(r)

	if len(eff.grants) != 1 {
		t.Fatalf("grants = %d, want 1; a ready model must be served regardless of the check", len(eff.grants))
	}
}

// TestFIFO_EvictBeyondPoolOnPressure is the controlled fallback: when the pool
// held an eviction back and that is why the model does not fit, giving up the
// pool's retention lets the load proceed.
func TestFIFO_EvictBeyondPoolOnPressure(t *testing.T) {
	eff := newFakeEffects()
	eff.states["m"] = process.StateStopped
	eff.states["old"] = process.StateReady
	// Short with the pool-limited set (empty), fine with the full set.
	eff.vramShortfallFor = func(_ string, evict []string) int {
		if len(evict) == 0 {
			return 4096
		}
		return 0
	}

	// Pool of 2 holds "old" back; the planner's full set evicts it.
	s := newVRAMFIFO(2, true, eff, map[string][]string{"m": {"old"}})
	s.markModelUsed("old")

	r := req("m")
	s.OnRequest(r)

	if len(eff.starts) != 1 {
		t.Fatalf("starts = %d, want 1; the fallback must let the load proceed", len(eff.starts))
	}
	if got := eff.starts[0].evict; len(got) != 1 || got[0] != "old" {
		t.Errorf("evict = %v, want [old]; the fallback must give up the pool's retention", got)
	}
}

// TestFIFO_EvictBeyondPoolStillRefusesWhenHopeless: when even the swapper's
// full eviction set does not free enough, something outside llama-swap holds
// the memory and the request is refused even with the fallback on.
func TestFIFO_EvictBeyondPoolStillRefusesWhenHopeless(t *testing.T) {
	eff := newFakeEffects()
	eff.states["m"] = process.StateStopped
	eff.vramShortfallFor = func(_ string, _ []string) int { return 8192 }

	s := newVRAMFIFO(2, true, eff, map[string][]string{"m": {"old"}})
	s.markModelUsed("old")

	s.OnRequest(req("m"))

	if len(eff.starts) != 0 {
		t.Errorf("started %d swaps; a hopeless load must still be refused", len(eff.starts))
	}
	if len(eff.grants) != 1 {
		t.Fatalf("grants = %d, want 1", len(eff.grants))
	}
	var vramErr swaputil.VRAMUnavailableError
	if !errors.As(eff.grants[0].err, &vramErr) {
		t.Fatalf("err = %v, want VRAMUnavailableError", eff.grants[0].err)
	}
}

// TestFIFO_VRAMCheckOffAdmitsEverything pins the default: with no shortfall
// reported the check is invisible, which is how every existing setup behaves.
func TestFIFO_VRAMCheckOffAdmitsEverything(t *testing.T) {
	eff := newFakeEffects()
	eff.states["m"] = process.StateStopped

	s := newVRAMFIFO(0, false, eff, nil)
	s.OnRequest(req("m"))

	if len(eff.starts) != 1 {
		t.Errorf("starts = %d, want 1", len(eff.starts))
	}
}

// TestFIFO_SleepingModelWakesByEvictingItsOwnGPU reproduces the deadlock a
// rotating group hit in production, one swap after sleeping started working:
//
//	<soberano-v1.1> slept in 18.035s — GPU memory released, process kept alive
//	group: evicted [soberano-v1.1] in 18.035s, now loading qwen3.6-35b-a3b
//	...
//	group: refusing to load soberano-v1.1, short 65433 MB of GPU memory
//
// The sleeper cannot change GPU — its process is alive and bound to the card
// it started on — so the only eviction that frees room for it is the model
// that took that card over. That model is the most recently used one, which is
// exactly what the recent-model pool holds back, so the pool trimmed the
// eviction set down to a sibling on the *other* GPU. Freeing that one moves
// nothing, and the sleeper was refused for its whole size, permanently: it
// went on holding its CUDA context on the card while answering only 503.
func TestFIFO_SleepingModelWakesByEvictingItsOwnGPU(t *testing.T) {
	eff := newFakeEffects()
	eff.states["sleeper"] = process.StateSleeping // asleep on GPU 0
	eff.states["tookGpu0"] = process.StateReady   // moved onto GPU 0 when it slept
	eff.states["onGpu1"] = process.StateReady     // busy holding the other card

	// Only giving up the model on GPU 0 makes room for the sleeper; the
	// sibling on GPU 1 is irrelevant to it, exactly as the real guard reports.
	eff.vramShortfallFor = func(modelID string, evict []string) int {
		if modelID != "sleeper" {
			return 0
		}
		if containsString(evict, "tookGpu0") {
			return 0
		}
		return 65433
	}

	planner := &stubPlanner{evict: map[string][]string{
		"sleeper": {"onGpu1", "tookGpu0"},
	}}
	// PoolSize 2 is what `gpus: ["0", "1"]` sets for every member of the group.
	s := NewFIFO("test", logmon.NewWriter(io.Discard), planner,
		config.FifoConfig{RecentPoolSize: 2}, nil, eff)
	// tookGpu0 is the most recently used, so the pool protects it first.
	s.recentPool = []string{"tookGpu0", "onGpu1"}

	r := req("sleeper")
	s.OnRequest(r)

	if n := len(eff.grants); n > 0 && eff.grants[0].err != nil {
		t.Fatalf("sleeper was refused (%v); it must be allowed to reclaim its own GPU", eff.grants[0].err)
	}
	if len(eff.starts) != 1 {
		t.Fatalf("starts = %+v, want one swap for sleeper", eff.starts)
	}
	got := eff.starts[0]
	if got.model != "sleeper" {
		t.Fatalf("swapped to %q, want sleeper", got.model)
	}
	if !containsString(got.evict, "tookGpu0") {
		t.Errorf("evict = %v, want it to include tookGpu0 — the only model whose "+
			"eviction frees the GPU the sleeper is bound to", got.evict)
	}
}

// TestFIFO_EscalationGivesUpTheLeastRecentlyUsedFirst pins that escalating past
// the pool is not a licence to clear it: the set grows one model at a time,
// oldest first, and stops as soon as there is room.
func TestFIFO_EscalationGivesUpTheLeastRecentlyUsedFirst(t *testing.T) {
	eff := newFakeEffects()
	eff.states["sleeper"] = process.StateSleeping
	eff.states["hot"] = process.StateReady
	eff.states["warm"] = process.StateReady
	eff.states["cold"] = process.StateReady

	// Any one eviction is enough here, so the choice is purely about recency.
	eff.vramShortfallFor = func(modelID string, evict []string) int {
		if modelID != "sleeper" || len(evict) > 0 {
			return 0
		}
		return 40000
	}

	planner := &stubPlanner{evict: map[string][]string{
		"sleeper": {"hot", "warm", "cold"},
	}}
	// A pool of 4 holds all three back, so escalation picks every one of them.
	s := NewFIFO("test", logmon.NewWriter(io.Discard), planner,
		config.FifoConfig{RecentPoolSize: 4}, nil, eff)
	s.recentPool = []string{"hot", "warm", "cold"}

	s.OnRequest(req("sleeper"))

	if len(eff.starts) != 1 {
		t.Fatalf("starts = %+v, want one swap", eff.starts)
	}
	if got := eff.starts[0].evict; len(got) != 1 || got[0] != "cold" {
		t.Errorf("evict = %v, want [cold] — the least recently used, and no more", got)
	}
}

// TestFIFO_EscalationSparesAModelThatIsServing pins that escalation never
// nominates a busy process. The in-flight check upstream ran against the
// smaller eviction set, so this is the one path that could newly reach one.
func TestFIFO_EscalationSparesAModelThatIsServing(t *testing.T) {
	eff := newFakeEffects()
	eff.states["sleeper"] = process.StateSleeping
	eff.states["busy"] = process.StateReady

	eff.vramShortfallFor = func(modelID string, evict []string) int {
		if modelID != "sleeper" || containsString(evict, "busy") {
			return 0
		}
		return 65433
	}

	planner := &stubPlanner{evict: map[string][]string{"sleeper": {"busy"}}}
	s := NewFIFO("test", logmon.NewWriter(io.Discard), planner,
		config.FifoConfig{RecentPoolSize: 2}, nil, eff)
	s.recentPool = []string{"busy"}
	s.inFlight = map[string]int{"busy": 1}

	s.OnRequest(req("sleeper"))

	for _, st := range eff.starts {
		if containsString(st.evict, "busy") {
			t.Fatalf("evicted a model that is serving: %+v", st)
		}
	}
}
