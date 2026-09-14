package scheduler

import (
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// defaultConcurrencyLimit caps simultaneous in-flight requests per model when
// the model config leaves concurrencyLimit unset.
const defaultConcurrencyLimit = 10

// activeSwap tracks one in-flight swap and the callers waiting on it.
type activeSwap struct {
	modelID string
	evict   []string
	waiters []HandlerReq
}

// FIFO is the default scheduler. Requests are handled in a first-in, first-out order.
// To reduce swapping requests for a model that is already running will be handled
// immediately by the running process.
//
// Requests into this schedule are handled like this:
//
// A B C A B C --> A A B B C C
//
// The strategy is simple and reduces the number of swaps required.
type FIFO struct {
	name    string
	logger  *logmon.Monitor
	planner Swapper
	cfg     config.FifoConfig
	effects Effects

	// recentPool is every model that has served, most recently used first. It
	// is deliberately NOT truncated to the pool size: a model that is never an
	// eviction candidate (a persistent group member answering a request
	// between two swaps) would otherwise push the pool's own models out of the
	// recency order and collapse their standing. poolEviction picks its keep
	// set out of the eviction candidates instead, so extra entries are
	// harmless.
	recentPool []string

	limits   map[string]int
	active   map[string]*activeSwap
	reserved map[string]int
	inFlight map[string]int
	queued   []HandlerReq
}

// NewFIFO builds a FIFO scheduler. Per-model concurrency limits are derived
// from models: each model's ConcurrencyLimit overrides defaultConcurrencyLimit
// when set to a value greater than zero.
func NewFIFO(name string, logger *logmon.Monitor, planner Swapper, cfg config.FifoConfig, models map[string]config.ModelConfig, eff Effects) *FIFO {
	limits := make(map[string]int, len(models))
	for id, mc := range models {
		limit := defaultConcurrencyLimit
		if mc.ConcurrencyLimit > 0 {
			limit = mc.ConcurrencyLimit
		}
		limits[id] = limit
	}

	return &FIFO{
		name:     name,
		logger:   logger,
		planner:  planner,
		cfg:      cfg,
		effects:  eff,
		limits:   limits,
		active:   make(map[string]*activeSwap),
		reserved: make(map[string]int),
		inFlight: make(map[string]int),
	}
}

// OnRequest decides what to do with one incoming ServeHTTP request. It never
// blocks indefinitely: any work that has to wait (starting a process, stopping
// siblings, waiting for ready) is deferred to a swap goroutine and reported back
// via OnSwapDone.
//
// The decision tree, in order:
//
//  1. Unknown model — respond with ErrModelNotFound and move on.
//  2. A swap to the same model is already in flight — attach this waiter so
//     one swap serves all callers that asked for the same model.
//  3. Fast path — the target process is already ready, the planner sees
//     nothing to evict, and no in-flight swap is evicting it. Hand back its
//     ServeHTTP immediately.
//  4. Would collide with an in-flight swap (we'd stop their target, or they're
//     stopping us) — park in the queue for OnSwapDone to drain.
//  5. Would evict a process that is still handling requests — park in the
//     queue. OnServeDone will retry when the busy process drains.
//  6. Otherwise — start a new swap. This may run in parallel with other active
//     swaps when their evict sets don't intersect.
func (s *FIFO) OnRequest(req HandlerReq) {
	// (1) Unknown model.
	state, ok := s.effects.ModelState(req.Model)
	if !ok {
		s.logger.Debugf("%s: model %s not handled by this router", s.name, req.Model)
		s.rejectAdmission(req, ErrModelNotFound)
		return
	}

	// Manual-only models: an inference request for a model that is not
	// ready is rejected immediately with a 503 so upstream clients (e.g.
	// LiteLLM) fail over to their fallback instead of waiting on a load.
	// Placed before admit() on purpose: a rejected request must not hold
	// a concurrency reservation. Load probes (GET — the dashboard's load
	// buttons use exactly this shape) fall through so the operator can
	// still start the model; a ready manual model also falls through and
	// gets the fast path below.
	if manual, _ := s.effects.ModelManual(req.Model); manual && !req.LoadRequest && state != process.StateReady {
		s.logger.Debugf("%s: rejecting request for manual-only model %s (not loaded)", s.name, req.Model)
		s.rejectAdmission(req, swaputil.ManualLoadError{ModelID: req.Model})
		return
	}

	if !s.admit(req) {
		return
	}

	// (2) Join an in-flight swap for the same model.
	if sw, ok := s.active[req.Model]; ok {
		s.logger.Debugf("%s: joining in-flight swap for model %s (%d waiters)", s.name, req.Model, len(sw.waiters)+1)
		sw.waiters = append(sw.waiters, req)
		return
	}

	running, evict := s.evictionFor(req.Model)

	// (3) Fast path: ready, nothing to evict, and nobody is evicting us.
	if state == process.StateReady && len(evict) == 0 && !collidesWith(req.Model, evict, s.active) {
		s.logger.Debugf("%s: fast-path serving model %s (already ready)", s.name, req.Model)
		s.grantHandler(req, req.Model)
		return
	}

	// (4) Collision with an in-flight swap — queue.
	if collidesWith(req.Model, evict, s.active) {
		s.logger.Debugf("%s: queuing request for model %s (collides with in-flight swap)", s.name, req.Model)
		s.enqueue(req)
		return
	}

	// (5) Would evict a busy process — queue until it drains.
	if conflictsWithInFlight(evict, s.inFlight) {
		s.logger.Debugf("%s: queuing request for model %s (would evict in-flight process)", s.name, req.Model)
		s.enqueue(req)
		return
	}

	// (6) Refuse a load the GPU has no room for, rather than letting it fail
	// deep inside the runtime minutes later.
	evict, short, ok := s.admitVRAM(req.Model, req.GpuOverride, evict, running, state)
	if !ok {
		s.logger.Warnf("%s: refusing to load %s, short %d MB of GPU memory", s.name, req.Model, short)
		// admit() already reserved a slot for this request; grantError
		// releases it on the way out.
		s.grantError(req, swaputil.VRAMUnavailableError{ModelID: req.Model, ShortfallMB: short})
		return
	}

	// (7) Start a new (possibly parallel) swap.
	s.logger.Debugf("%s: starting swap for model %s, evicting %v", s.name, req.Model, evict)
	s.startSwap(req, evict, running)
}

// OnCancel removes a request whose client has disconnected from the queue and
// from every in-flight swap's waiters. If the request was the sole waiter of an
// active swap, the swap goroutine is left to complete on its own — OnSwapDone
// will find no waiters and simply clean up. This prevents drainQueue from ever
// starting a model load for a caller that is no longer there.
func (s *FIFO) OnCancel(req HandlerReq) {
	removed := false

	// Prune from the queue.
	if len(s.queued) > 0 {
		kept := s.queued[:0]
		for _, q := range s.queued {
			if q.Respond == req.Respond {
				removed = true
				s.release(q.Model)
				continue
			}
			kept = append(kept, q)
		}
		s.queued = kept
	}

	// Prune from any active swap's waiters.
	for _, sw := range s.active {
		filtered := sw.waiters[:0]
		for _, w := range sw.waiters {
			if w.Respond == req.Respond {
				removed = true
				s.release(w.Model)
				continue
			}
			filtered = append(filtered, w)
		}
		sw.waiters = filtered
	}

	if removed {
		s.logger.Debugf("%s: cancelled request for model %s pruned from scheduler", s.name, req.Model)
		broadcastQueuePositions(s.queued)
	}
}

// OnSwapDone fans the result out to every waiter that joined this swap, removes
// the swap from the active map, then walks the queue once, promoting any items
// that no longer collide with the remaining active set. FIFO order is preserved:
// items still blocked stay in place.
func (s *FIFO) OnSwapDone(ev SwapDone) {
	sw, ok := s.active[ev.ModelID]
	if !ok {
		return
	}
	delete(s.active, ev.ModelID)

	for _, w := range sw.waiters {
		if ev.Err != nil {
			s.grantError(w, ev.Err)
		} else {
			s.grantHandler(w, ev.ModelID)
		}
	}

	s.drainQueue()
}

// OnServeDone decrements the per-model in-flight count and, when that drops to
// zero, retries the queue: requests whose swap was deferred because they would
// have evicted this (now-idle) process can now proceed.
func (s *FIFO) OnServeDone(ev ServeDoneEvent) {
	s.inFlight[ev.ModelID]--
	s.release(ev.ModelID)
	if s.inFlight[ev.ModelID] <= 0 {
		delete(s.inFlight, ev.ModelID)
		s.drainQueue()
	}
}

// OnUnload reconciles router-owned state with the impending Stop, performs the
// Stop (synchronously, via Effects) so callers of Unload remain blocked until
// each targeted process has exited, then drains the queue.
func (s *FIFO) OnUnload(targets []string, timeout time.Duration) {
	unloadErr := fmt.Errorf("%s: model unloaded", s.name)

	targetSet := make(map[string]bool, len(targets))
	for _, id := range targets {
		targetSet[id] = true
	}

	// Release waiters of any in-flight swap whose target is being unloaded.
	// The swap goroutine itself is left to finish on its own; when its
	// SwapDone arrives, OnSwapDone will find no entry in active and drop it.
	for id := range targetSet {
		sw, ok := s.active[id]
		if !ok {
			continue
		}
		for _, w := range sw.waiters {
			s.grantError(w, unloadErr)
		}
		delete(s.active, id)
	}

	// Drop queued requests addressed to unloaded models. Requests for other
	// models stay queued and may benefit from drainQueue at the end.
	if len(s.queued) > 0 {
		kept := s.queued[:0]
		for _, w := range s.queued {
			if targetSet[w.Model] {
				s.grantError(w, unloadErr)
				continue
			}
			kept = append(kept, w)
		}
		s.queued = kept
	}

	// Stop the targeted processes. Done synchronously so Unload's caller can
	// rely on "after Unload returns, the process is stopped". inFlight is
	// intentionally NOT cleared here: each dying handler will fire its tracked
	// serve and reach OnServeDone in the normal way.
	s.effects.StopProcesses(timeout, targets)

	// Removing entries from active above may have unblocked queued requests
	// that previously collided with the now-cancelled swaps.
	s.drainQueue()
}

// OnModelReload detaches model from the scheduler's state during a surgical
// config refresh: any in-flight swap is cancelled (its waiters are released
// with an error so their callers get a response) and queued requests for it
// are dropped the same way. Nothing may keep holding the about-to-be-stopped
// process afterwards. Reservations are released per item so the panics in
// release() stay reachable only for real accounting bugs.
func (s *FIFO) OnModelReload(model string) {
	refreshErr := fmt.Errorf("%s: model reloaded", s.name)
	s.forgetModel(model)
	if sw, ok := s.active[model]; ok {
		for _, w := range sw.waiters {
			s.release(w.Model)
			s.effects.GrantError(w, refreshErr)
		}
		delete(s.active, model)
	}
	if len(s.queued) > 0 {
		kept := s.queued[:0]
		for _, q := range s.queued {
			if q.Model == model {
				s.release(q.Model)
				s.effects.GrantError(q, refreshErr)
				continue
			}
			kept = append(kept, q)
		}
		s.queued = kept
	}
	broadcastQueuePositions(s.queued)
	s.drainQueue()
}

// UpdateModel resyncs the scheduler's per-model concurrency limits with the
// (possibly new) model config. Called on the run loop.
func (s *FIFO) UpdateModel(model string, mc config.ModelConfig, remove bool) {
	if remove {
		delete(s.limits, model)
		return
	}
	limit := defaultConcurrencyLimit
	if mc.ConcurrencyLimit > 0 {
		limit = mc.ConcurrencyLimit
	}
	s.limits[model] = limit
}

// admitVRAM decides whether a swap that would load target may start, given the
// eviction set the pool settled on. It returns the eviction set to use and
// ok=false when the load must be refused.
//
// When the GPU is short, evictBeyondPoolOnPressure decides what happens: with
// it off (the default) the request is refused, which keeps the pool's
// retention and reports capacity honestly; with it on, the pool's held-back
// evictions are given up so the load can proceed, and the request is only
// refused when even the swapper's full eviction set does not free enough.
//
// state is the target's current process state: a model that is already ready
// has its memory, and re-checking it would double-count what it already holds.
func (s *FIFO) admitVRAM(target, device string, evict, running []string, state process.ProcessState) ([]string, int, bool) {
	if state == process.StateReady {
		return evict, 0, true
	}
	short := s.effects.VRAMShortfall(target, device, evict)
	if short == 0 {
		return evict, 0, true
	}
	if !s.cfg.EvictBeyondPoolOnPressure {
		return evict, short, false
	}

	full := s.planner.EvictionFor(target, running)
	if s.effects.VRAMShortfall(target, device, full) > 0 {
		// Even giving up the whole pool does not free enough: something
		// outside llama-swap is holding the memory.
		return evict, short, false
	}
	s.logger.Infof("%s: %s is short %d MB; evicting beyond the recent pool to make room (evictBeyondPoolOnPressure)",
		s.name, target, short)
	return full, 0, true
}

// markModelUsed moves modelID to the front of the recency list. See the
// recentPool field for why the list is never truncated here.
func (s *FIFO) markModelUsed(modelID string) {
	for i, id := range s.recentPool {
		if id == modelID {
			s.recentPool = append(s.recentPool[:i], s.recentPool[i+1:]...)
			break
		}
	}
	s.recentPool = append([]string{modelID}, s.recentPool...)
}

// forgetModel drops a model from the recency list. Called when a surgical
// config reload replaces or removes the model, so a stale ID cannot keep
// holding a pool slot for a process that no longer exists.
func (s *FIFO) forgetModel(modelID string) {
	for i, id := range s.recentPool {
		if id == modelID {
			s.recentPool = append(s.recentPool[:i], s.recentPool[i+1:]...)
			return
		}
	}
}

// evictionFor is the single eviction decision for target: the swapper's own
// eviction set, minus whatever the recent-model pool holds back. Both
// OnRequest and drainQueue go through here, so a request that waited in the
// queue is decided exactly like one that never had to.
func (s *FIFO) evictionFor(target string) (running, evict []string) {
	running = s.runningSet(target)
	evict = s.planner.EvictionFor(target, running)
	return running, s.poolEviction(target, evict)
}

// poolEviction holds the swapper back: the RecentPoolSize most recently used
// models stay loaded even when the swapper asked for them to be unloaded, so
// the pool behaves as an LRU working set — loading a new model pushes out the
// least recently used one instead of every other model.
//
// It only ever removes IDs from evict, never adds any. Models the swapper
// deliberately keeps resident — a persistent group, a model the planner never
// nominated — were never in evict to begin with and stay untouched.
//
// Sizing the pool is the operator's call: what it keeps loaded is exactly what
// the swapper wanted unloaded to free capacity, so the hardware has to have
// room for RecentPoolSize models at once. 0 or 1 disables the pool and leaves
// every eviction decision to the swapper.
func (s *FIFO) poolEviction(target string, evict []string) []string {
	size := s.cfg.PoolSizeFor(target)
	if size <= 1 || len(evict) == 0 {
		return evict
	}

	candidates := make(map[string]struct{}, len(evict))
	for _, id := range evict {
		candidates[id] = struct{}{}
	}

	// target occupies one slot — it is about to serve, and becomes the most
	// recently used model once granted — leaving size-1 slots for the
	// candidates, handed out in recency order.
	keep := make(map[string]struct{}, size-1)
	for _, id := range s.recentPool {
		if len(keep) >= size-1 {
			break
		}
		if id == target {
			continue
		}
		if _, isCandidate := candidates[id]; isCandidate {
			keep[id] = struct{}{}
		}
	}

	kept := make([]string, 0, len(evict))
	for _, id := range evict {
		if _, held := keep[id]; held {
			continue
		}
		kept = append(kept, id)
	}
	return kept
}

// OnShutdown grants err to every waiter still held by the scheduler.
func (s *FIFO) OnShutdown(err error) {
	for _, sw := range s.active {
		for _, w := range sw.waiters {
			s.grantError(w, err)
		}
	}
	for _, w := range s.queued {
		s.grantError(w, err)
	}
}

// grantHandler hands the caller a tracked handler for modelID and, only if the
// caller was still there to receive it, bumps the in-flight count. Incrementing
// when the grant failed would strand the counter and block future evictions.
// Concurrency-limit rejection happens earlier in admit, before a request can
// start the loading stream.
func (s *FIFO) grantHandler(req HandlerReq, modelID string) {
	if err := swaputil.SetReqData(req.Ctx, "fifo_priority", strconv.Itoa(s.cfg.Priority[req.Model])); err != nil {
		s.logger.Debugf("failed to set fifo_priority metadata: %v", err)
	}

	s.markModelUsed(modelID)

	if s.effects.GrantServe(req, modelID) {
		s.inFlight[modelID]++
	} else {
		s.release(modelID)
	}
}

// grantError reports a post-admission error to the caller and releases the
// request's reserved concurrency slot.
func (s *FIFO) grantError(req HandlerReq, err error) {
	s.release(req.Model)
	s.effects.GrantError(req, err)
}

// admit performs the pre-stream admission handshake. Accepted requests reserve
// one future serving slot until they serve, cancel while waiting, or receive a
// post-admission error.
func (s *FIFO) admit(req HandlerReq) bool {
	if s.reserved[req.Model] >= s.limit(req.Model) {
		s.rejectAdmission(req, swaputil.ConcurrencyLimitError{})
		return false
	}
	if !sendAdmission(req, nil) {
		return false
	}
	s.reserved[req.Model]++
	return true
}

func (s *FIFO) rejectAdmission(req HandlerReq, err error) {
	sendAdmission(req, err)
}

func sendAdmission(req HandlerReq, err error) bool {
	if req.Admit == nil {
		return true
	}
	done := reqDone(req)
	select {
	case <-done:
		return false
	default:
	}
	select {
	case req.Admit <- err:
		return true
	case <-done:
		return false
	}
}

func reqDone(req HandlerReq) <-chan struct{} {
	if req.Ctx == nil {
		return nil
	}
	return req.Ctx.Done()
}

func (s *FIFO) release(modelID string) {
	if s.reserved[modelID] <= 0 {
		panic(fmt.Sprintf("%s: release without reservation for model %s", s.name, modelID))
	}
	s.reserved[modelID]--
	if s.reserved[modelID] == 0 {
		delete(s.reserved, modelID)
	}
}

// limit returns the per-model concurrency cap, defaulting to
// defaultConcurrencyLimit when the model has no explicit entry.
func (s *FIFO) limit(modelID string) int {
	if l, ok := s.limits[modelID]; ok {
		return l
	}
	return defaultConcurrencyLimit
}

// startSwap records the swap as active and launches it via Effects. running is
// the set EvictionFor saw, forwarded to OnSwapStart so the planner logs against
// the same picture it decided on.
func (s *FIFO) startSwap(initial HandlerReq, evict, running []string) {
	s.active[initial.Model] = &activeSwap{
		modelID: initial.Model,
		evict:   evict,
		waiters: []HandlerReq{initial},
	}
	s.planner.OnSwapStart(initial.Model, running)
	s.effects.StartSwap(initial.Model, evict, process.Options{GpuOverride: initial.GpuOverride})
}

// enqueue inserts req into the queue in priority order: it goes just before the
// first queued item whose priority is strictly lower, so higher-priority models
// are serviced first while equal-priority requests keep their arrival (FIFO)
// order. Priorities come from the FifoConfig; unlisted models default to 0.
func (s *FIFO) enqueue(req HandlerReq) {
	p := s.cfg.Priority[req.Model]
	i := len(s.queued)
	for j, q := range s.queued {
		if s.cfg.Priority[q.Model] < p {
			i = j
			break
		}
	}
	s.queued = append(s.queued, HandlerReq{})
	copy(s.queued[i+1:], s.queued[i:])
	s.queued[i] = req
	broadcastQueuePositions(s.queued)
}

// drainQueue walks the queued requests in order, re-running the OnRequest
// decision tree against the (now smaller) active set. Items that can now start
// or join become satisfied; items still blocked remain queued in original order
// so they get another chance on the next swap completion.
func (s *FIFO) drainQueue() {
	if len(s.queued) == 0 {
		return
	}
	pending := s.queued
	var remaining []HandlerReq
	for _, req := range pending {
		state, ok := s.effects.ModelState(req.Model)
		if !ok {
			s.grantError(req, ErrModelNotFound)
			continue
		}
		// Same rule as OnRequest (2a): a manual-only model that is not
		// ready does not serve inference requests. Keeps a config reload
		// that enables manualOnly from letting queued requests trigger
		// (or join) a load.
		if manual, _ := s.effects.ModelManual(req.Model); manual && !req.LoadRequest && state != process.StateReady {
			s.grantError(req, swaputil.ManualLoadError{ModelID: req.Model})
			continue
		}
		if sw, ok := s.active[req.Model]; ok {
			s.logger.Debugf("%s: queued request for model %s now joining in-flight swap", s.name, req.Model)
			sw.waiters = append(sw.waiters, req)
			continue
		}
		running, evict := s.evictionFor(req.Model)
		if state == process.StateReady && len(evict) == 0 && !collidesWith(req.Model, evict, s.active) {
			s.logger.Debugf("%s: queued request for model %s now served fast-path", s.name, req.Model)
			s.grantHandler(req, req.Model)
			continue
		}
		if collidesWith(req.Model, evict, s.active) {
			remaining = append(remaining, req)
			continue
		}
		evict, short, admitted := s.admitVRAM(req.Model, req.GpuOverride, evict, running, state)
		if !admitted {
			s.logger.Warnf("%s: refusing queued load of %s, short %d MB of GPU memory", s.name, req.Model, short)
			s.grantError(req, swaputil.VRAMUnavailableError{ModelID: req.Model, ShortfallMB: short})
			continue
		}
		if conflictsWithInFlight(evict, s.inFlight) {
			remaining = append(remaining, req)
			continue
		}
		s.logger.Debugf("%s: queued request for model %s now starting swap, evicting %v", s.name, req.Model, evict)
		s.startSwap(req, evict, running)
	}
	s.queued = remaining
	broadcastQueuePositions(s.queued)
}

// runningSet is the live model set handed to the Swapper: every process the
// baseRouter reports as running, unioned with the targets of in-flight swaps
// (excluding excludeActive, the model whose own swap is being decided — its
// in-flight entry must not count as "already running"). The result is sorted so
// eviction decisions derived from it are deterministic.
func (s *FIFO) runningSet(excludeActive string) []string {
	seen := make(map[string]struct{})
	var out []string
	add := func(id string) {
		if _, dup := seen[id]; dup {
			return
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	for id, state := range s.effects.RunningModels() {
		// A sleeping model has already released its GPU memory; it is alive
		// only to make its next load fast. Handing it to the planner would
		// have it nominated for eviction to free capacity it is not using —
		// and the eviction of an already-sleeping model can only end in it
		// being stopped, which is exactly the cold start sleeping exists to
		// avoid. Capacity-wise it is not running, so it is not listed here.
		if state == process.StateSleeping {
			continue
		}
		add(id)
	}
	for _, id := range activeTargets(s.active, excludeActive) {
		add(id)
	}
	sort.Strings(out)
	return out
}

// activeTargets returns the IDs of every in-flight swap target except exclude.
// The planner uses this to account for models committed to but not yet reflected
// in process state.
func activeTargets(active map[string]*activeSwap, exclude string) []string {
	if len(active) == 0 {
		return nil
	}
	out := make([]string, 0, len(active))
	for id := range active {
		if id == exclude {
			continue
		}
		out = append(out, id)
	}
	return out
}

// collidesWith reports whether a new swap with this target and evict set can
// safely run alongside the currently active swaps. Same-target callers should
// JOIN (handled before this) — they do not collide with themselves.
func collidesWith(target string, evict []string, active map[string]*activeSwap) bool {
	for id, sw := range active {
		if id == target {
			continue
		}
		if containsString(evict, id) {
			return true
		}
		if containsString(sw.evict, target) {
			return true
		}
		if slicesOverlap(evict, sw.evict) {
			return true
		}
	}
	return false
}

// slicesOverlap reports whether xs and ys share any common element.
func slicesOverlap(xs, ys []string) bool {
	for _, x := range xs {
		if containsString(ys, x) {
			return true
		}
	}
	return false
}

// conflictsWithInFlight reports whether any model in evict is still handling
// requests. Stopping a busy process would cancel its callers' connections, so
// the scheduler defers the swap until those callers finish.
func conflictsWithInFlight(evict []string, inFlight map[string]int) bool {
	for _, m := range evict {
		if inFlight[m] > 0 {
			return true
		}
	}
	return false
}

func containsString(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// broadcastQueuePositions sends each queued request its current 1-indexed
// position. Sends are non-blocking: if the channel is full, the old value is
// drained first so the consumer always sees the latest position.
func broadcastQueuePositions(queued []HandlerReq) {
	for i, req := range queued {
		pos := i + 1
		select {
		case req.PositionCh <- pos:
		default:
			select {
			case <-req.PositionCh:
			default:
			}
			select {
			case req.PositionCh <- pos:
			default:
			}
		}
	}
}
