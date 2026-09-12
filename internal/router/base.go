package router

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

type shutdownReq struct {
	timeout time.Duration
	respond chan error
}

type unloadReq struct {
	targets []string
	timeout time.Duration
	respond chan struct{}
}

// refreshModelReq asks the run loop to surgically replace one model's
// process (and the router's view of its config) after a config edit,
// without touching any other model.
type refreshModelReq struct {
	model   string
	newCfg  config.Config
	respond chan error
}

// baseRouter owns the channels, run-loop, and process machinery shared by every
// concrete router. Concrete routers embed *baseRouter and supply a
// scheduler.Swapper describing how eviction sets are decided. baseRouter
// implements scheduler.Effects so the scheduler can call back for side-effects.
type baseRouter struct {
	name string
	// config is the live config, read via configAt(). A surgical model
	// refresh (RefreshModel) swaps it atomically on the run loop, so request
	// paths — model lookup, websocket filtering, timeouts, manual-only —
	// observe the new values without rebuilding the router.
	config atomic.Pointer[config.Config]
	// upstreamlog captures child process output for rebuilt processes during
	// a surgical refresh.
	upstreamlog *logmon.Monitor
	// processes is the live model->process map, read via processesAt(). Like
	// config it is swapped atomically (copy-on-write) by a refresh, so the
	// map a running request captured stays consistent for its whole life.
	processes atomic.Pointer[map[string]process.Process]
	logger    *logmon.Monitor
	schedule  scheduler.Scheduler

	// shutdownCtx governs the request machinery: cancelling it tells grant()
	// and ServeHTTP to stop granting and reject callers. It is deliberately
	// separate from procCtx — see procCtx below.
	shutdownCtx  context.Context
	shutdownFn   context.CancelFunc
	shuttingDown atomic.Bool

	// procCtx is the parent context for every managed process and governs
	// process lifetime only. handleShutdown stops processes gracefully via
	// Stop() and cancels procCtx afterwards, so teardown is never a context
	// cancel racing the graceful path (which collapsed the grace to 100ms and
	// let the caller return before children were reaped — see process run loop).
	procCtx    context.Context
	procCancel context.CancelFunc

	handlerCh   chan scheduler.HandlerReq
	cancelCh    chan scheduler.HandlerReq
	shutdownCh  chan shutdownReq
	unloadCh    chan unloadReq
	refreshCh   chan refreshModelReq
	swapDoneCh  chan scheduler.SwapDone
	serveDoneCh chan scheduler.ServeDoneEvent

	runDone chan struct{}

	// testProcessed, when non-nil, receives one event after each handlerReq
	// or swapDone has been fully processed by run(). Tests use it to wait
	// for run() to reach a deterministic state without sleeping. serveDone
	// events are intentionally NOT signalled here so test event counts
	// remain stable.
	testProcessed chan struct{}
}

func newBaseRouter(
	name string,
	conf config.Config,
	processes map[string]process.Process,
	logger *logmon.Monitor,
	planner scheduler.Swapper,
) (*baseRouter, error) {
	shutdownCtx, shutdownFn := context.WithCancel(context.Background())
	procCtx, procCancel := context.WithCancel(context.Background())
	confPtr := &conf
	b := &baseRouter{
		name:        name,
		config:      atomic.Pointer[config.Config]{},
		processes:   atomic.Pointer[map[string]process.Process]{},
		logger:      logger,
		shutdownCtx: shutdownCtx,
		shutdownFn:  shutdownFn,
		procCtx:     procCtx,
		procCancel:  procCancel,
		handlerCh:   make(chan scheduler.HandlerReq),
		cancelCh:    make(chan scheduler.HandlerReq),
		shutdownCh:  make(chan shutdownReq),
		unloadCh:    make(chan unloadReq),
		refreshCh:   make(chan refreshModelReq),
		swapDoneCh:  make(chan scheduler.SwapDone),
		serveDoneCh: make(chan scheduler.ServeDoneEvent),
		runDone:     make(chan struct{}),
	}
	b.config.Store(confPtr)
	b.processes.Store(&processes)
	sched, err := scheduler.New(conf, name, logger, planner, b)
	if err != nil {
		return nil, err
	}
	b.schedule = sched
	return b, nil
}

// configAt returns the current live config. Readers hold a snapshot for the
// duration of one operation; RefreshModel swaps the pointer atomically on the
// run loop, so a concurrent refresh simply takes effect from the next read.
func (b *baseRouter) configAt() config.Config {
	return *b.config.Load()
}

// processesAt returns a snapshot of the model->process map. The map itself is
// never mutated after construction: a refresh builds a new map and swaps the
// pointer, so a request that captured a map mid-flight keeps a consistent
// view for its whole life.
func (b *baseRouter) processesAt() map[string]process.Process {
	return *b.processes.Load()
}

// swapProcess atomically publishes a new model->process map. Called on the
// run loop only; callers build the map from the current snapshot.
func (b *baseRouter) swapProcess(next map[string]process.Process) {
	b.processes.Store(&next)
}

// RefreshModel surgically replaces one model's process (and the router's view
// of its config) after a single-model config edit, without touching any other
// model's running process. It is funneled through the run loop so the swap is
// serialized with swaps/unloads. newCfg is the freshly loaded full config; the
// caller is responsible for ensuring only this model's block changed (the
// eviction planner holds the old config, which is still valid when the group/
// matrix structure is unchanged).
//
// If the model was serving (StateReady) before the refresh, the rebuilt
// process is restarted in the background.
func (b *baseRouter) RefreshModel(model string, newCfg config.Config) error {
	req := refreshModelReq{model: model, newCfg: newCfg, respond: make(chan error, 1)}
	select {
	case b.refreshCh <- req:
	case <-b.runDone:
		return fmt.Errorf("%s: router is shutting down", b.name)
	}
	select {
	case err := <-req.respond:
		return err
	case <-b.runDone:
		return fmt.Errorf("%s: router is shutting down", b.name)
	}
}

// handleRefreshModel is the run-loop body of RefreshModel.
func (b *baseRouter) handleRefreshModel(req refreshModelReq) error {
	model := req.model
	newCfg := req.newCfg

	// 1) Build the replacement process FIRST, so a build failure leaves the
	//    router completely untouched (old config, old process, still serving).
	procs := b.processesAt()
	old, hadOld := procs[model]
	wasReady := hadOld && old.State() == process.StateReady
	mc, exists := newCfg.Models[model]

	next := make(map[string]process.Process, len(procs))
	for id, p := range procs {
		next[id] = p
	}
	var newProc process.Process
	if exists {
		procLog := logmon.NewWriter(b.upstreamlog)
		np, err := process.New(b.procCtx, model, mc, procLog, b.logger)
		if err != nil {
			return fmt.Errorf("creating process for %q: %w", model, err)
		}
		newProc = np
		next[model] = np
	} else {
		// Model removed from the config: drop it from the map.
		delete(next, model)
	}

	// 2) Commit: publish the new config atomically (model lookup,
	//    manual-only, websocket filtering, and timeouts now observe the new
	//    values), drop this model's waiters/queued requests from the scheduler
	//    (nothing may keep the about-to-be-stopped process) and resync its
	//    per-model concurrency limit.
	b.config.Store(&newCfg)
	b.schedule.OnModelReload(model)
	b.schedule.UpdateModel(model, mc, !exists)

	// 3) Stop the old process, then publish the new map. In-flight requests
	//    being served by the old process are killed the same way an Unload
	//    kills them (their callers see an error and may retry).
	if hadOld {
		if stopErr := old.Stop(b.unloadTimeout(model)); stopErr != nil {
			b.logger.Warnf("%s: stopping %s during refresh failed: %v", b.name, model, stopErr)
		}
	}
	b.swapProcess(next)

	// 4) If the model was serving, restart the rebuilt process in the
	//    background. The scheduler does not need a swap-done report: it is
	//    not the scheduler that initiated this start, and it re-observes the
	//    process state on the next request.
	if exists && wasReady {
		np := newProc
		go func() {
			timeout := b.healthCheckTimeout()
			if opt, ok := np.(process.ProcessWithOptions); ok {
				_ = opt.EnsureReadyWithOptions(b.shutdownCtx, timeout, process.Options{})
			} else {
				_ = np.EnsureReady(b.shutdownCtx, timeout)
			}
		}()
	}

	b.logger.Infof("%s: refreshed model %s (wasReady=%v, exists=%v)", b.name, model, wasReady, exists)
	return nil
}

func (b *baseRouter) notifyProcessed() {
	if b.testProcessed != nil {
		b.testProcessed <- struct{}{}
	}
}

func (b *baseRouter) run() {
	defer close(b.runDone)

	for {
		select {
		case req := <-b.shutdownCh:
			b.handleShutdown(req)
			return

		case req := <-b.handlerCh:
			b.schedule.OnRequest(req)
			b.notifyProcessed()

		case req := <-b.cancelCh:
			b.schedule.OnCancel(req)
			b.notifyProcessed()

		case req := <-b.unloadCh:
			b.schedule.OnUnload(req.targets, req.timeout)
			close(req.respond)
			b.notifyProcessed()

		case req := <-b.refreshCh:
			req.respond <- b.handleRefreshModel(req)
			b.notifyProcessed()

		case ev := <-b.swapDoneCh:
			b.schedule.OnSwapDone(ev)
			b.notifyProcessed()

		case ev := <-b.serveDoneCh:
			b.schedule.OnServeDone(ev)
		}
	}
}

// grant sends a response back to the caller of ServeHTTP and tells us
// whether the caller was still there to receive it.
//
// Each ServeHTTP creates a fresh, UNBUFFERED respond channel and parks in
// a select waiting on it. "Unbuffered" is the important word: a send only
// completes when the other side is actively receiving. So if this send
// succeeds, we know for a fact the caller picked up the response and will
// act on it. If the caller has already given up (its request context was
// cancelled, e.g. the HTTP client disconnected) or the router is shutting
// down, the send never lands, one of the other select cases fires, and we
// report back that the grant did NOT happen.
//
// That distinction matters for in-flight bookkeeping — see GrantServe.
func (b *baseRouter) grant(req scheduler.HandlerReq, resp scheduler.HandlerResp) bool {
	select {
	case req.Respond <- resp:
		return true
	case <-req.Ctx.Done():
		return false
	case <-b.shutdownCtx.Done():
		return false
	}
}

// ModelState implements scheduler.Effects.
func (b *baseRouter) ModelState(modelID string) (process.ProcessState, bool) {
	p, ok := b.processesAt()[modelID]
	if !ok {
		var zero process.ProcessState
		return zero, false
	}
	return p.State(), true
}

// ModelManual implements scheduler.Effects.
func (b *baseRouter) ModelManual(modelID string) (bool, bool) {
	mc, ok := b.configAt().Models[modelID]
	return ok && mc.ManualOnly, ok
}

// StartSwap implements scheduler.Effects, launching the swap goroutine.
func (b *baseRouter) StartSwap(modelID string, evict []string, opts process.Options) {
	go b.doSwap(modelID, evict, opts)
}

// GrantError implements scheduler.Effects.
func (b *baseRouter) GrantError(req scheduler.HandlerReq, err error) {
	b.grant(req, scheduler.HandlerResp{Err: err})
}

// GrantServe implements scheduler.Effects. It hands the caller a wrapped
// p.ServeHTTP (via trackedServe) so the run loop hears about the request
// finishing, and reports whether the caller received it. The scheduler bumps
// its in-flight count only on a true return: if grant() returns false the
// caller already walked away and trackedServe will never run, so no matching
// decrement will ever arrive — incrementing would strand the counter at >0 and
// the router would never again be willing to evict this model.
func (b *baseRouter) GrantServe(req scheduler.HandlerReq, modelID string) bool {
	p := b.processesAt()[modelID]
	return b.grant(req, scheduler.HandlerResp{HandleFunc: b.trackedServe(modelID, p)})
}

// StopProcesses implements scheduler.Effects, stopping the named processes in
// parallel and blocking until all have stopped.
func (b *baseRouter) StopProcesses(timeout time.Duration, ids []string) {
	var wg sync.WaitGroup
	for _, id := range ids {
		p, ok := b.processesAt()[id]
		if !ok {
			continue
		}
		wg.Add(1)
		go func(id string, p process.Process) {
			defer wg.Done()
			if err := p.Stop(timeout); err != nil {
				b.logger.Warnf("%s: stopping %s failed: %v", b.name, id, err)
			}
		}(id, p)
	}
	wg.Wait()
}

// trackedServe is the wrapper that closes the loop on in-flight tracking.
// It runs p.ServeHTTP normally; the only added behaviour is a deferred
// send on serveDoneCh after the handler returns. That send is what tells
// the run loop "this model now has one fewer request in flight — go look
// at the queue again, you may be able to start a swap you previously had
// to defer."
//
// The select on shutdownCtx.Done() is a release valve: if the router is
// already shutting down, nobody is reading serveDoneCh, so we drop the
// notification rather than blocking the HTTP goroutine forever.
func (b *baseRouter) trackedServe(modelID string, p process.Process) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			select {
			case b.serveDoneCh <- scheduler.ServeDoneEvent{ModelID: modelID}:
			case <-b.shutdownCtx.Done():
			}
		}()
		p.ServeHTTP(w, r)
	}
}

func (b *baseRouter) doSwap(modelID string, toStop []string, opts process.Options) {
	timeout := b.healthCheckTimeout()

	var wg sync.WaitGroup
	for _, mID := range toStop {
		wg.Add(1)
		go func(p process.Process, id string) {
			defer wg.Done()
			if err := p.Stop(timeout); err != nil {
				b.logger.Warnf("%s: stopping %s failed: %v", b.name, id, err)
			}
		}(b.processesAt()[mID], mID)
	}
	wg.Wait()

	// EnsureReady rather than a State() check followed by Run: the router must
	// not assume anything about the process. Deciding out here means acting on
	// a snapshot that the process's own run loop can invalidate at any moment —
	// a TTL unload landing in that window used to leave the swap waiting on a
	// process nobody was ever going to start (issue #946). EnsureReady makes
	// the same decision inside the process, where the state is owned.
	//
	// The WithOptions variant applies any per-load options (a GPU override);
	// it falls back to the plain EnsureReady when a process does not implement
	// optional options, so the behaviour is identical to before in that case.
	target := b.processesAt()[modelID]
	var err error
	if opt, ok := target.(process.ProcessWithOptions); ok {
		err = opt.EnsureReadyWithOptions(b.shutdownCtx, timeout, opts)
	} else {
		err = target.EnsureReady(b.shutdownCtx, timeout)
	}
	if err != nil && b.shutdownCtx.Err() == nil {
		// Quiet during shutdown: every in-flight swap fails at once there, and
		// that is expected rather than worth a warning per model.
		b.logger.Warnf("%s: starting %s failed: %v", b.name, modelID, err)
	}

	select {
	case b.swapDoneCh <- scheduler.SwapDone{ModelID: modelID, Err: err}:
	case <-b.shutdownCtx.Done():
	}
}

func (b *baseRouter) handleShutdown(req shutdownReq) {
	shutdownErr := fmt.Errorf("%s is shutting down", b.name)

	// Cancel shutdownCtx first so any waiter that is currently parked on
	// its respond channel can exit via its own shutdownCtx.Done() branch.
	// The OnShutdown grants below then either land (waiter happened to receive
	// before noticing shutdown) or fall through immediately via grant's
	// shutdownCtx case — either way the waiter sees a non-OK response.
	// This does NOT touch processes: their lifetime is procCtx, cancelled
	// only after the graceful Stop() calls below have reaped them.
	b.shutdownFn()

	b.schedule.OnShutdown(shutdownErr)

	stopTimeout := req.timeout
	if stopTimeout <= 0 {
		stopTimeout = b.healthCheckTimeout()
	}

	var wg sync.WaitGroup
	for i, p := range b.processesAt() {
		wg.Add(1)
		go func(id string, p process.Process) {
			defer wg.Done()
			if err := p.Stop(stopTimeout); err != nil {
				b.logger.Warnf("%s failed to stop process %s: %v", b.name, id, err)
			}
		}(i, p)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	if req.timeout > 0 {
		select {
		case <-done:
		case <-time.After(req.timeout):
			<-done
		}
	} else {
		<-done
	}

	// Every process is stopped (children reaped via Stop()). Cancel procCtx so
	// the process run-loop goroutines exit; they are already StateStopped, so
	// this is a clean no-op kill rather than a forced teardown.
	b.procCancel()

	req.respond <- nil
}

func (b *baseRouter) healthCheckTimeout() time.Duration {
	t := time.Duration(b.configAt().HealthCheckTimeout) * time.Second
	if t <= 0 {
		return 30 * time.Second
	}
	return t
}

// unloadTimeout returns the graceful stop timeout for a model. Config parsing
// guarantees both the global and per-model unloadTimeout are populated (a zero
// model value is rewritten to the global default on parse), so no zero handling
// is needed here.
func (b *baseRouter) unloadTimeout(modelID string) time.Duration {
	cfg := b.configAt()
	if mc, ok := cfg.Models[modelID]; ok {
		return time.Duration(mc.UnloadTimeout) * time.Second
	}
	return time.Duration(cfg.UnloadTimeout) * time.Second
}

func (b *baseRouter) Handles(model string) bool {
	_, ok := b.processesAt()[model]
	return ok
}

func (b *baseRouter) ProcessLogger(modelID string) (*logmon.Monitor, bool) {
	if p, ok := b.processesAt()[modelID]; ok {
		return p.Logger(), true
	}
	return nil, false
}

// RunningModels returns the current state of every process that is not stopped
// or shut down. The processes map keys are fixed at construction and State()
// is a snapshot, so this is safe to call without the run loop.
func (b *baseRouter) RunningModels() map[string]process.ProcessState {
	running := make(map[string]process.ProcessState)
	for id, p := range b.processesAt() {
		st := p.State()
		if st == process.StateStopped || st == process.StateShutdown {
			continue
		}
		running[id] = st
	}
	return running
}

// ProcessGPU reports the GPU a named model is loaded onto. Processes are
// asked only while running; a stopped process reports "" by construction.
func (b *baseRouter) ProcessGPU(modelID string) string {
	p, ok := b.processesAt()[modelID]
	if !ok {
		return ""
	}
	if gpu, ok := p.(interface{ GPU() string }); ok {
		if st := p.State(); st != process.StateStopped && st != process.StateShutdown {
			return gpu.GPU()
		}
	}
	return ""
}

// Unload stops the named models, or every running model when none are named.
// It blocks until each targeted process has stopped.
//
// The request is funneled through the run loop so eviction is coordinated
// with the rest of the router's state: pending swap waiters for an
// unloaded model are released with an error, queued requests for unloaded
// models are dropped, and any deferred swaps that were waiting on those
// models become eligible to start.
//
// In-flight requests being served by an unloaded process are not waited
// for — Stop kills the upstream, those callers see whatever error the
// reverse proxy surfaces and may retry. Their trackedServe defers fire
// normally and decrement inFlight as the dying handlers return.
//
// A timeout <= 0 unloads each targeted model with its configured
// unloadTimeout: targets sharing a timeout are stopped in parallel within one
// unload request, and the requests are processed smallest timeout first. The
// requests are sequential, so a hung stop on a large model (long timeouts
// usually mean multi-node unloads) cannot delay reclaiming the quick ones
// queued behind it. A positive timeout overrides the configured values and
// stops every target with that timeout.
func (b *baseRouter) Unload(timeout time.Duration, models ...string) {
	targets := models
	if len(targets) == 0 {
		procs := b.processesAt()
		targets = make([]string, 0, len(procs))
		for id := range procs {
			targets = append(targets, id)
		}
	}
	if len(targets) == 0 {
		return
	}

	if timeout > 0 {
		b.sendUnload(targets, timeout)
		return
	}
	buckets := make(map[time.Duration][]string)
	for _, id := range targets {
		t := b.unloadTimeout(id)
		buckets[t] = append(buckets[t], id)
	}
	timeouts := make([]time.Duration, 0, len(buckets))
	for t := range buckets {
		timeouts = append(timeouts, t)
	}
	sort.Slice(timeouts, func(i, j int) bool { return timeouts[i] < timeouts[j] })
	for _, t := range timeouts {
		b.sendUnload(buckets[t], t)
	}
}

// sendUnload funnels one unload request through the run loop and blocks until
// the scheduler has stopped the targeted processes.
func (b *baseRouter) sendUnload(targets []string, timeout time.Duration) {
	req := unloadReq{targets: targets, timeout: timeout, respond: make(chan struct{})}
	select {
	case b.unloadCh <- req:
	case <-b.runDone:
		return
	}
	<-req.respond
}

func (b *baseRouter) Shutdown(timeout time.Duration) error {
	if !b.shuttingDown.CompareAndSwap(false, true) {
		return fmt.Errorf("%s shutdown already in progress", b.name)
	}
	req := shutdownReq{timeout: timeout, respond: make(chan error, 1)}
	select {
	case b.shutdownCh <- req:
	case <-b.runDone:
		return nil
	}
	return <-req.respond
}

func (b *baseRouter) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if b.shuttingDown.Load() {
		swaputil.SendError(w, req, fmt.Errorf("%s is shutting down", b.name))
		return
	}

	data, err := swaputil.FetchContext(req, b.configAt())
	if err != nil {
		swaputil.SendError(w, req, err)
		return
	}

	// Ignored websocket connections are deliberately kept outside the
	// scheduler: they cannot start or queue a model, consume concurrency, or
	// prevent another model from swapping the process out. A process may stop
	// immediately after this readiness check; dropping that websocket is the
	// intended tradeoff of opting out of lifecycle tracking.
	if swaputil.ShouldIgnoreWebsocket(req, b.configAt()) {
		p, ok := b.processesAt()[data.ModelID]
		if !ok {
			swaputil.SendError(w, req, scheduler.ErrModelNotFound)
			return
		}
		if p.State() != process.StateReady {
			swaputil.SendResponse(w, req, http.StatusConflict,
				fmt.Sprintf("model %s is not loaded; ignored websocket requests cannot start it", data.ModelID))
			return
		}
		p.ServeHTTP(w, req)
		return
	}

	hr := scheduler.HandlerReq{
		Model: data.ModelID,
		Ctx:   req.Context(),
		// Unbuffered: a successful send on Respond proves the waiter is
		// alive and consuming. grant() relies on this to avoid handing a
		// handleFunc to a cancelled waiter and leaking the inFlight count.
		Admit: make(chan error, 1),
		// A GET against the model endpoint is a load/keepalive probe (the
		// dashboard's load buttons use exactly this shape), not inference.
		// Manual-only models honor these so the operator can still start
		// the model, while POST inference calls are rejected until it is
		// loaded.
		LoadRequest: req.Method == http.MethodGet,
		Respond:     make(chan scheduler.HandlerResp),
		PositionCh:  make(chan int, 1),
		GpuOverride: swaputil.GpuOverrideFromContext(req.Context()),
	}

	select {
	case b.handlerCh <- hr:
	case <-req.Context().Done():
		return
	case <-b.shutdownCtx.Done():
		swaputil.SendError(w, req, fmt.Errorf("%s is shutting down", b.name))
		return
	}

	var admissionErr error
	select {
	case admissionErr = <-hr.Admit:
	case <-req.Context().Done():
		select {
		case b.cancelCh <- hr:
		case <-b.shutdownCtx.Done():
		}
		return
	case <-b.shutdownCtx.Done():
		swaputil.SendError(w, req, fmt.Errorf("%s is shutting down", b.name))
		return
	}
	if admissionErr != nil {
		swaputil.SendError(w, req, admissionErr)
		return
	}

	isModelReady := false
	if p, ok := b.processesAt()[data.ModelID]; ok {
		isModelReady = p.State() == process.StateReady
	}
	shouldShowLoading := data.Streaming && data.SendLoadingState && isLoadingPath(req.URL.Path) && !isModelReady

	var lw *loadingWriter
	cancelLoad := func() {}
	if shouldShowLoading {
		var swapCtx context.Context
		swapCtx, cancelLoad = context.WithCancel(req.Context())
		lw = newLoadingWriter(b.logger, data.ModelID, w, req)
		go lw.start(swapCtx)
		go func() {
			for {
				select {
				case pos := <-hr.PositionCh:
					lw.setUpdate(fmt.Sprintf("Queue position: #%d", pos))
				case <-swapCtx.Done():
					return
				}
			}
		}()
	}

	// finishLoading stops the loading stream and fences its goroutine off from
	// the ResponseWriter before the real handler (or ServeHTTP's return)
	// reclaims it. release() must run even when waitForCompletion times out:
	// otherwise a still-streaming goroutine flushes a finalized response and
	// panics on the recycled *bufio.Writer.
	//
	// A non-nil streamErr is framed into the stream first, while writes still
	// reach the client: the 200 is already committed, so this is the only way
	// the error can be reported at all.
	finishLoading := func(streamErr error) {
		cancelLoad()
		if lw != nil {
			lw.waitForCompletion(1 * time.Second)
			if streamErr != nil {
				lw.sendError(streamErr)
			}
			lw.release()
		}
	}

	// reportError sends err as a normal error response, for the paths where no
	// loading stream was live to carry it in-band. When one was, finishLoading
	// has already framed it into the stream and a second report would append a
	// bare JSON line that SSE parsers discard.
	reportError := func(err error) {
		if lw == nil {
			swaputil.SendError(w, req, err)
		}
	}

	var resp scheduler.HandlerResp
	select {
	case resp = <-hr.Respond:
		// Pass the dispatch error in so it is framed into the stream before
		// release fences the writer; a nil error just ends the stream.
		finishLoading(resp.Err)
	case <-req.Context().Done():
		// The client is gone, so there is nobody to report to.
		finishLoading(nil)
		// Notify the scheduler so it can prune this request from its queue
		// and swap waiters. Without this, a queued request whose client left
		// would sit in the scheduler until drainQueue eventually starts a
		// wasted model load for it.
		select {
		case b.cancelCh <- hr:
		case <-b.shutdownCtx.Done():
		}
		return
	case <-b.shutdownCtx.Done():
		shutdownErr := fmt.Errorf("%s is shutting down", b.name)
		finishLoading(shutdownErr)
		reportError(shutdownErr)
		return
	}

	if resp.Err != nil {
		reportError(resp.Err)
		return
	}
	resp.HandleFunc(w, req)
}
