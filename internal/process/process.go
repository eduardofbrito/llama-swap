package process

import (
	"context"
	"net/http"
	"time"

	"github.com/mostlygeek/llama-swap/internal/logmon"
)

type ProcessState string

const (
	StateStopped  ProcessState = ProcessState("stopped")
	StateStarting ProcessState = ProcessState("starting")
	StateReady    ProcessState = ProcessState("ready")
	StateStopping ProcessState = ProcessState("stopping")

	// process is shutdown and will not be restarted
	StateShutdown ProcessState = ProcessState("shutdown")
)

// Options carries optional, per-load parameters for the methods in the
// ProcessWithOptions interface. An empty Options value behaves exactly like
// the unparameterized Process methods.
type Options struct {
	// GpuOverride, when non-empty, is injected as the CUDA_VISIBLE_DEVICES
	// environment variable (overriding any set in the model config) for the
	// process launched by this load. This is how the UI's per-model GPU
	// selector chooses which GPU a model is loaded onto; when empty the model
	// uses the GPU configured for it.
	GpuOverride string
}

type Process interface {
	// Run starts the process blocks until the process is terminated.
	// The timeout parameter controls how long to wait for the process to get
	// to a ready state to process traffic
	Run(timeout time.Duration) error

	// WaitReady blocks until the process is ready to serve requests
	// or the context is cancelled. It returns nil when the process is ready
	//
	// WaitReady only subscribes, it never starts anything, and it cannot tell
	// "stopped, but a start is coming" apart from "stopped, and nothing is
	// coming" — a subscription that arrives after a start has already failed or
	// been aborted waits for a process nobody is going to start. Pass a context
	// with a deadline if that matters. Callers that want the process serving
	// should use EnsureReady, which has neither problem.
	WaitReady(context.Context) error

	// EnsureReady brings the process to a ready state and blocks until it is
	// serving, the start fails, or ctx is cancelled. The timeout parameter
	// controls how long to wait for the process to become ready.
	//
	// Unlike Run, the decision of whether a start is needed is made inside the
	// process's own state machine, so callers never inspect State() first and
	// cannot race a concurrent transition:
	//
	//	ready    -> returns nil immediately
	//	stopped  -> starts the process and waits for it to become ready
	//	stopping -> waits for the stop to finish, then starts
	//	shutdown -> returns an error
	EnsureReady(ctx context.Context, timeout time.Duration) error

	// Stop blocks until the process has terminated. It returns nil when
	// the process terminated as expected (exit 0)
	Stop(timeout time.Duration) error

	// State returns the current state of the process
	// Note: this is a snapshot of the state at the time of the call
	// and may change at any time after the call returns.
	State() ProcessState

	// ServeHTTP forwards requests to the underlying process
	// Calling it when the process is not ready will result in a
	// 503 response with an error body identifying llama-swap as the source
	ServeHTTP(http.ResponseWriter, *http.Request)

	// Logger returns the monitor that captures this process's stdout/stderr.
	Logger() *logmon.Monitor
}

// ProcessWithOptions is implemented by processes that accept per-load
// Options. It is a superset of Process: the base methods keep their existing
// signatures, so callers that do not need per-load options can keep treating a
// process as a plain Process. Routers type-assert to this interface at load
// time so that optional capabilities (like a GPU override) degrade gracefully
// when a process does not support them.
type ProcessWithOptions interface {
	Process

	// RunWithOptions behaves like Run but applies Options to the start.
	RunWithOptions(timeout time.Duration, opts Options) error

	// EnsureReadyWithOptions behaves like EnsureReady but applies Options to
	// the start. Options are only consulted when this call actually starts the
	// process; they are ignored when it merely observes an existing state.
	EnsureReadyWithOptions(ctx context.Context, timeout time.Duration, opts Options) error
}
