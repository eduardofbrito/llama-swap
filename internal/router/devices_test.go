package router

import (
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// gpuGroupYAML has one persistent group pinned to its own GPU by the model's own
// env, and one dynamic group that schedules three members over two GPUs.
const gpuGroupYAML = `
healthCheckTimeout: 15
startPort: 5800
models:
  fixed:
    cmd: fake --port ${PORT}
    env:
      - "CUDA_VISIBLE_DEVICES=0"
  a:
    cmd: fake --port ${PORT}
  b:
    cmd: fake --port ${PORT}
  c:
    cmd: fake --port ${PORT}
groups:
  fixo:
    swap: false
    persistent: true
    exclusive: false
    members:
      - fixed
  dinamico:
    swap: true
    exclusive: false
    gpus: [1, 2]
    members:
      - a
      - b
      - c
`

// newGPUGroupBase builds a router over gpuGroupYAML with fake processes and the
// real groupSwapper, so eviction, pool sizing and device placement are all
// exercised the way a real config wires them.
func newGPUGroupBase(t *testing.T) (*baseRouter, map[string]*fakeProcess) {
	t.Helper()

	conf, err := config.LoadConfigFromReader(strings.NewReader(gpuGroupYAML))
	if err != nil {
		t.Fatalf("LoadConfigFromReader: %v", err)
	}

	modelToGroup := make(map[string]string)
	for gid, gcfg := range conf.Routing.Router.Settings.Groups {
		for _, mid := range gcfg.Members {
			modelToGroup[mid] = gid
		}
	}

	procs := map[string]*fakeProcess{}
	models := map[string]process.Process{}
	for mid := range modelToGroup {
		p := newFakeProcess(mid)
		p.autoReady = true
		procs[mid] = p
		models[mid] = p
	}

	swapper := &groupSwapper{config: conf, modelToGroup: modelToGroup}
	return newTestBaseWithConfig(t, conf, models, swapper), procs
}

// device reads back the GPU the router placed a model on, via the env it handed
// the process.
func device(t *testing.T, p *fakeProcess) string {
	t.Helper()
	env := p.env()
	if len(env) == 0 {
		return ""
	}
	if len(env) != 1 {
		t.Fatalf("%s: runtime env=%v, want a single device entry", p.id, env)
	}
	value, found := strings.CutPrefix(env[0], "CUDA_VISIBLE_DEVICES=")
	if !found {
		t.Fatalf("%s: runtime env=%q, want a CUDA_VISIBLE_DEVICES entry", p.id, env[0])
	}
	return value
}

func TestBaseRouter_GroupGPUsPlaceMembersOnFreeDevices(t *testing.T) {
	b, procs := newGPUGroupBase(t)

	// Two members, two free GPUs: both load, neither displaces the other.
	serveModel(t, b, "a")
	serveModel(t, b, "b")

	if got := device(t, procs["a"]); got != "1" {
		t.Fatalf("a on device %q, want 1", got)
	}
	if got := device(t, procs["b"]); got != "2" {
		t.Fatalf("b on device %q, want 2", got)
	}
	if got := procs["a"].stopCalls.Load(); got != 0 {
		t.Fatalf("a stopCalls=%d want 0, the group still has a free GPU", got)
	}
	if procs["a"].State() != process.StateReady {
		t.Fatalf("a state=%q want ready", procs["a"].State())
	}
}

func TestBaseRouter_GroupGPUsEvictLeastRecentlyUsedWhenFull(t *testing.T) {
	b, procs := newGPUGroupBase(t)

	serveModel(t, b, "a") // GPU 1
	serveModel(t, b, "b") // GPU 2

	// Both GPUs are busy. c must displace a — the least recently used — and take
	// over the GPU a was holding. b, used more recently, stays put.
	serveModel(t, b, "c")

	if got := procs["a"].stopCalls.Load(); got != 1 {
		t.Fatalf("a stopCalls=%d want 1, it is the least recently used", got)
	}
	if got := device(t, procs["c"]); got != "1" {
		t.Fatalf("c on device %q, want 1 — the GPU freed by evicting a", got)
	}
	if got := procs["b"].stopCalls.Load(); got != 0 {
		t.Fatalf("b stopCalls=%d want 0, it was used more recently than a", got)
	}
	if procs["b"].State() != process.StateReady {
		t.Fatalf("b state=%q want ready", procs["b"].State())
	}

	// Bringing a back evicts b, now the least recently used, and a takes the GPU
	// b frees — not the GPU 1 it ran on before, which c now holds.
	serveModel(t, b, "a")

	if got := procs["b"].stopCalls.Load(); got != 1 {
		t.Fatalf("b stopCalls=%d want 1, it is now the least recently used", got)
	}
	if got := device(t, procs["a"]); got != "2" {
		t.Fatalf("a on device %q, want 2 — GPU 1 is held by c", got)
	}
	if got := procs["c"].stopCalls.Load(); got != 0 {
		t.Fatalf("c stopCalls=%d want 0", got)
	}
}

// A persistent group is not device-managed: its members keep the GPU their own
// env pins them to, and the dynamic group's rotation never unloads them.
func TestBaseRouter_GroupGPUsLeavePersistentMembersAlone(t *testing.T) {
	b, procs := newGPUGroupBase(t)

	serveModel(t, b, "fixed")
	serveModel(t, b, "a")
	serveModel(t, b, "b")
	serveModel(t, b, "c")

	if env := procs["fixed"].env(); env != nil {
		t.Fatalf("fixed got runtime env %v, want none — its group declares no gpus", env)
	}
	if got := procs["fixed"].stopCalls.Load(); got != 0 {
		t.Fatalf("fixed stopCalls=%d want 0, a persistent member is never evicted", got)
	}
	if procs["fixed"].State() != process.StateReady {
		t.Fatalf("fixed state=%q want ready", procs["fixed"].State())
	}
}

// Serving a model outside the pool between two swaps must not cost a pool slot.
// The persistent model is never an eviction candidate, so it cannot push a
// dynamic member out of the pool's recency order.
func TestBaseRouter_GroupGPUsIgnoreNonCandidatesInRecency(t *testing.T) {
	b, procs := newGPUGroupBase(t)

	serveModel(t, b, "a")
	serveModel(t, b, "b")
	serveModel(t, b, "fixed") // most recently used overall, but never evictable
	serveModel(t, b, "c")

	if got := procs["a"].stopCalls.Load(); got != 1 {
		t.Fatalf("a stopCalls=%d want 1", got)
	}
	if got := procs["b"].stopCalls.Load(); got != 0 {
		t.Fatalf("b stopCalls=%d want 0 — 'fixed' must not consume b's pool slot", got)
	}
}

func TestBaseRouter_GroupGPUsConfigRejectsPersistent(t *testing.T) {
	const bad = `
healthCheckTimeout: 15
startPort: 5800
models:
  a:
    cmd: fake --port ${PORT}
groups:
  g:
    swap: true
    persistent: true
    gpus: [0]
    members: [a]
`
	_, err := config.LoadConfigFromReader(strings.NewReader(bad))
	if err == nil {
		t.Fatal("expected an error: gpus on a persistent group never frees a device")
	}
	if !strings.Contains(err.Error(), "persistent") {
		t.Fatalf("error = %v, want it to name the persistent conflict", err)
	}
}
