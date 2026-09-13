package server

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/perf"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/store"
)

// vramOracle implements router.VRAMOracle over the two things the server owns
// that the router does not: the performance monitor, which polls live per-GPU
// memory, and the store, which persists what each model needed on its last
// successful load.
//
// Measurements are cached in memory because the router asks on its run loop:
// the admission check must never wait on SQL. Writes go to the database on a
// background goroutine for the same reason.
type vramOracle struct {
	perf   *perf.Monitor
	store  *store.Store
	logger *logmon.Monitor

	mu       sync.RWMutex
	measured map[string]int
}

// newVRAMOracle loads the persisted measurements once. A nil perf monitor or
// store yields a nil oracle, which disables the VRAM check in the router.
func newVRAMOracle(perfMon *perf.Monitor, st *store.Store, logger *logmon.Monitor) router.VRAMOracle {
	if perfMon == nil || st == nil {
		return nil
	}
	o := &vramOracle{perf: perfMon, store: st, logger: logger, measured: map[string]int{}}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if rows, err := st.ModelVRAMAll(ctx); err == nil {
		o.measured = rows
	} else if logger != nil {
		logger.Warnf("vram: could not load stored measurements: %v", err)
	}
	return o
}

// newestGPUSamples reduces the performance monitor's ring to the most recent
// sample per GPU. perf.Current() hands back the whole ring, so a later sample
// for the same device wins. Shared by the VRAM oracle and the /api/gpus
// handler so there is one definition of "the current reading".
func newestGPUSamples(perfMon *perf.Monitor) map[int]perf.GpuStat {
	if perfMon == nil {
		return nil
	}
	_, gpuStats := perfMon.Current()
	return newestPerGPU(gpuStats)
}

// newestPerGPU keeps the latest sample for each GPU id. Split out from
// newestGPUSamples so the reduction can be tested without a live monitor.
func newestPerGPU(stats []perf.GpuStat) map[int]perf.GpuStat {
	if len(stats) == 0 {
		return nil
	}
	newest := make(map[int]perf.GpuStat, len(stats))
	for _, g := range stats {
		if prev, ok := newest[g.ID]; ok && g.Timestamp.Before(prev.Timestamp) {
			continue
		}
		newest[g.ID] = g
	}
	return newest
}

// DeviceMemory implements router.VRAMOracle.
func (o *vramOracle) DeviceMemory() map[string]router.DeviceMemory {
	newest := newestGPUSamples(o.perf)
	out := make(map[string]router.DeviceMemory, len(newest))
	for id, g := range newest {
		if g.MemTotalMB <= 0 {
			continue
		}
		out[strconv.Itoa(id)] = router.DeviceMemory{UsedMB: g.MemUsedMB, TotalMB: g.MemTotalMB}
	}
	return out
}

// MeasuredVRAMMB implements router.VRAMOracle.
func (o *vramOracle) MeasuredVRAMMB(modelID string) (int, bool) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	mb, ok := o.measured[modelID]
	return mb, ok
}

// RecordVRAMMB implements router.VRAMOracle. The in-memory value is updated
// synchronously so the next admission check sees it; the database write is
// backgrounded so the router's run loop never waits on it.
func (o *vramOracle) RecordVRAMMB(modelID string, vramMB int) {
	if vramMB <= 0 {
		return
	}
	o.mu.Lock()
	o.measured[modelID] = vramMB
	o.mu.Unlock()

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := o.store.RecordModelVRAM(ctx, modelID, vramMB); err != nil && o.logger != nil {
			o.logger.Warnf("vram: could not persist measurement for %s: %v", modelID, err)
		}
	}()
}
