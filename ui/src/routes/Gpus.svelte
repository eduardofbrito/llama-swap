<script lang="ts">
  import { onMount } from "svelte";
  import { link } from "svelte-spa-router";
  import { Gpu as GpuIcon, ArrowUpToLine, ArrowDownToLine, TriangleAlert } from "@lucide/svelte";
  import { gpus, models, unloadSingleModel, fetchGpus } from "../stores/api";
  import { formatGpuMemory, gpuMemoryPct } from "../lib/format";
  import { handleLoadModel } from "../stores/modelLoad";
  import type { GpuInfo, Model } from "../lib/types";
  import { cn } from "$lib/utils.js";
  import { Button } from "$lib/components/ui/button/index.js";

  // Key an operation as "gpuIndex:modelId" so concurrent actions on different
  // GPUs/models stay independent.
  let busy = $state<Record<string, boolean>>({});
  let errors = $state<Record<string, string>>({});

  // Models currently loaded onto a GPU, grouped by the GPU value their
  // process reports (model.gpu, a CUDA_VISIBLE_DEVICES value). A value can
  // list several devices ("0,1"); each listed device maps to the model.
  function loadedOn(gpu: GpuInfo): Model[] {
    const result: Model[] = [];
    for (const model of $models) {
      if (model.peerID) continue;
      const gpuValue = (model.gpu ?? "").trim();
      if (!gpuValue || model.state === "stopped" || model.state === "shutdown") {
        continue;
      }
      if (gpuValue.split(",").map((v) => v.trim()).includes(String(gpu.index))) {
        result.push(model);
      }
    }
    return result;
  }

  // Models defined in the config that are not currently loaded on this GPU:
  // they can be loaded onto it (a model running elsewhere gets swapped over).
  function availableFor(gpu: GpuInfo): Model[] {
    const loaded = loadedOn(gpu).map((m) => m.id);
    return $models
      .filter((model) => !model.peerID && !model.unlisted && !loaded.includes(model.id))
      .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
  }

  function keyOf(gpu: GpuInfo, model: Model): string {
    return `${gpu.index}:${model.id}`;
  }

  async function handleLoad(gpu: GpuInfo, model: Model): Promise<void> {
    const key = keyOf(gpu, model);
    busy[key] = true;
    try {
      // Shared load helper: dedupes concurrent loads of the same model and
      // mirrors the dashboard's loading state; the GPU index pins the load.
      await handleLoadModel(model.id, gpu.index);
    } finally {
      busy[key] = false;
    }
  }

  async function handleUnload(gpu: GpuInfo, model: Model): Promise<void> {
    const key = keyOf(gpu, model);
    busy[key] = true;
    delete errors[key];
    try {
      await unloadSingleModel(model.id);
    } catch (error) {
      errors[key] = error instanceof Error ? error.message : "Failed to unload model";
    } finally {
      busy[key] = false;
    }
  }

  // Memory readings change constantly, so the page refreshes them on its own
  // rather than showing whatever was true when the SSE connection opened.
  const MEMORY_REFRESH_MS = 5000;
  onMount(() => {
    const tick = () => void fetchGpus().catch(() => {});
    tick();
    const timer = setInterval(tick, MEMORY_REFRESH_MS);
    return () => clearInterval(timer);
  });

  // A GPU with no llama-swap model on it should be empty. Anything above this
  // is another process: a training job, a second llama-swap, a desktop
  // session. The floor exists because drivers and compositors hold a little
  // memory on an otherwise idle card, and flagging that would be noise.
  const EXTERNAL_USE_FLOOR_MB = 512;

  type GpuStatus = "idle" | "in-use" | "external" | "unknown";

  // Models on this GPU that are actually holding its memory. A sleeping model
  // is still listed on the card — it will wake back onto this device — but it
  // released its VRAM, so it must not make the GPU read as "In use". Counting
  // it would also hide the case this page exists for: memory held by a process
  // llama-swap did not start.
  function residentOn(gpu: GpuInfo): Model[] {
    return loadedOn(gpu).filter((model) => model.state !== "sleeping");
  }

  // What the badge says about a GPU. "external" is the case worth surfacing:
  // memory is occupied but llama-swap did not put it there.
  function gpuStatus(gpu: GpuInfo, loadedCount: number): GpuStatus {
    if (loadedCount > 0) return "in-use";
    if (!gpu.totalMB) return "unknown";
    return (gpu.usedMB ?? 0) >= EXTERNAL_USE_FLOOR_MB ? "external" : "idle";
  }

  const statusLabel: Record<GpuStatus, string> = {
    idle: "Idle",
    "in-use": "In use",
    external: "Used externally",
    unknown: "Idle",
  };

  const statusClass: Record<GpuStatus, string> = {
    idle: "bg-muted text-muted-foreground",
    "in-use": "bg-success/15 text-success",
    external: "bg-warning/15 text-warning",
    unknown: "bg-muted text-muted-foreground",
  };

  const statusTitle: Record<GpuStatus, string> = {
    idle: "No models loaded and the device is essentially free",
    "in-use": "llama-swap has one or more models loaded on this GPU",
    external: "Memory is in use but llama-swap has no model here — another process is holding it",
    unknown: "No memory reading available (the performance monitor is off or has not sampled this device)",
  };

  // The bar turns amber for memory llama-swap did not allocate, so a glance at
  // the page separates "my models" from "someone else's".
  function barClass(status: GpuStatus): string {
    return status === "external" ? "bg-warning" : "bg-success";
  }

  type DotColor = "grey" | "yellow" | "green" | "blue" | "red";
  // Kept in step with statusDotColor in stores/modelLoad, so a model does not
  // change colour between here and the Models page.
  function statusDotColor(model: Model): DotColor {
    if (model.state === "ready") return "green";
    if (model.state === "sleeping") return "yellow";
    if (model.state === "starting") return "blue";
    if (model.state === "stopping") return "red";
    return "grey";
  }

  const dotClass: Record<DotColor, string> = {
    grey: "bg-muted-foreground/40",
    yellow: "bg-warning",
    green: "bg-success",
    blue: "bg-primary",
    red: "bg-destructive",
  };
</script>

<div class="p-2">
  <div class="mt-4 mb-4">
    <h3 class="text-lg font-semibold">GPUs</h3>
    <p class="text-sm text-muted-foreground">
      GPUs available on this host, their live memory use, the models loaded on
      each one, and controls to load or unload any model defined in the
      configuration. A GPU flagged <span class="text-warning">Used externally</span>
      has memory held by a process llama-swap did not start.
    </p>
  </div>

  {#if $gpus.length === 0}
    <div class="rounded-lg border p-6 text-sm text-muted-foreground">
      No GPUs detected on this host (or none available for selection).
    </div>
  {:else}
    <div class="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
      {#each $gpus as gpu (gpu.index)}
        {@const loaded = loadedOn(gpu)}
        {@const available = availableFor(gpu)}
        {@const status = gpuStatus(gpu, residentOn(gpu).length)}
        {@const memory = formatGpuMemory(gpu.usedMB, gpu.totalMB)}
        {@const pct = gpuMemoryPct(gpu.usedMB, gpu.totalMB)}
        <article
          class={cn(
            "rounded-md border p-4 transition-colors",
            loaded.length > 0 ? "bg-muted/20" : "bg-transparent",
          )}
        >
          <div class="mb-2 flex items-center gap-2">
            <GpuIcon class="size-4 text-muted-foreground" />
            <h5 class="font-medium">{gpu.label}</h5>
            <span
              class={cn("ml-auto flex items-center gap-1 rounded-full px-2 py-0.5 text-xs", statusClass[status])}
              title={statusTitle[status]}
            >
              {#if status === "external"}<TriangleAlert class="size-3" />{/if}
              {statusLabel[status]}
            </span>
          </div>

          <!-- Live device memory. This is the only place that shows memory
               held by processes llama-swap did not start. -->
          <div class="mb-3">
            {#if memory !== null && pct !== null}
              <div class="flex items-baseline justify-between gap-2 text-xs">
                <span class="text-muted-foreground">Memory</span>
                <span class="font-mono tabular-nums">{memory}</span>
              </div>
              <div class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div class={cn("h-full rounded-full transition-all", barClass(status))} style={`width: ${pct}%`}></div>
              </div>
              <div class="mt-1 text-right text-[0.625rem] text-muted-foreground">
                {pct.toFixed(0)}% used
              </div>
            {:else}
              <p class="text-xs text-muted-foreground">
                No memory reading (performance monitoring is off)
              </p>
            {/if}
          </div>

          {#if loaded.length === 0}
            <p class="text-sm text-muted-foreground">No models loaded.</p>
          {:else}
            <ul class="flex flex-col gap-1.5">
              {#each loaded as model (model.id)}
                {@const key = keyOf(gpu, model)}
                <li class="flex items-center gap-2 rounded-md border bg-background px-3 py-2 text-sm">
                  <a href="/models/{encodeURIComponent(model.id)}" use:link class="flex flex-1 items-center gap-2">
                    <span class={`size-2 shrink-0 rounded-full ${dotClass[statusDotColor(model)]}`}></span>
                    <span class="truncate font-medium">{model.id}</span>
                  </a>
                  <span
                    class="shrink-0 text-xs text-muted-foreground"
                    title={model.state === "sleeping"
                      ? "Sleeping: the process is alive with its weights in host RAM and holds no memory on this GPU. The next request wakes it here in seconds."
                      : undefined}
                  >
                    {model.state}
                  </span>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy[key]}
                    onclick={() => handleUnload(gpu, model)}
                    title={`Unload ${model.id}`}
                    class="h-7 shrink-0"
                  >
                    <ArrowDownToLine class="size-3.5" />
                    Unload
                  </Button>
                </li>
              {/each}
            </ul>
          {/if}

          {#if available.length > 0}
            <div class="mt-3 border-t pt-3">
              <p class="mb-2 text-xs font-medium text-muted-foreground">
                Load model
              </p>
              <div class="flex flex-col gap-1.5">
                {#each available as model (model.id)}
                  {@const key = keyOf(gpu, model)}
                  <div class="flex items-center gap-2 text-sm">
                    <span class="flex-1 truncate" title={model.id}>
                      {model.id}
                      {#if model.state !== "stopped" && model.state !== "shutdown"}
                        <span class="text-xs text-muted-foreground">
                          ({model.state} on another GPU)
                        </span>
                      {/if}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={busy[key]}
                      onclick={() => handleLoad(gpu, model)}
                      title={`Load ${model.id} onto ${gpu.label}`}
                      class="h-7 shrink-0"
                    >
                      <ArrowUpToLine class="size-3.5" />
                      Load
                    </Button>
                  </div>
                  {#if errors[key]}
                    <p class="mt-1 text-xs text-destructive">{errors[key]}</p>
                  {/if}
                {/each}
              </div>
            </div>
          {/if}
        </article>
      {/each}
    </div>
  {/if}
</div>
