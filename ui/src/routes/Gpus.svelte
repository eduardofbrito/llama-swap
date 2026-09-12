<script lang="ts">
  import { link } from "svelte-spa-router";
  import { Gpu as GpuIcon, ArrowUpToLine, ArrowDownToLine } from "@lucide/svelte";
  import { gpus, models, unloadSingleModel } from "../stores/api";
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

  type DotColor = "grey" | "yellow" | "green";
  function statusDotColor(model: Model): DotColor {
    if (model.state === "ready") return "green";
    if (model.state === "starting" || model.state === "stopping") return "yellow";
    return "grey";
  }

  const dotClass: Record<DotColor, string> = {
    grey: "bg-muted-foreground/40",
    yellow: "bg-warning",
    green: "bg-success",
  };
</script>

<div class="p-2">
  <div class="mt-4 mb-4">
    <h3 class="text-lg font-semibold">GPUs</h3>
    <p class="text-sm text-muted-foreground">
      GPUs available on this host, the models loaded on each one, and controls
      to load or unload any model defined in the configuration.
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
        <article
          class={cn(
            "rounded-md border p-4 transition-colors",
            loaded.length > 0 ? "bg-muted/20" : "bg-transparent",
          )}
        >
          <div class="mb-3 flex items-center gap-2">
            <GpuIcon class="size-4 text-muted-foreground" />
            <h5 class="font-medium">{gpu.label}</h5>
            {#if loaded.length === 0}
              <span class="ml-auto rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                Idle
              </span>
            {:else}
              <span class="ml-auto rounded-full bg-success/15 px-2 py-0.5 text-xs text-success">
                In use
              </span>
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
                  <span class="shrink-0 text-xs text-muted-foreground">{model.state}</span>
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
