<script lang="ts">
  import { link } from "svelte-spa-router";
  import { Gpu as GpuIcon } from "@lucide/svelte";
  import { gpus, models } from "../stores/api";
  import type { Model } from "../lib/types";
  import { cn } from "$lib/utils.js";

  // Models currently loaded onto a GPU, grouped by the GPU value their
  // process reports (model.gpu, a CUDA_VISIBLE_DEVICES value). A value can
  // list several devices ("0,1"); each listed device maps to the model.
  let loadedByGpu = $derived.by(() => {
    const map = new Map<string, Model[]>();
    for (const model of $models) {
      if (model.peerID) continue;
      const gpu = (model.gpu ?? "").trim();
      if (!gpu || model.state === "stopped" || model.state === "shutdown") {
        continue;
      }
      for (const part of gpu.split(",").map((v) => v.trim()).filter(Boolean)) {
        const list = map.get(part) ?? [];
        list.push(model);
        map.set(part, list);
      }
    }
    return map;
  });

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
      GPUs disponíveis no host e o modelo atualmente carregado em cada uma.
    </p>
  </div>

  {#if $gpus.length === 0}
    <div class="rounded-lg border p-6 text-sm text-muted-foreground">
      Nenhuma GPU detectada no host (ou indisponível para seleção).
    </div>
  {:else}
    <div class="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
      {#each $gpus as gpu (gpu.index)}
        {@const loaded = loadedByGpu.get(String(gpu.index)) ?? []}
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
                Livre
              </span>
            {:else}
              <span class="ml-auto rounded-full bg-success/15 px-2 py-0.5 text-xs text-success">
                Em uso
              </span>
            {/if}
          </div>

          {#if loaded.length === 0}
            <p class="text-sm text-muted-foreground">Nenhum modelo carregado.</p>
          {:else}
            <ul class="flex flex-col gap-1.5">
              {#each loaded as model (model.id)}
                <li class="flex items-center gap-2 rounded-md border bg-background px-3 py-2 text-sm">
                  <a href="/models/{encodeURIComponent(model.id)}" use:link class="flex flex-1 items-center gap-2">
                    <span class={`size-2 shrink-0 rounded-full ${dotClass[statusDotColor(model)]}`}></span>
                    <span class="truncate font-medium">{model.id}</span>
                  </a>
                  <span class="shrink-0 text-xs text-muted-foreground">{model.state}</span>
                </li>
              {/each}
            </ul>
          {/if}
        </article>
      {/each}
    </div>
  {/if}
</div>
