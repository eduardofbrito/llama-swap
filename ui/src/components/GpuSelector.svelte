<script lang="ts">
  import * as Select from "$lib/components/ui/select/index.js";
  import Input from "$lib/components/ui/input/input.svelte";
  import { gpus } from "../stores/api";
  import {
    gpuSelections,
    setGpuForModel,
    clearGpuForModel,
    defaultGpuDisplay,
  } from "../stores/modelGpu";
  import type { Model } from "../lib/types";
  import { cn } from "$lib/utils.js";

  interface Props {
    model: Model;
    /** "md" for list rows, "sm" for the detail header. */
    size?: "md" | "sm";
  }

  let { model, size = "md" }: Props = $props();

  // The host reports no selectable GPU: the selector is hidden and the model
  // always uses the GPU from its config file.
  let show = $derived($gpus.length > 0);

  // "" means "use the configured default"; otherwise it's a GPU index.
  let value = $derived(String($gpuSelections[model.id] ?? ""));
  let label = $derived.by(() => {
    const idx = $gpuSelections[model.id];
    return idx === undefined ? "Config" : `GPU ${idx}`;
  });

  // The model's configured GPU (its config.yaml env), shown read-only next to
  // the selector so the default is visible when picking a different one.
  let defaultGpu = $derived(defaultGpuDisplay(model.defaultGpu));

  let h = $derived(size === "sm" ? "h-7 text-xs" : "h-8 text-xs");

  function onChange(v: string): void {
    if (v === "") {
      clearGpuForModel(model.id);
    } else {
      setGpuForModel(model.id, Number(v));
    }
  }
</script>

{#if show}
  <div class="flex items-center gap-1">
    <Select.Root
      type="single"
      value={value}
      onValueChange={onChange}
    >
      <Select.Trigger
        size="sm"
        aria-label={`GPU for ${model.id}`}
        class={`w-fit min-w-[5rem] gap-1 ${h}`}
        title="Load onto GPU (config default: as configured in config.yaml)"
      >
        <span class="truncate">{label}</span>
      </Select.Trigger>
      <Select.Content>
        <Select.Item value="">Use config (default)</Select.Item>
        <Select.Separator />
        {#each $gpus as gpu (gpu.index)}
          <Select.Item value={String(gpu.index)}>{gpu.label}</Select.Item>
        {/each}
      </Select.Content>
    </Select.Root>
    <Input
      type="text"
      readonly
      value={defaultGpu}
      aria-label={`Default GPU for ${model.id}`}
      title="GPU configured for this model in config.yaml"
      class={cn("w-28 select-text text-xs", size === "sm" && "h-7")}
    />
  </div>
{/if}
