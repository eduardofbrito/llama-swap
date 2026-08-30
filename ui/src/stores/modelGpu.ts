import { persistentStore } from "./persistent";

// gpuSelections maps a model id -> selected GPU index. An absent entry means
// "use the GPU configured for the model in the config file" (the default).
export const gpuSelections = persistentStore<Record<string, number>>(
  "llamaswap:gpu-selections",
  {},
);

export function getGpuForModel(id: string): number | undefined {
  let val: Record<string, number> = {};
  gpuSelections.subscribe((s) => (val = s))();
  return val[id];
}

export function setGpuForModel(id: string, index: number): void {
  gpuSelections.update((s) => ({ ...s, [id]: index }));
}

export function clearGpuForModel(id: string): void {
  gpuSelections.update((s) => {
    const next = { ...s };
    delete next[id];
    return next;
  });
}

// defaultGpuDisplay formats the model's configured GPU for the read-only
// text box next to the selector: the configured value (index, comma list,
// or any CUDA_VISIBLE_DEVICES value), or "default" when the model's config
// does not pin a GPU.
export function defaultGpuDisplay(gpu: string | undefined): string {
  const trimmed = (gpu ?? "").trim();
  return trimmed === "" ? "default" : trimmed;
}
