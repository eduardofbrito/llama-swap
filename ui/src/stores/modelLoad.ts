import { writable } from "svelte/store";
import { loadModel, unloadSingleModel } from "./api";
import { getGpuForModel } from "./modelGpu";
import type { Model } from "../lib/types";

export const pendingLoads = writable<Record<string, boolean>>({});
const loadControllers = new Map<string, AbortController>();

export async function handleLoadModel(id: string, gpu?: number): Promise<void> {
  if (isPending(id)) return;

  const controller = new AbortController();
  loadControllers.set(id, controller);
  pendingLoads.update((p) => ({ ...p, [id]: true }));
  try {
    await loadModel(id, controller.signal, gpu !== undefined ? String(gpu) : undefined);
  } catch (e) {
    console.error(e);
  } finally {
    loadControllers.delete(id);
    pendingLoads.update((p) => {
      const next = { ...p };
      delete next[id];
      return next;
    });
  }
}

export function cancelLoad(id: string): void {
  loadControllers.get(id)?.abort();
}

export function isPending(id: string): boolean {
  let val = false;
  pendingLoads.subscribe((p) => (val = !!p[id]))();
  return val;
}

export function onToggleLoad(m: Model): void {
  if (m.state === "stopped" && isPending(m.id)) {
    cancelLoad(m.id);
  } else if (m.state === "stopped") {
    // Pass the user's GPU choice (if any) for this model; an undefined value
    // means "use the GPU configured for the model".
    void handleLoadModel(m.id, getGpuForModel(m.id));
  } else if (m.state === "ready") {
    void unloadSingleModel(m.id);
  }
}

export function statusDotColor(m: Model | undefined): string {
  if (!m) return "bg-muted-foreground/40";
  // One colour per state, so the dot alone says which it is without reading
  // the label beside it. The three in-between states are genuinely different
  // things to an operator: one is coming up, one is going down, one is parked
  // and will come back cheaply.
  if (m.state === "ready") return "bg-success"; // green: serving
  if (m.state === "sleeping") return "bg-warning"; // yellow: parked, wakes on demand
  if (m.state === "starting") return "bg-primary"; // blue: coming up
  if (m.state === "stopping") return "bg-destructive"; // red: going down
  return "bg-muted-foreground/40"; // grey: stopped / shutdown / unknown
}
