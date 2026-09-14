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
  if (m.state === "ready") return "bg-success";
  if (m.state === "starting" || m.state === "stopping") return "bg-warning";
  // Sleeping sits between ready and stopped: the process is alive and will
  // wake in seconds, but it is holding no GPU memory and cannot serve yet.
  // A dimmed success colour reads as "almost ready" rather than "off".
  if (m.state === "sleeping") return "bg-success/40";
  return "bg-muted-foreground/40";
}
