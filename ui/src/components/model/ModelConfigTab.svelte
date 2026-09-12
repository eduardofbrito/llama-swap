<script lang="ts">
  import * as Card from "$lib/components/ui/card/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Loader2, FileText, RefreshCw, Check, AlertTriangle } from "@lucide/svelte";
  import type { Model } from "../../lib/types";
  import { cn } from "$lib/utils.js";

  interface Props {
    model: Model;
  }
  let { model }: Props = $props();

  let modelId = $derived(model.id);

  let yamlText = $state("");
  let original = $state("");
  let loading = $state(true);
  let loadError = $state("");
  let saving = $state(false);
  let result = $state<{ ok: boolean; msg: string } | null>(null);

  let dirty = $derived(yamlText !== original);

  // Guards against a slow response for an older model overwriting a newer
  // load (rapid model-to-model navigation while the network is slow).
  let loadGeneration = $state(0);

  async function load(): Promise<void> {
    loading = true;
    loadError = "";
    const gen = ++loadGeneration;
    try {
      const response = await fetch(`/api/config/model/${encodeURIComponent(modelId)}`);
      if (gen !== loadGeneration) return; // a newer load started; discard
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const msg = (body as any).error?.message ?? (body as any).error ?? `HTTP ${response.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      const data = (await response.json()) as { yaml: string };
      if (gen !== loadGeneration) return; // a newer load started; discard
      yamlText = data.yaml;
      original = data.yaml;
    } catch (error) {
      if (gen !== loadGeneration) return;
      loadError = error instanceof Error ? error.message : "Failed to load configuration";
    } finally {
      if (gen === loadGeneration) loading = false;
    }
  }

  async function save(): Promise<void> {
    result = null;
    saving = true;
    try {
      const put = await fetch(`/api/config/model/${encodeURIComponent(modelId)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ yaml: yamlText }),
      });
      if (!put.ok) {
        const body = await put.json().catch(() => ({}));
        const msg = (body as any).error?.message ?? (body as any).error ?? `HTTP ${put.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      // Re-read the canonical text the server wrote (formatting/round-trip).
      await load();
      // Apply WITHOUT a full server rebuild: ask the backend to reload only
      // this model's config/process. The server diffs the file against the
      // live config and, when the change is model-only, swaps just this model
      // (every other running model is left untouched). If the diff is not
      // model-only the server falls back to a full reload on its own.
      const reload = await fetch(
        `/api/config/reload?model=${encodeURIComponent(modelId)}`,
        { method: "POST" },
      );
      let msg = reload.ok ? "Saved and configuration reloaded." : "Saved (reload unavailable).";
      if (reload.ok) {
        const body = (await reload.json().catch(() => ({}))) as { scope?: string };
        if (body.scope === "full") {
          msg = "Saved; structural change detected \u2014 the server did a full reload.";
        } else {
          msg = "Saved; only this model was reloaded (every other model kept running).";
        }
      }
      result = { ok: reload.ok, msg };
    } catch (error) {
      result = { ok: false, msg: error instanceof Error ? error.message : "Failed to save" };
    } finally {
      saving = false;
    }
  }

  // Load on mount and again whenever the model id changes. The SPA reuses this
  // component when navigating between /models/:id pages, so a mount-only hook
  // would leave the previous model's YAML on screen for the new one. Reading
  // modelId here is what registers the dependency.
  $effect(() => {
    modelId;
    result = null;
    void load();
  });
</script>

<Card.Root class="gap-0 overflow-hidden py-0">
  <Card.Header class="flex-row items-center gap-2 border-b px-4 py-3">
    <FileText class="size-4 text-muted-foreground" />
    <Card.Title class="text-sm">Configuration</Card.Title>
    <span class="text-muted-foreground text-xs">
      models.{modelId} — edits the server's configuration file
    </span>
    <div class="ml-auto flex items-center gap-2">
      {#if result}
        <span class={cn("flex items-center gap-1 text-xs", result.ok ? "text-success" : "text-destructive")}>
          {#if result.ok}<Check class="size-3.5" />{:else}<AlertTriangle class="size-3.5" />{/if}
          {result.msg}
        </span>
      {/if}
      <Button variant="outline" size="sm" onclick={load} disabled={loading || saving}>
        <RefreshCw class="size-3.5" /> Reload
      </Button>
      <Button size="sm" onclick={save} disabled={saving || loading || !dirty || !yamlText}>
        {#if saving}<Loader2 class="size-3.5 animate-spin" />{:else}<Check class="size-3.5" />{/if}
        Save &amp; Reload
      </Button>
    </div>
  </Card.Header>
  <Card.Content class="p-0">
    {#if loading}
      <div class="flex items-center gap-2 px-4 py-6 text-sm text-muted-foreground">
        <Loader2 class="size-4 animate-spin" /> Loading…
      </div>
    {:else if loadError}
      <div class="px-4 py-6 text-sm text-destructive">{loadError}</div>
    {:else}
      <div class="px-4 py-3">
        <textarea
          class="min-h-112 w-full resize-y rounded-md border bg-muted/20 p-3 font-mono text-xs leading-5"
          aria-label="Model configuration YAML"
          spellcheck="false"
          bind:value={yamlText}
        ></textarea>
        <p class="text-muted-foreground mt-2 text-xs">
          YAML for the <code>models.{modelId}</code> block. Save validates it, writes it to the file and
          triggers the reload (no restart). {dirty ? "Unsaved changes." : "In sync with the file."}
        </p>
      </div>
    {/if}
  </Card.Content>
</Card.Root>
