<script lang="ts">
  import { onMount } from "svelte";
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

  async function load(): Promise<void> {
    loading = true;
    loadError = "";
    try {
      const response = await fetch(`/api/config/model/${encodeURIComponent(modelId)}`);
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const msg = (body as any).error?.message ?? (body as any).error ?? `HTTP ${response.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      const data = (await response.json()) as { yaml: string };
      yamlText = data.yaml;
      original = data.yaml;
    } catch (error) {
      loadError = error instanceof Error ? error.message : "Falha ao carregar configuração";
    } finally {
      loading = false;
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
      let msg = reload.ok ? "Salvo e configuração recarregada." : "Salvo (recarregar indisponível).";
      if (reload.ok) {
        const body = (await reload.json().catch(() => ({}))) as { scope?: string };
        if (body.scope === "full") {
          msg = "Salvo; mudança estrutural detectada — servidor fez recarga completa.";
        } else {
          msg = "Salvo e apenas este modelo recarregado (demais modelos mantidos).";
        }
      }
      result = { ok: reload.ok, msg };
    } catch (error) {
      result = { ok: false, msg: error instanceof Error ? error.message : "Falha ao salvar" };
    } finally {
      saving = false;
    }
  }

  onMount(() => {
    void load();
  });
</script>

<Card.Root class="gap-0 overflow-hidden py-0">
  <Card.Header class="flex-row items-center gap-2 border-b px-4 py-3">
    <FileText class="size-4 text-muted-foreground" />
    <Card.Title class="text-sm">Configuration</Card.Title>
    <span class="text-muted-foreground text-xs">
      models.{modelId} — edita o arquivo de configuração do servidor
    </span>
    <div class="ml-auto flex items-center gap-2">
      {#if result}
        <span class={cn("flex items-center gap-1 text-xs", result.ok ? "text-success" : "text-destructive")}>
          {#if result.ok}<Check class="size-3.5" />{:else}<AlertTriangle class="size-3.5" />{/if}
          {result.msg}
        </span>
      {/if}
      <Button variant="outline" size="sm" onclick={load} disabled={loading || saving}>
        <RefreshCw class="size-3.5" /> Recarregar
      </Button>
      <Button size="sm" onclick={save} disabled={saving || loading || !dirty || !yamlText}>
        {#if saving}<Loader2 class="size-3.5 animate-spin" />{:else}<Check class="size-3.5" />{/if}
        Salvar & Recarregar
      </Button>
    </div>
  </Card.Header>
  <Card.Content class="p-0">
    {#if loading}
      <div class="flex items-center gap-2 px-4 py-6 text-sm text-muted-foreground">
        <Loader2 class="size-4 animate-spin" /> Carregando…
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
          YAML do bloco <code>models.{modelId}</code>. O botão Salvar valida, escreve no arquivo e
          dispara o reload (sem restart). {dirty ? "Alterações não salvas." : "Em sincronia com o arquivo."}
        </p>
      </div>
    {/if}
  </Card.Content>
</Card.Root>
