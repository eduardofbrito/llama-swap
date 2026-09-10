<script lang="ts">
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Loader2, Plus, Check, AlertTriangle } from "@lucide/svelte";
  import { cn } from "$lib/utils.js";

  // Prefill a minimal template so an editor who has never touched the file
  // still gets a starting point that passes validation.
  const template = (id: string) =>
`proxy: http://127.0.0.1:11435
name: ${id}
# description: what this model does
# unloadAfter: 300
# env:
#   - CUDA_VISIBLE_DEVICES=0
`;

  let open = $state(false);
  let modelId = $state("");
  let yamlText = $state("");
  let saving = $state(false);
  let done = $state(false);
  let error = $state("");
  let reloadOk = $state(false);

  function reset(): void {
    modelId = "";
    yamlText = "";
    error = "";
    done = false;
    reloadOk = false;
  }

  function prefill(e: Event): void {
    e.preventDefault();
    const id = modelId.trim();
    if (id) {
      yamlText = template(id);
    }
  }

  async function add(): Promise<void> {
    error = "";
    const id = modelId.trim();
    if (!id) {
      error = "Informe o id do modelo (a chave em models:).";
      return;
    }
    if (!yamlText.trim()) {
      error = "Cole o YAML do bloco do modelo.";
      return;
    }
    saving = true;
    try {
      const res = await fetch("/api/config/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, yaml: yamlText }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const msg = (body as any).error?.message ?? (body as any).error ?? `HTTP ${res.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      // Apply without restart.
      const reload = await fetch("/api/config/reload", { method: "POST" });
      reloadOk = reload.ok;
      done = true;
      setTimeout(reset, 1500);
    } catch (err) {
      error = err instanceof Error ? err.message : "Falha ao adicionar modelo";
    } finally {
      saving = false;
    }
  }
</script>

<Button variant="outline" size="sm" onclick={() => { reset(); open = true; }}>
  <Plus class="size-3.5" /> Add Model
</Button>

<Dialog.Root bind:open>
  <Dialog.Portal>
    <Dialog.Content>
      <Dialog.Header>
        <Dialog.Title>Add model</Dialog.Title>
        <Dialog.Description>
          Acrescenta um novo bloco em models: no arquivo de configuração e
          recarrega sem restart.
        </Dialog.Description>
      </Dialog.Header>

      <div class="flex flex-col gap-3 py-1">
        <div class="flex flex-col gap-1">
          <label class="text-sm font-medium" for="add-model-id">Model id</label>
          <input
            id="add-model-id"
            class="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm"
            placeholder="ex.: my-model"
            bind:value={modelId}
            onblur={prefill}
            disabled={done}
          />
          <p class="text-muted-foreground text-xs">
            Use o id sem barras para manter simples; o YAML abaixo é o conteúdo
            de <code>models.&lt;id&gt;:</code>.
          </p>
        </div>

        <div class="flex flex-col gap-1">
          <label class="text-sm font-medium" for="add-model-yaml">YAML do modelo</label>
          <textarea
            id="add-model-yaml"
            class="min-h-64 w-full resize-y rounded-md border border-input bg-muted/20 p-3 font-mono text-xs leading-5"
            spellcheck="false"
            bind:value={yamlText}
            disabled={done}
          ></textarea>
        </div>

        {#if error}
          <p class="flex items-start gap-1 text-sm text-destructive">
            <AlertTriangle class="mt-0.5 size-4 shrink-0" /> {error}
          </p>
        {/if}
        {#if done}
          <p class={cn("flex items-center gap-1 text-sm", reloadOk ? "text-success" : "text-warning")}>
            <Check class="size-4 shrink-0" />
            {reloadOk ? "Modelo adicionado e configuração recarregada." : "Modelo adicionado (recarregar indisponível)."}
          </p>
        {/if}
      </div>

      <Dialog.Footer>
        <Button variant="outline" size="sm" onclick={() => (open = false)} disabled={saving}>
          Cancelar
        </Button>
        <Button size="sm" onclick={add} disabled={saving || done}>
          {#if saving}<Loader2 class="size-3.5 animate-spin" />{:else}<Plus class="size-3.5" />{/if}
          Adicionar
        </Button>
      </Dialog.Footer>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
