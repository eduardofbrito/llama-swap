---
title: API keys, access control, and keeping secrets out of your config
summary: apiKeys authentication has no per-key rate limits or user permissions; use env macros for secrets.
category: guides
tags: [security, api-keys, auth, secrets, env, rate-limit, users, permissions, access-control]
config_keys: [apiKeys, uiApiKeys, macros, peers.*.apiKey, models.*.env]
updated: 2026-09-14
---

# API keys and keeping secrets out of your config

## Requiring a key

By default llama-swap is unauthenticated. Add `apiKeys` and every request needs
one:

```yaml
apiKeys:
  - "sk-hunter2"
```

Clients may present it as `Authorization: Bearer <key>`, `x-api-key: <key>`, or
HTTP Basic. `apiKeys` gates the inference endpoints (model dispatch,
`/upstream`, `/comfyui`) and, when `uiApiKeys` is unset, the web UI and
everything under `/api/` too — see [Separate keys for the UI](#separate-keys-for-the-ui)
below if you want those to require a different key.

Generate a real one:

```console
$ printf "sk-%s\n" "$(head -c 48 /dev/urandom | base64)"
```

Multiple keys are allowed, which is how you rotate without downtime: add the
new key, move clients over, remove the old one.

All keys are equivalent within their set. llama-swap has no per-key rate
limiting, user accounts, roles, or per-key permissions. Put a reverse proxy or
API gateway in front of llama-swap when you need those controls.

**`apiKeys` is not a substitute for a firewall.** llama-swap starts processes
on your machine. Do not expose it to the internet on the strength of a bearer
token alone.

## Keep the keys out of the file

Use env macros so the config itself is safe to commit:

```yaml
apiKeys:
  - "${env.LLAMA_SWAP_KEY}"
  - "${env.LLAMA_SWAP_KEY_ROTATE}"
```

`${env.VAR}` is substituted before anything else. **If the variable is not set,
config loading fails with an error** — which is what you want. A typo becomes a
startup failure rather than an instance running with an empty key list.

The same applies anywhere a secret appears:

```yaml
peers:
  openrouter:
    proxy: https://openrouter.ai/api
    apiKey: ${env.OPENROUTER_API_KEY}
    models: [z-ai/glm-4.7, moonshotai/kimi-k2-0905]

models:
  hf-model:
    env:
      - "HF_TOKEN=${env.HF_TOKEN}"
    cmd: llama-server --port ${PORT} -hf some/repo
```

## Separate keys for the UI

By default one key set (`apiKeys`) gates everything: inference and the
dashboard/control-plane alike. Add `uiApiKeys` to split them:

```yaml
apiKeys:
  - "${env.LLAMA_SWAP_INFERENCE_KEY}" # for LiteLLM, apps, external clients

uiApiKeys:
  - "${env.LLAMA_SWAP_UI_KEY}" # for operators logging into the dashboard
```

With both set:

- `uiApiKeys` gates the dashboard, `/api/*` (including the config-editing
  endpoints), `/logs`, `/metrics`, `/unload`, `/running`.
- `apiKeys` gates inference only (model dispatch, `/upstream`, `/comfyui`).
- A `uiApiKeys` key **also** works for inference — a dashboard login can drive
  the Playground's real chat/completions calls without a second key.
- An `apiKeys` (inference-only) key does **not** open the UI once `uiApiKeys`
  is set. This is the point: hand an external client an inference-only key
  that cannot unload models or edit the config file, while operators keep a
  separate, stronger key.

Leave `uiApiKeys` empty (the default) and it falls back to `apiKeys` — an
existing config that only ever set `apiKeys` keeps gating the UI exactly as it
did before `uiApiKeys` existed.

### Lock the dashboard, leave inference open

The other single-key case is worth calling out, because it is the one that
makes `-enable-config-api` usable on an instance that deliberately serves
inference without authentication:

```yaml
uiApiKeys:
  - "${env.LLAMA_SWAP_UI_KEY}"
# apiKeys stays unset
```

**`apiKeys` alone decides whether inference is gated.** Setting `uiApiKeys`
never turns that gate on by itself, so every client already pointed at this
server keeps working unauthenticated while the dashboard, `/api/*` and the
config editor start requiring a key.

The two rules are not symmetric, and that is deliberate:

| `apiKeys` | `uiApiKeys` | Inference | UI / `/api/*` |
|---|---|---|---|
| unset | unset | open | open |
| set | unset | `apiKeys` | `apiKeys` |
| unset | set | **open** | `uiApiKeys` |
| set | set | both keys | `uiApiKeys` |

`-enable-config-api` follows `uiApiKeys` (falling back to `apiKeys`) for its
own key requirement, since the config-editing endpoints live under `/api/`.

## How peer keys are used

`peers.*.apiKey` is injected into outgoing requests to that peer, as **both**
`Authorization: Bearer <key>` and `x-api-key: <key>`. Leave it blank and no key
is added. It accepts a macro, so use `${env.*}`.

This is the key llama-swap presents *to* the peer. It is unrelated to the
`apiKeys` clients present *to* llama-swap.

## What ends up in your config file

Worth being aware of, because config files get pasted into issues and chats:

- `apiKeys` / `uiApiKeys` — literal keys, unless you used env macros
- `peers.*.apiKey` — literal peer keys, same
- `models.*.env` — literal `NAME=value` pairs
- `models.*.cmd` / `cmdStop` — full command lines, which often carry
  `--api-key` flags and absolute paths
- `macros` — **after** env substitution has already run

That last one is the non-obvious one: a macro like
`llama: "llama-server --api-key ${env.KEY}"` holds the resolved secret once the
config is loaded.

Before sharing a config, strip those. Redact rather than delete so the shape of
the problem survives.

## Related

- `reference/config/apiKeys`, `reference/config/uiApiKeys`, `reference/config/peers`
- `guides/configuration/macros` — how env macros resolve
