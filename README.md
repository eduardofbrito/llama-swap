![llama-swap header image](docs/assets/hero4.webp)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/mostlygeek/llama-swap/total)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mostlygeek/llama-swap/go-ci.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/mostlygeek/llama-swap)

# llama-swap

Run multiple generative AI models on your machine and hot-swap between them on demand. llama-swap works with any OpenAI and Anthropic API compatible server and is used by thousands of people to power their local AI workflows.

Built in Go for performance and simplicity, llama-swap has zero dependencies and is incredibly easy to set up. Get started in minutes - just one binary and one configuration file.

> [!NOTE]
> This is **eduardofbrito's fork** of [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap), tracking upstream `main` with additional GPU management and in-UI config editing features (see [Fork features](#fork-features) below).

## Fork features

Features added in this fork on top of upstream `main`:

- ✅ **Per-model GPU selector** - pick which GPU each model loads onto from the Models list, the model detail page, or any API request
  - The model's configured GPU (`CUDA_VISIBLE_DEVICES` in `env`) is shown read-only next to the selector as `GPU Default <value>`, so the config default is always visible
  - The override travels as the `llama-swap-gpu` query parameter (a `CUDA_VISIBLE_DEVICES` value) on **every** request path — `/upstream/...` loads and normal OpenAI/Anthropic-compatible requests that trigger an on-demand swap
- ✅ **Manual-only models** (`manualOnly: true` per model) - a model that must never be loaded on demand
  - An inference request (POST) for an unloaded manual-only model is answered immediately with a `503` (`manual_load_required`) instead of starting — or queueing — a load, so upstream clients like LiteLLM fail over to their fallback right away
  - A manual model that **is** loaded still serves normally, and it can be started on purpose: the dashboard's load buttons (a GET against the model endpoint) are honored
  - The Models list shows a `manual` badge on such models; combine with a `persistent` group to keep it resident
  - See [routing capacity and request queues](docs/kb/guides/routing/capacity-and-queues.md) for the full behaviour and failure modes
- ✅ **Sleep mode** (`sleepMode: true` per model) - evicting a model puts it to sleep instead of killing it
  - The process stays alive with its weights in host RAM and releases its GPU memory, so the next request **wakes** it instead of paying for a full start (spawn, weight load, `torch.compile`, CUDA graph capture, warmup) — minutes become seconds on a large model
  - Requires a vLLM upstream with sleep mode available: `VLLM_SERVER_DEV_MODE=1` in `env` **and** `--enable-sleep-mode` on the `cmd`. Without both, `/sleep` answers 404 and llama-swap falls back to stopping the model, so nothing breaks — only the speedup is lost
  - Costs roughly the model's weight size in host RAM while it sleeps. Only eviction sleeps: an explicit unload and a `ttl` expiry still stop the process, so `ttl` bounds how long that RAM is held
  - Stacks with `recentPoolSize` into a hierarchy — VRAM for the hottest models, host RAM for the ones pushed out of the pool, stopped after `ttl`
  - A sleeping model shows as `sleeping` on the Models page and stops counting against its GPU on the GPUs page
- ✅ **GPUs page** (`/gpus` menu item, right below Models)
  - Lists every GPU the host exposes with the model(s) currently loaded on each
  - Shows each device's live used/total memory, refreshed while the page is open
  - Flags a GPU **Used externally** when memory is occupied but llama-swap has no model there — the way to spot a card held by a training job, a second llama-swap or a desktop session
  - Load/unload controls per GPU for every model defined in the config (a model running on another GPU is swapped over)
- ✅ **Model config tab** (`Models / <model>`, `Conf` tab)
  - Shows the model's block from the config file as editable YAML. Reading is a pure read: the file on disk is never rewritten, so nothing is reformatted and `-watch-config` is not woken
  - Saving validates the block, writes it back to the file (other models, comments, unrelated keys and the file's permission bits are preserved; the full result is validated through the config load pipeline before the file is touched, and the write is atomic) and triggers a hot reload — no restart
- ✅ **Add model from the UI** - the Models page has an **Add Model** dialog that appends a new `models:` block to the config file (duplicate IDs rejected) and hot-reloads
- ✅ **Docker image over vLLM** - `docker/vllm-swap.Dockerfile` builds this fork's llama-swap (UI embedded) on top of the vLLM runtime image:

  ```bash
  docker build -f docker/vllm-swap.Dockerfile -t vllm-swap .
  ```

  It pins the fork commit through `LLAMA_SWAP_REF` so a build is reproducible, and stamps the binary's version from the checked-out tag (or its short SHA). The file's header documents the config keys worth enabling on a multi-GPU host.
- ✅ **Config API** - machine-accessible endpoints, **off by default**:
  - `GET /api/gpus` - list the GPUs available for model loading (real device indexes) with each device's live used/total memory
  - `GET /api/config/status` - whether config editing is available on this instance (the UI hides the `Conf` tab and **Add Model** when it is not)
  - `GET /api/config/model/{model_id}` - one model's block from the config file as YAML
  - `PUT /api/config/model/{model_id}` - replace one model's block (invalid YAML → 422, file untouched)
  - `POST /api/config/model` - add a new model block (`{"id": ..., "yaml": ...}`)
  - `POST /api/config/reload` - hot reload the config without a restart

> [!WARNING]
> A caller that can write a model block chooses that model's `cmd` and can then start it, which is arbitrary command execution on the host. The editing endpoints are therefore gated twice:
> 1. start llama-swap with `-enable-config-api` (and a single `-config` file); otherwise they return **501**, and
> 2. configure at least one key under `uiApiKeys` (or `apiKeys` as its fallback); otherwise they return **403**. llama-swap's auth middleware is a deliberate pass-through when no keys are set, so this second gate is what keeps the opt-in from publishing an unauthenticated write surface.
>
> Setting `uiApiKeys` alone is enough, and it does **not** gate inference — `apiKeys` alone decides that. So an instance that serves inference without authentication can still lock the dashboard and use the config editor, without breaking the clients already pointed at it.

## Features:

- ✅ Easy to deploy and configure: one binary, one configuration file. no external dependencies
- ✅ On-demand model switching for many local AI servers (llama.cpp + forks, vllm, stable-diffusion.cpp, audio.cpp, ComfyUI, etc.)
  - future proof, upgrade your inference servers at any time.
- ✅ OpenAI API supported endpoints:
  - `v1/completions`
  - `v1/chat/completions`
  - `v1/responses`
  - `v1/embeddings`
  - `v1/models` - list available models
  - `v1/audio/speech` ([#36](https://github.com/mostlygeek/llama-swap/issues/36))
  - `v1/audio/transcriptions` ([docs](https://github.com/mostlygeek/llama-swap/issues/41#issuecomment-2722637867))
  - `v1/audio/voices`
  - `v1/images/generations`
  - `v1/images/edits`
- ✅ Anthropic API supported endpoints:
  - `v1/messages`
  - `v1/messages/count_tokens`
- ✅ llama-server (llama.cpp) supported endpoints
  - `v1/rerank`, `v1/reranking`, `/rerank`
  - `/infill` - for code infilling
  - `/completion` - for completion endpoint
  - `/models` - list available models. same behavior as `v1/models`
  - `/props` - requires `?model={model_id}` query parameter to be provided. The autoload parameter is not supported and will be ignored.
- ✅ SDAPI via [stable-diffusion.cpp's server](https://github.com/leejet/stable-diffusion.cpp/tree/master/examples/server)
  - `/sdapi/v1/txt2img`
  - `/sdapi/v1/img2img`
  - `/sdapi/v1/loras` - requires `model` in request body to fetch the correct loras
- ✅ [audio.cpp](https://github.com/0xShug0/audio.cpp) supported [extra endpoints](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md#post-v1tasksrun)
  - `/audioapi/v1/tasks/run`
- ✅ `/comfyui/` - ComfyUI custom endpoint ([#1001](https://github.com/mostlygeek/llama-swap/issues/1001)) for more reliable swapping
- ✅ llama-swap API
  - `/ui` - web UI
  - `/upstream/:model_id` - direct access to upstream server ([demo](https://github.com/mostlygeek/llama-swap/pull/31))  
  - `/running` - list currently running models ([#61](https://github.com/mostlygeek/llama-swap/issues/61))
  - `POST /api/models/unload` - manually unload all running models ([#58](https://github.com/mostlygeek/llama-swap/issues/58))
  - `POST /api/models/unload/:model_id` - unload a specific model
  - `GET /api/profiles` - list configured profiles and the active selection
  - `PUT /api/profiles/active` - activate a profile or select none
  - `GET /api/gpus` - list the GPUs available for model loading, with live memory per device (fork)
  - `GET /api/config/status` - report whether config editing is enabled (fork)
  - `GET /api/config/model/:model_id` - read one model's block from the config file as YAML (fork)
  - `PUT /api/config/model/:model_id` - replace one model's block in the config file (fork)
  - `POST /api/config/model` - add a new model block to the config file (fork)
  - `POST /api/config/reload` - hot reload the config without a restart (fork)
  - `/logs` - remote log monitoring
    - `GET /logs` returns buffered plain text logs.
      - If `Accept: text/html` is sent, `/logs` redirects to `/ui/`.
    - `GET /logs/stream` keeps the connection open for live log streaming.
      - Stream endpoints send buffered history first by default; add `?no-history` to stream only new lines.
    - `GET /logs/stream/proxy` streams proxy logs only.
    - `GET /logs/stream/upstream` streams upstream process logs only.
    - `GET /logs/stream/{model_id}` streams logs for one model (including IDs with slashes, like `author/model`).
  - `/health` - just returns "OK"
  - `/metrics` - system and GPU metrics for prometheus
- ✅ API Key support - define keys to restrict access to API endpoints
- ✅ Customization
  - Switch model ID routing at runtime with profiles
  - Run concurrent models with a custom DSL swap matrix ([#643](https://github.com/mostlygeek/llama-swap/issues/643))
  - Automatic unloading of models after timeout by setting a `ttl`
  - Docker and Podman support using `cmd` and `cmdStop` together
  - Preload models on startup with `hooks` ([#235](https://github.com/mostlygeek/llama-swap/pull/235))
  - Apply filters to requests to control inference with `stripParams`, `setParams` and `setParamsByID`

### Web UI

llama-swap includes a real time web interface with a playground for testing out all sorts of local models:

<img width="1094" height="667" alt="image" src="https://github.com/user-attachments/assets/a79b3cea-5ee1-45f1-8db9-5f5331690e64" />

View detailed token metrics:

<img width="1090" height="672" alt="image" src="https://github.com/user-attachments/assets/145f4ece-af2f-4a45-a3c1-45ae5d3c7e7f" />

Inspect request and responses:

<img width="1078" height="668" alt="image" src="https://github.com/user-attachments/assets/947cda4f-9aa1-4fa5-a550-5c469968c1d9" />

Manually load and unload models:

<img width="1088" height="659" alt="image" src="https://github.com/user-attachments/assets/b6b850f3-c5b0-4c14-ba90-be2de25b51c7" />

Per-model GPU selector and the configured default (fork):

- In the Models list and on each model's detail page, a GPU selector lets you pin a model to a specific GPU; the model's `config.yaml` GPU default is shown read-only as `GPU Default <value>` next to it
- Any request can override the GPU with the `llama-swap-gpu` query parameter, e.g. `POST /v1/chat/completions?llama-swap-gpu=2`

GPUs page with per-GPU load/unload (fork):

- A `GPUs` menu item (right below Models) lists every GPU the host exposes, the model(s) currently loaded on each, and load/unload controls for every configured model

Model config tab and Add Model (fork):

- `Models / <model>` has a `Conf` tab with the model's block from the config file as editable YAML; saving validates, writes back and hot-reloads without a restart
- The Models page has an `Add Model` dialog that appends a new model to the config file
- Both are hidden unless the server was started with `-enable-config-api` and has `uiApiKeys` (or `apiKeys` as its fallback) configured

Real time log streaming:

<img width="1087" height="668" alt="image" src="https://github.com/user-attachments/assets/9bb0c362-862c-4e68-820c-4c977fc9de4e" />

## Installation

llama-swap can be installed in multiple ways

1. Docker
2. Homebrew (macOS and Linux)
3. MacPorts (macOS)
4. WinGet
5. From release binaries
6. From source

### Docker Install ([download images](https://github.com/mostlygeek/llama-swap/pkgs/container/llama-swap))

Two types of container images are built nightly for llama-swap:

1. A unified container with llama-server, ik-llama-server, stable-diffusion.cpp, whisper.cpp, audio.cpp and llama-swap all built from source. Available for CUDA 12, CUDA 13 and Vulkan. This one is recommended for use.
2. A legacy image, which is llama.cpp's own `llama-server` container with llama-swap copied in. It carries only what that base image ships, so no image generation, speech or ik-llama-server.

#### Unified container (Recommended)

There are three unified images. Pick the one that matches your GPU:

| tag | platforms | GPUs |
| --- | --- | --- |
| `unified-cuda13` | amd64, arm64 | NVIDIA Ampere through Blackwell: A100, RTX 30xx/40xx/50xx, H100, RTX PRO, and GB10 on DGX Spark. Built with CUDA 13. |
| `unified-cuda` | amd64 | NVIDIA Pascal through Ada: P40, P100, GTX 10xx, RTX 20xx/30xx/40xx. Built with CUDA 12, for cards CUDA 13 dropped. |
| `unified-vulkan` | amd64 | AMD and other Vulkan capable GPUs. |

`unified-cuda13` is a multi-arch tag, so `docker pull` picks the right image for
the host — including the aarch64 one a DGX Spark needs.

```shell
$ docker pull ghcr.io/mostlygeek/llama-swap:unified-cuda13

# run with a custom configuration and models directory
$ docker run -it --rm --runtime nvidia -p 9292:8080 \
 -v /path/to/models:/models \
 -v /path/to/custom/config.yaml:/etc/llama-swap/config/config.yaml \
 ghcr.io/mostlygeek/llama-swap:unified-cuda13
```

##### Configuring startup with environment variables

The unified images can be configured with `LLAMA_SWAP_*` environment variables
instead of command line flags, which is usually easier in a compose file or a
Kubernetes manifest. Each one maps to a llama-swap flag:

| variable | flag | default in the image |
| --- | --- | --- |
| `LLAMA_SWAP_CONFIG` | `-config` | `/etc/llama-swap/config/config.yaml` |
| `LLAMA_SWAP_CONFIG_DIR` | `-config-dir` | — |
| `LLAMA_SWAP_LISTEN` | `-listen` | `0.0.0.0:8080` |
| `LLAMA_SWAP_TLS_CERT_FILE` | `-tls-cert-file` | — |
| `LLAMA_SWAP_TLS_KEY_FILE` | `-tls-key-file` | — |
| `LLAMA_SWAP_LISTEN_TAILCAT` | `-listen-tailcat` | — |
| `LLAMA_SWAP_WATCH_CONFIG` | `-watch-config` | `true` |

```shell
# configure startup with environment variables instead of flags
$ docker run -it --rm --runtime nvidia -p 9292:8080 \
 -v /path/to/models:/models \
 -e LLAMA_SWAP_CONFIG=/models/llama-swap.yaml \
 -e LLAMA_SWAP_LISTEN=0.0.0.0:8080 \
 -e LLAMA_SWAP_WATCH_CONFIG=false \
 ghcr.io/mostlygeek/llama-swap:unified-cuda13
```

```yaml
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:unified-cuda13
    ports:
      - "9292:8080"
    volumes:
      - /path/to/models:/models
    environment:
      LLAMA_SWAP_CONFIG: /models/llama-swap.yaml
      LLAMA_SWAP_LISTEN: 0.0.0.0:8080
      LLAMA_SWAP_WATCH_CONFIG: "false"
```

**Keep `LLAMA_SWAP_LISTEN` on `0.0.0.0` whenever you publish a port.** A
container that binds its own loopback is not reachable through `-p`, so
`localhost:8080` there gives connection refused. Bind loopback only when the
container shares the host's network, where it usefully limits llama-swap to the
host itself:

```shell
# reachable from the host only, not from the network
$ docker run -it --rm --runtime nvidia --network host \
 -v /path/to/models:/models \
 -e LLAMA_SWAP_LISTEN=localhost:8080 \
 ghcr.io/mostlygeek/llama-swap:unified-cuda13
```

An unset or empty variable contributes nothing and leaves llama-swap's own
default. Booleans accept `true`/`false`, `1`/`0`, `yes`/`no` or `on`/`off` in any
case; anything else stops the container rather than being read as "off".
`-version` is deliberately not mapped — use `docker run <image> -version`.

Passing flags to the container still works and behaves exactly as it always has:
arguments replace every default rather than adding to the variables above.

```shell
# unchanged: runs `llama-swap -config /models/my.yaml`, nothing else added
$ docker run ghcr.io/mostlygeek/llama-swap:unified-cuda13 -config /models/my.yaml
```

See [docker/unified/README.md](docker/unified/README.md#configuring-with-environment-variables)
for the details.

#### Legacy container

This image is llama.cpp's own `llama-server` container
([ghcr.io/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp))
with the llama-swap binary copied into it. It tracks llama.cpp's nightly server
images closely, which is its only real advantage.

**Prefer the unified container.** The legacy image inherits whatever the
`llama-server` base ships and nothing else, so it has no stable-diffusion.cpp,
whisper.cpp, audio.cpp or ik-llama-server, and no `LLAMA_SWAP_*` environment
variable support. Use it only if staying on llama.cpp's exact server image
matters to you.

```shell
$ docker pull ghcr.io/mostlygeek/llama-swap:cuda

# run with a custom configuration and models directory
$ docker run -it --rm --runtime nvidia -p 9292:8080 \
 -v /path/to/models:/models \
 -v /path/to/custom/config.yaml:/app/config.yaml \
 ghcr.io/mostlygeek/llama-swap:cuda
```

<details>
<summary>
more examples
</summary>

```shell
# pull latest images per platform
docker pull ghcr.io/mostlygeek/llama-swap:cpu
docker pull ghcr.io/mostlygeek/llama-swap:cuda
docker pull ghcr.io/mostlygeek/llama-swap:vulkan
docker pull ghcr.io/mostlygeek/llama-swap:intel
docker pull ghcr.io/mostlygeek/llama-swap:musa

# tagged llama-swap, platform and llama-server version images
docker pull ghcr.io/mostlygeek/llama-swap:v166-cuda-b6795

# non-root cuda
docker pull ghcr.io/mostlygeek/llama-swap:cuda-non-root

```

</details>

### Homebrew Install (macOS/Linux)

```shell
brew tap mostlygeek/llama-swap
brew install llama-swap
llama-swap --config path/to/config.yaml --listen localhost:8080
```

### MacPorts (macOS)

> [!NOTE]
> Maintained by MacPorts community - [llama-swap port](https://ports.macports.org/port/llama-swap). It is not an official part of llama-swap.

```shell
sudo port install llama-swap
llama-swap --config path/to/config.yaml --listen localhost:8080
```

### WinGet Install (Windows)

> [!NOTE]
> WinGet is maintained by community contributor [Dvd-Znf](https://github.com/Dvd-Znf) ([#327](https://github.com/mostlygeek/llama-swap/issues/327)). It is not an official part of llama-swap.

```shell
# install
C:\> winget install llama-swap

# upgrade
C:\> winget upgrade llama-swap
```

### Pre-built Binaries

Binaries are available on the [release](https://github.com/mostlygeek/llama-swap/releases) page for Linux, Mac, Windows and FreeBSD.

### Building from source

1. Building requires Go and Node.js (for UI).
1. `git clone https://github.com/eduardofbrito/llama-swap.git` (fork) or `https://github.com/mostlygeek/llama-swap.git` (upstream)
1. To build a specific commit reproducibly (as this fork's Dockerfile does), `git checkout <sha>` first
1. `make clean all`
1. look in the `build/` subdirectory for the llama-swap binary

## Configuration

```yaml
# minimum viable config.yaml

models:
  model1:
    cmd: llama-server --port ${PORT} --model /path/to/model.gguf
```

That's all you need to get started:

1. `models` - holds all model configurations
2. `model1` - the ID used in API calls
3. `cmd` - the command to run to start the server.
4. `${PORT}` - an automatically assigned port number

Almost all configuration settings are optional and can be added one step at a time:

- Advanced features
  - `matrix` to run concurrent models with a custom swap logic DSL
  - `hooks` to run things on startup
  - `macros` reusable snippets
- Model customization
  - `ttl` to automatically unload models
  - `unloadTimeout` to tune graceful unloads (manual, API and `ttl` expiry)
  - `aliases` to use familiar model names (e.g., "gpt-4o-mini")
  - `env` to pass custom environment variables to inference servers
  - `cmdStop` gracefully stop Docker/Podman containers
  - `useModelName` to override model names sent to upstream servers
  - `${PORT}` automatic port variables for dynamic port assignment
  - `filters` rewrite parts of requests before sending to the upstream server

See the [configuration overview](docs/kb/guides/configuration/configuration-overview.md)
to get started, and the [knowledge base](docs/kb/) for focused guides on the features
people ask about most.

You can also edit the config from the web UI (fork): the `Conf` tab of each model
(`Models / <model>`) edits that model's block in place, and the `Add Model` dialog on
the Models page appends new models. Both validate, write back and hot-reload without
a restart. Editing is off unless you start llama-swap with `-enable-config-api` and
configure `uiApiKeys` (or `apiKeys` as its fallback); until then the UI hides both
controls.

You can also just ask. The **Help** page (in the sidebar) is an agent that calls
llama-swap's own documentation tools and answers questions about your
configuration using the real text of `config.example.yaml` and the knowledge
base, running entirely on a local model. Pick a tool-capable model on the
**Help** page and ask away — see
[Writing the cmd for a model](docs/kb/guides/model-runtime/writing-cmd.md) if the model
answers without calling anything (older llama-server builds need `--jinja` added explicitly).

Those same tools are served as an MCP endpoint at `/api/mcp`, so any MCP client
can ask about your configuration too. See
[Connecting an MCP client](docs/kb/guides/api-integration/mcp-endpoint.md).

## How does llama-swap work?

When a request is made to an OpenAI compatible endpoint, llama-swap will extract the `model` value and load the appropriate server configuration to serve it. If the wrong upstream server is running, it will be replaced with the correct one. This is where the "swap" part comes in. The upstream server is automatically swapped to handle the request correctly.

In the most basic configuration llama-swap handles one model at a time. For more advanced use cases, using a `matrix` allows multiple models to be loaded at the same time. You have complete control over how your system resources are used.

## Reverse Proxy Configuration (nginx)

If you deploy llama-swap behind nginx, disable response buffering for streaming endpoints. By default, nginx buffers responses which breaks Server‑Sent Events (SSE) and streaming chat completion. ([#236](https://github.com/mostlygeek/llama-swap/issues/236))

Recommended nginx configuration snippets:

```nginx
# SSE for UI events/logs
location /api/events {
    proxy_pass http://your-llama-swap-backend;
    proxy_buffering off;
    proxy_cache off;
}

# Streaming chat completions (stream=true)
location /v1/chat/completions {
    proxy_pass http://your-llama-swap-backend;
    proxy_buffering off;
    proxy_cache off;
}
```

As a safeguard, llama-swap also sets `X-Accel-Buffering: no` on SSE responses. However, explicitly disabling `proxy_buffering` at your reverse proxy is still recommended for reliable streaming behavior.

## Monitoring Logs on the CLI

```sh
# sends up to the last 10KB of logs
$ curl http://host/logs

# streams combined logs
curl -Ns http://host/logs/stream

# stream llama-swap's proxy status logs
curl -Ns http://host/logs/stream/proxy

# stream logs from upstream processes that llama-swap loads
curl -Ns http://host/logs/stream/upstream

# stream logs only from a specific model
curl -Ns http://host/logs/stream/{model_id}

# stream and filter logs with linux pipes
curl -Ns http://host/logs/stream | grep 'eval time'

# appending ?no-history will disable sending buffered history first
curl -Ns 'http://host/logs/stream?no-history'
```

## Do I need to use llama.cpp's server (llama-server)?

Any OpenAI compatible server would work. llama-swap was originally designed for llama-server and it is the best supported.

For Python based inference servers like vllm or tabbyAPI it is recommended to run them via podman or docker. This provides clean environment isolation as well as responding correctly to `SIGTERM` signals for proper shutdown.
