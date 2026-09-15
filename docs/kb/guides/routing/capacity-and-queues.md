---
title: Routing capacity and request queues
summary: Configure concurrencyLimit and globalConcurrencyLimit, and understand queued work while a model is loading or busy.
category: guides
tags: [routing, queue, capacity, concurrency, concurrency-limit, max-concurrent-requests, global-concurrency-limit, rate-limit, manual-only, fallback, 503, recent-pool, lru, eviction, vram, gpu-memory, oom, insufficient-vram, sleep-mode, vllm, swap-latency, cold-start, wake]
config_keys: [routing, models.*.concurrencyLimit, globalConcurrencyLimit, models.*.manualOnly, models.*.vramMB, models.*.sleepMode, routing.scheduler.settings.fifo.recentPoolSize, routing.scheduler.settings.fifo.vramCheck, routing.scheduler.settings.fifo.vramMarginPct, routing.scheduler.settings.fifo.evictBeyondPoolOnPressure, routing.scheduler.settings.fifo.maxSleepingPerGPU]
updated: 2026-09-15
---

# Routing capacity and request queues

There is no `models.*.maxConcurrentRequests` setting. The real per-model key
is `models.*.concurrencyLimit`, which limits active parallel requests.

The router serializes model swaps and queues requests until the selected model
is ready. Avoid using many simultaneous client retries as a capacity control:
they create more queued work. Choose a routing policy that fits the models that
may coexist, then set client timeouts high enough for the queue and load time.

Inspect the Activity view and logs when latency grows. A queue that never drains
usually means the command, proxy, or health check is wrong; see
`guides/model-runtime/troubleshooting-model-wont-load`.

## Global concurrency limit

`globalConcurrencyLimit` is a top-level setting, separate from
`models.*.concurrencyLimit`. It caps the number of inference requests served
at once across every model combined, using a single shared semaphore:

```yaml
globalConcurrencyLimit: 8
```

The default is `0`, meaning no limit — the semaphore is not even added to the
request chain, so a default config pays no cost for the feature. Once the
limit is reached, further requests are rejected immediately with an HTTP 429
response rather than queued; there is no wait for a slot to free up. Clients
should treat a 429 as a signal to back off and retry, the same way they handle
a per-model concurrency limit rejection.

Use this to protect shared hardware (CPU, disk, network) from being
overwhelmed by traffic spread across many different models, which a per-model
`concurrencyLimit` cannot do since it only counts requests to one model at a
time.

## Never load a model on demand

`models.*.manualOnly` takes a model out of on-demand loading entirely. An
inference request for it, while it is not loaded, is answered immediately with
HTTP 503 and the error code `manual_load_required` — the router does not start
a swap and does not queue the request:

```yaml
models:
  big-70b:
    cmd: llama-server --model /models/big-70b.gguf --port ${PORT}
    manualOnly: true
```

This exists for gateways that front several backends. LiteLLM, OpenWebUI and
similar clients treat a fast 503 as "this backend is out", so they fail over to
their fallback right away. Without `manualOnly` they instead sit through a cold
start that can take minutes, and the user sees a hung request rather than an
answer from the fallback model.

What still works:

- a `manualOnly` model that **is** loaded serves inference normally — the flag
  only governs who may *start* it;
- the dashboard's load button starts it, because that is a `GET` against the
  model endpoint rather than an inference call;
- pair it with a `persistent` group when you want it to stay resident once
  started, instead of being evicted by the next swap.

The failure mode to watch for: with `manualOnly: true` and no one loading the
model, **every** inference request 503s forever. If clients report a model that
is permanently unavailable, check the Models page — a `manual` badge and a
`stopped` state mean it is waiting to be loaded, not broken.

## Keep recently used models loaded

By default a swap unloads every model the router's swapper nominated, so
loading a second model flushes the first even when the GPU had room for both.
`recentPoolSize` turns that into an LRU working set:

```yaml
routing:
  scheduler:
    use: fifo
    settings:
      fifo:
        recentPoolSize: 3
```

With a pool of 3, the three most recently used models stay loaded; requesting a
fourth unloads only the least recently used one. `0` and `1` both disable it
and leave every eviction decision to the swapper.

Two properties worth relying on:

- **The pool only ever holds an eviction back — it never invents one.** What it
  keeps loaded is always a subset of what the swapper asked to unload, so a
  model the swapper deliberately keeps resident (a `persistent` group member) is
  never affected, and nothing is unloaded that would not have been anyway.
- **A model that is not an eviction candidate does not consume a pool slot.** A
  persistent member answering requests between two swaps would otherwise sit at
  the top of the recency list and starve the pool of the models it exists to
  protect.

The failure mode: `recentPoolSize` is a promise about your hardware, not a
request. What the pool keeps loaded is precisely what the swapper wanted
unloaded to free capacity, so setting it to 3 on a GPU that fits two models
means the third load fails — or the driver starts swapping — instead of the
swapper cleanly making room. Set it to the number of models that genuinely fit
in VRAM at once, and see
`guides/model-runtime/troubleshooting-model-wont-load` when a load starts
failing after you raise it.

## Make eviction cheap instead of avoiding it

`recentPoolSize` avoids evictions by keeping models in VRAM. `sleepMode` makes
the evictions you cannot avoid cheap:

```yaml
models:
  qwen-27b:
    env: ["VLLM_SERVER_DEV_MODE=1"]
    cmd: vllm serve /models/qwen-27b-fp8 --port ${PORT} --enable-sleep-mode
    sleepMode: true
```

A model evicted to make room for another is asked to release its GPU memory and
stay alive, instead of being killed. The next request **wakes** it — the weights
are copied back from host RAM — rather than paying for a full start.

That matters because loading weights is the small half of a cold start. A vLLM
start is process spawn, weight load, `torch.compile`, CUDA graph capture and
warmup; waking skips all of it but the copy, because the process never died.

**This requires vLLM's sleep mode, which is not on by default.** Both parts are
needed: `VLLM_SERVER_DEV_MODE=1` in `env` (the endpoints are not mounted
without it) and `--enable-sleep-mode` on the command. Miss either and `/sleep`
answers 404; llama-swap logs that and stops the model instead, so the model
keeps working and only the speedup is lost.

### What it costs

Host RAM, for as long as the model sleeps — roughly the model's weight size.
Two sleeping 30 GB models are 60 GB of system memory. Budget for it.

### What still stops a model outright

Sleeping is what *eviction* means for the model, and nothing else:

- an explicit unload (the dashboard's unload button, `/unload`) stops it
- a `ttl` expiry stops it

`ttl` is therefore how you bound the RAM: sleeping keeps swaps fast while a
model is in rotation, and `ttl` reclaims it once nobody has asked for it in a
while.

The three settings stack into a memory hierarchy — `recentPoolSize` holds the
hottest models in VRAM, `sleepMode` catches the ones pushed out of the pool, and
`ttl` eventually gives the RAM back:

```yaml
models:
  a: { sleepMode: true, ttl: 3600, env: ["VLLM_SERVER_DEV_MODE=1"], cmd: "..." }
  b: { sleepMode: true, ttl: 3600, env: ["VLLM_SERVER_DEV_MODE=1"], cmd: "..." }

routing:
  scheduler:
    use: fifo
    settings:
      fifo:
        recentPoolSize: 1
```

A sleeping model shows as `sleeping` on the Models page and in `/running`, and
the GPUs page stops counting it against its device — it is holding no VRAM, so
a card whose only model is asleep reads as idle.

### It does not free the whole card, and it does not move

Two limits to plan around, both consequences of the process staying alive:

- **A sleeping model still holds a few GB on its GPU** — the CUDA context and
  whatever its runtime allocated outside the sleep-mode allocator. vLLM reports
  it: `Sleep mode freed 65.37 GiB memory, 4.29 GiB memory is still in use`.
  Budget 4–6 GB per sleeping model on that card, and subtract it from what the
  resident model may claim. A model sized with `--gpu-memory-utilization 0.92`
  will fail to start on a card where something is asleep, because vLLM computes
  that fraction against the card's **total**, not what is free.
- **A sleeping model cannot change GPU.** A live CUDA process is bound to the
  device it started on, so waking copies the weights back onto that same card.
  Group `gpus` placement and an explicit `llama-swap-gpu` override are both
  ignored on a wake (the override logs a warning), because honouring either
  would mean restarting the process — the cold start sleeping exists to avoid.
  Placement is decided once, on the load that starts the process.

So `gpus` and `sleepMode` combine, but not into "wake onto whichever card is
free". What you get is rotation *within* each card: the group spreads models
across devices at cold-start time, and sleeping makes re-activation cheap on
the card each model was given. Size the group accordingly — one or two sleepers
per card, not five.

`maxSleepingPerGPU` turns that sizing into a rule instead of arithmetic you
have to redo every time the group grows:

```yaml
routing:
  scheduler:
    use: fifo
    settings:
      fifo:
        maxSleepingPerGPU: 1
```

Once a device is at the cap, parking another model there stops the one that has
been asleep longest — which is also the least recently used, since a model
serves right up to the moment it is evicted. The default, `0`, is no cap and
the behaviour sleeping had before this setting existed.

Use it when the group has more members than devices plus one. Below that a card
never collects a second sleeper anyway, and the cap costs nothing but also does
nothing.

Two further interactions worth knowing:

- **A sleeping model is never evicted again.** It is already holding no GPU
  memory, so the scheduler does not consider it when deciding what to unload,
  and it keeps sleeping while other models come and go.
- **`manualOnly` still refuses to auto-load a sleeping model.** The model is not
  loaded, so an inference request gets the usual 503; start it from the
  dashboard as you would any manual model.

## Refuse loads the GPU has no room for

`recentPoolSize` assumes llama-swap owns the GPU: it reasons about the models
it started and nothing else. That assumption breaks when a training job, a
second llama-swap, or a desktop session holds VRAM — "every model I evicted is
stopped" stops meaning "the memory is free", and the load fails deep inside the
runtime minutes later.

`vramCheck` turns that into an immediate answer:

```yaml
routing:
  scheduler:
    use: fifo
    settings:
      fifo:
        vramCheck: true
        vramMarginPct: 5

models:
  big-70b:
    cmd: llama-server --model /models/big-70b.gguf --port ${PORT}
    env:
      - CUDA_VISIBLE_DEVICES=0
    vramMB: 41000
```

Before starting a swap the router compares the target's requirement (plus
`vramMarginPct` headroom) against what is actually free on the GPU it would
load onto, counting the memory this swap's evictions are about to give back. A
model that does not fit gets HTTP 503 with error code `insufficient_vram`, so a
gateway fails over at once instead of waiting out a doomed load.

### Where the requirement comes from

`models.*.vramMB` is the declared value and always wins. When it is absent,
llama-swap uses what it measured on that model's last successful load — the
rise in the GPU's used memory across the load, persisted in the store so it
survives a restart.

Declare the number for models whose load must never be attempted on a GPU that
cannot hold them. Leave it out for the rest and let llama-swap learn it; the
first load of an unmeasured model is always admitted, because refusing on
ignorance would be worse than not checking at all.

That rule covers the models being **evicted** too, not just the target: a swap
that frees room by stopping a model with no known requirement leaves the
device's post-swap free memory unknowable, so the check admits rather than
guesses. The practical consequence is that `vramCheck` only really protects a
GPU once every model that lands on it has a number — declare `vramMB` on the
big ones instead of relying on measurement alone.

Measurement is also easier to miss than it looks. It is taken as the rise in
the GPU's used memory across a load, read from the performance monitor's
sampled ring — so a swap that finishes inside one sampling period (see
`performance.every`) sees the same sample at both ends, reads a zero rise, and
records nothing. `models.*.sleepMode` makes this much more likely, because
waking is seconds where a cold start was minutes. Wakes are deliberately not
measured at all: restoring weights is a different quantity from a cold load,
and a wrong requirement is worse than none. Run with `logLevel: debug` to see
`no VRAM measurement for <model>` and why.

### When there is no room

By default the request is refused and the recent-model pool keeps what it
promised. `evictBeyondPoolOnPressure: true` reverses that trade: the pool's
held-back evictions are given up so the load can proceed. Even then a request
is refused when evicting everything still does not free enough — that is the
signal that the memory belongs to something outside llama-swap.

### What it deliberately does not do

Every case where the guard cannot answer honestly admits the model, so turning
`vramCheck` on cannot break a working setup:

- a model with no declared and no measured requirement;
- a model pinned to a device the host does not report, such as a multi-GPU
  `CUDA_VISIBLE_DEVICES=0,1`;
- a host that reports no GPU memory at all (no performance monitor, CPU-only);
- a model that is already loaded — it holds its memory already, and re-checking
  would double-count it.

A model that pins no GPU is checked against the device with the most free
memory, since that is where it has the best chance of fitting.

The failure mode to watch for: a model whose measured value came from a
smaller context or `-ngl` than it now runs with will be admitted onto a GPU
that cannot hold it. Declaring `vramMB` is what makes the check exact.
