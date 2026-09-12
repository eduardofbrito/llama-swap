---
title: Routing capacity and request queues
summary: Configure concurrencyLimit and globalConcurrencyLimit, and understand queued work while a model is loading or busy.
category: guides
tags: [routing, queue, capacity, concurrency, concurrency-limit, max-concurrent-requests, global-concurrency-limit, rate-limit, manual-only, fallback, 503, recent-pool, lru, eviction, vram]
config_keys: [routing, models.*.concurrencyLimit, globalConcurrencyLimit, models.*.manualOnly, routing.scheduler.settings.fifo.recentPoolSize]
updated: 2026-09-12
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
