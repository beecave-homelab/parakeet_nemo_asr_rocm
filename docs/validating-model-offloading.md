# Validating Model Offloading (CLI, WebUI, API)

This guide explains how to validate that GPU model offloading works correctly
across all three runtime surfaces: the **CLI watch mode**, the **Gradio WebUI**,
and the **FastAPI API**. It covers the architecture invariants, the automated
test suites that pin behavior, and a manual runtime protocol for verifying real
VRAM release on a ROCm machine.

**Context**: written after PR #55 (`fix/watch-idle-model-cleanup`), which fixed
model-name forwarding to the watcher, PyTorch allocator release after cache
eviction, and silent failure in idle-cleanup retry logic.

## Architecture invariants

All surfaces share the same offload primitives in
`parakeet_rocm/models/parakeet.py`:

- **`unload_model_to_cpu(model_name)`** — no-load offload. Uses
  `_peek_cached_model` so a cache miss never triggers a model load or device
  promotion. Serialized by the module-level `_cache_lock`. Calls
  `torch.cuda.empty_cache()` after moving the model to CPU.
- **`clear_model_cache()`** — evicts the LRU cache and `_cached_keys`, then
  releases allocator memory (`gc.collect()` + `torch.cuda.empty_cache()`).
  Returns `True` on successful eviction, `False` on failure so idle-cleanup
  callers know to retry instead of marking the cleanup complete.

Any validation of offloading behavior should verify these invariants hold:

1. Idle offload never loads or promotes a model on cache miss.
2. A failed cache eviction is retried on the next idle poll (no silent
   "done" marking).
3. The model name that is actually cached is the one offloaded (relevant when
   the user selects a non-default model).
4. VRAM is actually returned to the driver, not parked in PyTorch's
   caching-allocator free pool.

## Per-surface wiring

| Surface | Idle loop | Model-name forwarding | Failure → retry |
|---|---|---|---|
| CLI (`parakeet_rocm/utils/watch.py`) | Poll loop in `watch_and_transcribe` | `model_name` forwarded from `--model` | `unloaded` set only on success; `cleared` honors `clear_model_cache()` result |
| WebUI (`parakeet_rocm/webui/app.py`, `_start_idle_offload_thread`) | Daemon thread (`webui-idle-offloader`) | Not forwarded — offload uses the default model name (known follow-up) | Retry-on-failure in place |
| API (`parakeet_rocm/api/app.py`, `_start_api_idle_offload_thread`) | Daemon thread (`api-idle-offloader`) | Forwarded via `_active_api_model_name` in `api/routes.py` | Clear stage marks done unconditionally (known gap, see below) |

Both idle timeouts are environment-driven (`parakeet_rocm/utils/constant.py`):

- `IDLE_UNLOAD_TIMEOUT_SEC` (default `300`) — idle duration before the model is
  moved to CPU.
- `IDLE_CLEAR_TIMEOUT_SEC` (default `360`) — idle duration before the model
  cache is evicted entirely and allocator memory released.

### Known gaps

- **API idle loop, clear stage**: `clear_api_model_cache()` returns `None` and
  discards `clear_model_cache()`'s `bool`, so the API idle worker sets
  `cleared = True` even when eviction failed. A failed clear is never retried.
  Fix: return the bool from `clear_api_model_cache()` and honor it in
  `_start_api_idle_offload_thread`.
- **WebUI unload stage**: `_start_idle_offload_thread` calls
  `unload_model_to_cpu()` without a model name, so a user-selected non-default
  model is not moved to CPU at the unload stage. (It is still evicted at the
  later clear stage, which clears the whole cache.)

## Test-level validation

The following suites pin offloading behavior without needing a GPU:

```bash
pdm run pytest -v \
  tests/unit/test_models_parakeet.py \
  tests/unit/test_utils_watch.py \
  tests/unit/test_webui_app.py \
  tests/unit/test_api_app.py
```

What each suite should assert:

- **`test_models_parakeet.py`** — `unload_model_to_cpu` never loads on cache
  miss; `clear_model_cache()` returns `False` when eviction fails and still
  releases the allocator; the cache lock serializes offload vs. clear.
- **`test_utils_watch.py`** — `unload_model_to_cpu` is called with the
  CLI-selected model name; a `False`-returning `clear_model_cache` triggers a
  retry on the next idle poll (does not mark cleanup done).
- **`test_webui_app.py`** — the WebUI idle thread retries after a failed
  unload/clear; `_cleanup_models` performs shutdown cleanup without redundant
  allocator-release calls; fakes return `bool` for `clear_model_cache()`.
- **`test_api_app.py`** — startup wires warmup and idle-offload threads; the
  idle loop calls `unload_active_api_model` / `clear_api_model_cache` past the
  configured timeouts.

> **Coverage note**: there is currently no test asserting retry behavior for
> the API idle loop's clear stage — this matches the known gap above.

## Runtime validation (ROCm machine)

Lower the idle timeouts via environment variables so each phase is observable
in seconds instead of minutes, and watch VRAM in a second terminal:

```bash
watch -n2 rocm-smi --showmeminfo vram
```

### CLI watch mode

```bash
IDLE_UNLOAD_TIMEOUT_SEC=20 IDLE_CLEAR_TIMEOUT_SEC=45 \
  parakeet-rocm --watch ./data/samples --model nvidia/parakeet-tdt-0.6b-v3 --verbose
```

Expected sequence:

1. Drop an audio file into the watched directory → VRAM rises (model on GPU),
   transcription completes.
2. ~20s idle → log line `Idle for >= 20s - offloading model to CPU`; VRAM
   drops partially.
3. ~45s idle → log line `Idle for >= 45s - clearing model cache`; VRAM returns
   near baseline (allocator released, not just model moved).
4. Drop another file → model is re-promoted and transcription succeeds (idle
   flags reset correctly).
5. With a non-default `--model`, the offload log must name that model —
   this validates the model-name forwarding fix from PR #55.

### API + WebUI (docker compose)

```bash
IDLE_UNLOAD_TIMEOUT_SEC=20 IDLE_CLEAR_TIMEOUT_SEC=45 docker compose up --build
```

1. Submit one transcription via the WebUI, then one via
   `POST /v1/audio/transcriptions` — this exercises both idle loops in the
   shared process.
2. Observe VRAM as above; container logs should show
   `[webui] Idle threshold reached - offloading model to CPU` and
   `[webui] Extended idle - clearing model cache`.
3. On `docker compose down` (or Ctrl+C for the standalone WebUI), shutdown
   logs should show `_cleanup_models()` completing without errors.

### Startup warmup

With `API_MODEL_WARMUP_ON_START=true`, startup logs should show the warmup
thread loading `API_MODEL_NAME`, and the first transcription request should
not pay the model-load latency.

## Failure-path validation

The retry-on-failure behavior is best validated with fault injection in tests
rather than at runtime:

- Make `clear_model_cache()` return `False` (e.g. monkeypatch
  `_get_cached_model.cache_clear` to raise) and assert the idle loop retries
  on the next poll and logs the failure.
- The existing suites already do this for the CLI watcher and the WebUI idle
  thread; the API idle loop's clear stage lacks this coverage (see known
  gaps).

## Checklist

- [ ] `pdm run pytest tests/unit/test_models_parakeet.py tests/unit/test_utils_watch.py tests/unit/test_webui_app.py tests/unit/test_api_app.py` passes
- [ ] CLI watch: VRAM rises on activity, drops at unload timeout, returns to baseline at clear timeout
- [ ] CLI watch: offload log names the `--model` value when non-default
- [ ] CLI watch: model re-promotes and transcribes after idle
- [ ] API/WebUI: both idle loops log and release VRAM in the shared process
- [ ] Shutdown cleanup logs clean on SIGINT / `docker compose down`
- [ ] (Follow-up) API idle loop clear stage propagates `clear_model_cache()` failure and retries
- [ ] (Follow-up) WebUI unload stage forwards the selected model name
