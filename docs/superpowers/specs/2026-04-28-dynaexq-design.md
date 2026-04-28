# DynaExq on SGLang — Design Spec

**Date:** 2026-04-28
**Branch:** `feat/dynaexq` (off `refactor/expert-precision-assignment-reorg`)
**Owner:** huanchen
**Reference paper:** *Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference* (Chu et al., arXiv:2511.15015v3)

## 1. Problem statement

The existing SGLang heter-MoE path (`HeterFusedMoE`) uses a **static** partial-BF16 set: at startup we decide which experts get BF16 weights resident on GPU, and the rest stay INT4-only. This wastes HBM on cold experts under workload shift and over-compresses experts that become hot.

DynaExq (the paper) makes that set **dynamic**: BF16 residency follows long-horizon hotness, under a hard HBM cap, with non-blocking async swaps. We want DynaExq's runtime mechanism on top of our existing kernel and dispatch policies.

This spec scopes the implementation as **modifying the BF16 weight call path** in `HeterFusedMoE` to use a compressed `[n_hi, ...]` pool with a mutable expert→slot remap, and adding a new `dynaexq/` package that owns the host store, budget, scheduler, and migration pipeline.

## 2. Goals & non-goals

### Goals

- BF16 resident set is a moving target driven by router-observed long-horizon hotness.
- HBM footprint of BF16 weights is bounded by an explicit, configurable cap.
- Resident-set updates are non-blocking: forward path never waits on H2D copies.
- Reuse the existing dispatch-policy ABC (`HeterDispatchPolicy`) — no policy rewrites.
- CUDA-graph-safe: no shape changes, no allocator pressure during steady-state.
- Demonstrate the dynamism via a workload-shift trace and an elastic-budget trace.
- Reproduce the paper's Q1 (quality) and Q2 (performance) eval structure on our hardware.

### Non-goals

- Implement ExpertFlow comparison (cited as paper baseline; integration is its own engineering effort and is flagged as stretch).
- Replace INT4 fallback. INT4 weights stay always-resident — no eviction-to-host of INT4.
- Multi-GPU / EP / TP coordination of dynaexq state. Single-GPU only in v1.
- Online quality probes (LM head loss, etc). Hotness is pure router-trace EMA.

## 3. Architecture

### 3.1 HBM accounting

```
M_total = W_nonexpert + F_runtime + I_int4_experts(fixed) + B_bf16_pool + K_kv_cache
```

INT4 weights for **all** experts are always-resident. The BF16 pool `B` is sized at startup as `n_hi_max × bf16_bytes_per_expert × num_layers`; this is the ceiling. The KV cache `K` is sized to consume the remainder.

**Conceptual model:** `B` and `K` would compete for leftover HBM, with `B` shrinking under KV pressure to give bytes back to KV. **v1 reality** depends on whether SGLang's KV pool supports growth at runtime (verification gate, §10). Three possible semantics, in decreasing order of preference:

- **(α) Fully elastic.** `B` shrinks → KV pool grows to absorb freed bytes. Most useful; requires KV pool API support.
- **(β) Fixed `K_max`, shrinkable `B`.** `K` is sized at startup for worst case. `B` can still shrink within `[n_hi_min, n_hi_max]` to reduce migration churn under load and leave headroom for activation buffers / fragmentation, but freed bytes don't help KV directly. Resize signal still useful as a load-aware migration brake.
- **(γ) Both fixed.** No resize at all; only reorder. Signal (i) becomes informational. Worst case — falls back to a static-`n_hi` system with dynamic resident-set identity.

**v1 ships with whichever of α/β is feasible.** γ is acceptable as a temporary state if KV API blocks β.

### 3.2 Code organization

```
python/sglang/srt/layers/moe/
├── heter_moe.py                          # existing; +1 mutator + resident-mask path
├── heter_policy.py                       # existing; +optional resident_mask input
└── dynaexq/                              # NEW
    ├── __init__.py                       # public API: build_controller(...)
    ├── config.py                         # DynaExqConfig dataclass + CLI parsing
    ├── host_store.py                     # pinned-host BF16 store; pluggable
    ├── budget.py                         # HBM accounting + n_hi(t)
    ├── ema.py                            # long-horizon hotness from router traces
    ├── policy.py                         # top-N + hysteresis; signal (ii) detection
    ├── migration.py                      # async H2D pipeline; promote/demote/publish
    ├── controller.py                     # top-level glue + scheduler thread
    └── signals.py                        # KV pressure monitor (signal i)
```

### 3.3 Integration points (4 total)

1. **`server_args.py`** — `--dynaexq-*` CLI flags (Section 7).
2. **`model_loader/loader.py`** — after `apply_heter_precision`, if dynaexq enabled, build controller, attach to each `HeterFusedMoE`, start scheduler thread.
3. **`heter_moe.py`** — add `swap_bf16_slot(slot, eid, host_buf, stream, evt)` + `publish_slot(eid, slot)` + a `set_resident_mask(mask)` setter; `_bf16_id_remap` becomes controller-mutable; `_bf16_compact` is sized by controller's `n_hi_max` rather than by static int4-only count.
4. **Model runner / scheduler** — expose KV-cache occupancy as a callable for `signals.py`. Exact location TBD during implementation; preference is read-only callable rather than callback registration.

## 4. Data model

### 4.1 Host store (`host_store.py`)

Per-layer, per-expert pinned-memory tensors holding BF16 expert weights. Two modes:

- **`all`** (v1 default): pre-populate all experts at startup. Memory: `num_layers × num_experts × bf16_bytes_per_expert` (~54 GB for Qwen3-30B). Built by stealing what `HeterFusedMoE.from_fused_moe` would have copied to GPU.
- **`working-set`** (future): admission-controlled subset; cache miss raises `NotPromotable` and the policy demotes a different expert. Same interface.

Public API:
```python
class HostExpertStore:
    def get(self, layer_id: int, expert_id: int) -> tuple[Tensor, Tensor]: ...
    def has(self, layer_id: int, expert_id: int) -> bool: ...
```

### 4.2 GPU BF16 pool (in `HeterFusedMoE`)

The existing `_bf16_compact` tensor is reused. Its size is set by `n_hi_max` (computed once from the HBM cap), not by the static int4-only set. Slots addressed by integer index. Allocated once at startup; never resized.

A `BF16PoolFreeList` (in `migration.py`) tracks free vs occupied slots — a list of ints, no GPU state.

### 4.3 Resident map (`_bf16_id_remap`)

`int32[num_experts]`. `slot = remap[eid]`; `slot == -1` means not BF16-resident. Already exists in the codebase. Fixed shape; mutated in-place by the controller via `publish_slot`. CUDA-graph-safe because shape and storage pointer never change.

### 4.4 Resident mask

`bool[num_experts]`, controller-owned, kept in sync with the remap. Cached so the dispatch policy can `&` it into BF16 group selection without a `!= -1` reduction every step.

### 4.5 Synchronization protocol

**Promote `expert e to slot s`:**
1. Migration stream: alloc slot `s` from free list (host-side bookkeeping).
2. Migration stream: async H2D copy of BF16 weights into `_bf16_compact[s]`.
3. Migration stream: `evt.record(migration_stream)`.
4. Worker thread polls `evt.query()`. When done, calls `publish_slot(e, s)` from host.
5. Inside `publish_slot`: `_bf16_id_remap[e] = s` and `_resident_mask[e] = True` (in-place writes, fixed shape).

**Demote `expert e (was at slot s)`:**
1. Worker writes `_bf16_id_remap[e] = -1` and `_resident_mask[e] = False` first.
2. Schedule slot `s` to free list **after** one compute-stream advance. We use a `cudaEventRecord` on the compute stream after the next forward boundary; worker reaps when the event fires. This prevents a promote-into-just-freed-slot race overwriting weights an in-flight kernel still reads.

**CUDA graph caveat (verification gate):** captured decode graphs read `_bf16_id_remap` at replay time, so contents updates flow through *if* the BF16 group dispatch logic is not specialized at capture time on a fixed resident set. Verification step: capture a graph, perform a publish, replay, assert remap honored. If it turns out the captured graph specializes (e.g., Triton autotuner caches keyed on expert pattern), fallback is **drop into eager mode for one step after each publish** — fires at most once per cooldown (~2s), negligible cost.

### 4.6 Memory budget

`budget.py` exposes:
```python
class HBMBudget:
    n_hi_max: int                          # per-layer max, set at startup
    n_hi_min: int                          # per-layer min (default 0.25 * max)
    def current_n_hi(self, layer_id: int) -> int: ...
    def kv_cache_pressure(self) -> float: ...
    def proposed_resize(self) -> int | None: ...    # consults pressure + cooldown
```

`n_hi_max` is computed at startup as `floor((cap_bytes - W_nonexpert - F_runtime - I_int4 - K_max) / bf16_bytes_per_expert / num_moe_layers)`. The semantics of resize action depend on the elasticity tier from §3.1 (verification gate, §10).

## 5. Migration pipeline (`migration.py`)

### 5.1 Streams

- **Compute stream:** SGLang's default. Forward only.
- **Migration stream:** dedicated `torch.cuda.Stream()`. All H2D copies for promotion. No forward work touches it.

### 5.2 Queues

```python
promote_q: list[(layer_id, expert_id, target_slot)]   # bounded by promote_cap
demote_q:  list[(layer_id, expert_id, slot_to_free)]  # unbounded
```

Filled by the controller on trigger; drained by `MigrationWorker` thread.

### 5.3 Worker loop

Single Python thread, started at controller attach time:

```
while not stop:
    drain demote_q                           # frees slots that promotes need
    while promote_q non-empty and have_free_slot():
        (layer, eid, slot) = promote_q.pop()
        h_w13, h_w2 = host_store.get(layer.layer_id, eid)
        with torch.cuda.stream(migration_stream):
            layer.bf16_pool_w13[slot].copy_(h_w13, non_blocking=True)
            layer.bf16_pool_w2[slot].copy_(h_w2,  non_blocking=True)
            evt = torch.cuda.Event(); evt.record(migration_stream)
        in_flight.append((layer, eid, slot, evt))
    for promo in completed(in_flight):
        layer.publish_slot(promo.eid, promo.slot)
        in_flight.remove(promo)
    sleep(short_interval)                     # ~10 ms
```

### 5.4 Backpressure & admission

- Per-trigger: ≤ `promote_cap = 4` promotions / layer / trigger.
- Global in-flight: ≤ `2 × num_moe_layers` outstanding H2D copies.
- Promote with no free slot: deferred to next trigger; never blocks. INT4 fallback covers correctness.

### 5.5 Failure modes

- **No free slot, no pending demote** → log + retry next trigger.
- **Host store miss** (mode `working-set`) → skip; expert can't be promoted unless populated.
- **CUDA error on copy** → slot returned to free list; expert remains INT4. Server stays up.

## 6. Scheduler & signals

### 6.1 Scheduler thread

Wakes ~10 Hz. Polls signals. On any trigger: compute action, enqueue migrations, set cooldown timer.

### 6.2 Signal (i): KV cache pressure (load-driven resize)

```
pressure = kv_used_pages / kv_total_pages
crosses up_thresh (0.70):  shrink n_hi by Δ_resize     (if not in cooldown)
crosses down_thresh (0.50): grow n_hi by Δ_resize
```

Hysteresis via separate up/down thresholds. Cooldown = 2s default.

### 6.3 Signal (ii): EMA composition shift (workload-driven reorder)

Every `step_window = 200` forward steps:
1. Snapshot router counts from `routed_experts_capturer` device buffer.
2. Update EMA: `S[ℓ,e] ← α·S[ℓ,e] + (1-α)·c[ℓ,e]`, `α = 0.9`.
3. Compute `proposed = top_n(S[ℓ,:], current_n_hi[ℓ])`.
4. If `|proposed Δ resident[ℓ]| ≥ composition_delta = 4`: reorder.

Hysteresis margin = 1 EMA score unit at top-N selection so an expert at the boundary doesn't ping-pong.

### 6.4 Reorder action

`demote := resident \ proposed`, `promote := proposed \ resident`. Demotes first (free slots), promotes second, capped by `promote_cap = 4`/layer.

### 6.5 Resize action

- Shrink: pick lowest-EMA in resident, demote them, decrement `n_hi`.
- Grow: pick highest-EMA not in resident, promote, increment `n_hi` after promotion succeeds.

Effect on HBM depends on the elasticity tier from §3.1: under **(α)** freed bytes are returned to the KV pool; under **(β)** they sit unused as headroom (still useful — reduces concurrent migration traffic and activation-buffer pressure); under **(γ)** resize is disabled at config load time and only reorder fires.

### 6.6 Event log

Every resize/reorder writes one record to `dynaexq_events.jsonl`:

```json
{"ts": ..., "layer_id": ..., "action": "reorder|resize",
 "before_set": [...], "after_set": [...], "ema_scores": [...],
 "kv_pressure": ..., "n_hi": ...}
```

Data source for the workload-shift and elastic-budget plots.

## 7. Configuration

### 7.1 CLI flags (`server_args.py`)

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--dynaexq-enable` | bool | False | Master switch |
| `--dynaexq-hbm-cap-gb` | float | (required) | Total cap for `B + K` |
| `--dynaexq-bf16-min-ratio` | float | 0.25 | `n_hi_min / n_hi_max` |
| `--dynaexq-ema-alpha` | float | 0.9 | EMA smoothing |
| `--dynaexq-step-window` | int | 200 | Steps between EMA updates |
| `--dynaexq-cooldown-s` | float | 2.0 | Min seconds between triggers |
| `--dynaexq-promote-cap` | int | 4 | Promotions / layer / trigger |
| `--dynaexq-resize-delta` | int | 4 | Δ experts on resize |
| `--dynaexq-kv-up-thresh` | float | 0.70 | Shrink trigger |
| `--dynaexq-kv-down-thresh` | float | 0.50 | Grow trigger |
| `--dynaexq-host-store-mode` | enum | `all` | `all` / `working-set` |
| `--dynaexq-event-log` | path | None | If set, write JSONL events |

### 7.2 Recipe extension

```yaml
dynaexq:
  enable: true
  hbm_cap_gb: 36.0
  ema_alpha: 0.9
  cooldown_s: 2.0
  trigger:
    kv_pressure: { up: 0.70, down: 0.50 }
    composition_delta: 4
```

Recipe values override CLI.

## 8. Evaluation (mirrors paper §5)

### 8.1 Q1 — Quality (paper Table 4)

**Goal:** at the same HBM budget, dynamic precision allocation recovers accuracy vs static low-bit.

| Model | Methods | Notes |
|---|---|---|
| Qwen3-30B | FP16 / static Int4 / DynaExq (`n_hi=32` of 128) | DynaExq HBM matches Int4 |
| Qwen3-30B | static partial-BF16 (existing heter, fixed) / DynaExq | strictly equal HBM at same `n_hi` |
| Qwen3-80B | static Int4 / static Int2 / DynaExq (Int4 hot + Int2 cold) | DynaExq matches Int2 |
| Phi-3.5-MoE | FP16 / static Int4 / DynaExq | DynaExq matches Int4 |

**Benchmarks:** MMLU-Pro, GPQA, AIME25, GSM8K, HumanEval, WikiText-2 perplexity. Reuse existing `bench_eval` pipeline; add `--dynaexq-*` recipe variants.

**Output:** CSV at `expert_precision_assignment/experiments/data/results/dynaexq_quality/`.

### 8.2 Q2 — Performance (paper Figs 6–10)

**Goal:** at fixed HBM, DynaExq tracks static-low-bit latency closely.

| Plot | X axis | Y axis | Methods |
|---|---|---|---|
| Fig 6 (TTFT) | batch ∈ {1,2,4,8,16,32} | avg + P99 TTFT | static low-bit, static partial-BF16, DynaExq |
| Fig 7 (TPOP) | same bs | avg + P99 TPOP | same |
| Fig 9 (Throughput) | same bs | tokens/s | same |
| Fig 10 (TTFT vs prompt) | tokens ∈ {32,64,128,256,512,1024}, bs=1 | avg + P99 TTFT | same |

ExpertFlow comparison flagged as stretch goal — its own integration branch.

**Output:** plots at `expert_precision_assignment/experiments/data/results/dynaexq_perf/`.

### 8.3 Bonus dynamism demos

1. **Workload shift trace.** Mixed workload switching every 30s: WikiText → GSM8K → HumanEval → loop. Plot `dynaexq_events.jsonl` resident-set composition over time. Expect: top-N rotates at workload boundaries.
2. **Elastic budget trace.** Synthetic load ramp bs=1 → 32 → 1 over 5 minutes. Plot `n_hi(t)` and KV pressure(t) on shared axes. Expect: `n_hi` shrinks under load, regrows when load drops.

Both demos run from `expert_precision_assignment/experiments/dynaexq_demos/`.

### 8.4 Hardware

A100 (per branch's `ba7f00e25` tuned config), not paper's A6000. HBM budgets adjusted to A100 capacity. Relative ordering of methods is the load-bearing claim, not absolute latencies.

## 9. Testing

- **Unit** (`test/dynaexq/`):
  - `test_host_store.py` — pinned alloc, get/has, both modes.
  - `test_budget.py` — `n_hi_max` derivation, resize math.
  - `test_ema.py` — EMA correctness, top-N + hysteresis stability.
  - `test_pool_freelist.py` — alloc/free/exhaust/reuse.
  - `test_migration_pipeline.py` — promote/demote/publish state machine vs stub layer.
- **Integration:**
  - `test_dynaexq_e2e_smoke.py` — Qwen3-30B forward, dynaexq enabled, 64 tokens out, no crash, ≥ 1 reorder logged when router scores are perturbed.
  - `test_cudagraph_compat.py` — capture decode graph, publish, replay, assert remap honored. Verification gate from §4.5.
- Run on conda `sglang` env, GPU 4 (per project convention).

## 10. Open risks & verification gates

| Risk | Resolution |
|---|---|
| CUDA graph captures specialize on resident set | Verify with `test_cudagraph_compat.py`; fallback to one eager step per publish if needed (negligible cost) |
| KV cache size is fixed at startup; can't grow on BF16 demote | Determines elasticity tier (§3.1). v1 attempts (α); falls back to (β) if KV pool can't grow; (γ) only if neither is viable. Decision documented at first commit of `dynaexq/` package. |
| Pinned host alloc of ~54 GB may pressure system RAM | Mode `working-set` (iii) is the planned mitigation; deferred. |
| Migration stream still contends for PCIe with NCCL / disk IO | Bound by `promote_cap` + cooldown. Out-of-scope to coordinate. |

## 11. Out of scope for v1

- ExpertFlow comparison (stretch).
- `working-set` host store (mode iii).
- Multi-GPU / TP / EP coordination.
- Dynamic KV cache growth (best effort; may end up fixed-`K_max`).
- Online quality probes (LM head loss, etc.).
