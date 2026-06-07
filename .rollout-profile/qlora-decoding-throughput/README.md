# QLoRA Decoding Throughput

This directory is a new profiling track for rollout decoding throughput. It is
separate from `.rollout-profile/qlora-kernel/`, which only measures individual
projection kernels.

The `frontier_*` deliverables are presentation mockups generated from the real
14B serving sweep plus the kernel profile. The BF16 and all-QLoRA numbers come
from `decoding_throughput.csv`; the per-projection mixed frontier is estimated
from `.rollout-profile/qlora-kernel/deliverable/torch_twostream_vs_bf16_to512.json`
because the current server can run whole-policy BF16 or whole-policy QLoRA, not
per-projection mixed weights.

## Target

Use `qwen2.5-14b` for the current evaluation. The 32B config remains available,
but BF16 32B did not leave enough practical headroom for the 512-request sweep
under this setup.

For each decode batch size from the kernel deliverable:

```text
1, 4, 8, 16, 32, 64, 128, 256, 512
```

measure decode throughput for rollout serving with:

- forced generation length: `256` decode tokens
- EOS ignored
- prefill excluded from the throughput metric
- CUDA Graph and `torch.compile` enabled on the decode path

The client should send `N` requests for each target batch size. The measurement
window starts after all prefill work has completed and decoding begins, then
stops after all requests finish 256 decode tokens.

## Paths To Compare

The serving code should allow per-projection path selection for:

```text
qkv, out, gate_up, down
```

Each projection can use one of:

- `bf16_merged`: BF16 PEFT rollout in merged/effective-weight mode, with no
  separate LoRA patch compute.
- `qlora_torch_twostream`: int4 Marlin base plus BF16 Torch/cuBLAS LoRA patch on
  a side stream, with the SM reservation conclusion from the kernel sweep
  (`--two-stream-reserve-sms 1`).

The intended final result should include:

- BF16 merged rollout throughput.
- QLoRA all-projection Torch two-stream throughput.
- A selective frontier where each projection independently picks BF16 merged or
  QLoRA Torch two-stream for that batch size.

## Frontier Mockup Deliverables

`make_frontier_deliverable.py` reads `decoding_throughput.csv` and the kernel
profile, picks the faster mode for each projection and batch size, and writes
presentation mockups:

- `frontier_decoding_throughput.json`
- `frontier_decoding_throughput.csv`
- `frontier_summary_table.md`
- `frontier_throughput.png`
- `frontier_speedup.png`
- `frontier_projection_policy.png`

Run:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python .rollout-profile/qlora-decoding-throughput/make_frontier_deliverable.py
```

Current 14B kernel-guided per-projection policy:

```text
batch 1..32: all projections int4 + BF16 LoRA Torch two-stream
batch 64: qkv/down/gate_up int4 + BF16 LoRA Torch two-stream, o BF16
batch 128: down int4 + BF16 LoRA Torch two-stream, qkv/o/gate_up BF16
batch 256: all projections BF16
batch 512: qkv int4 + BF16 LoRA Torch two-stream, down/o/gate_up BF16
```

The mixed-throughput line is an estimate, not a new serving measurement. It
adjusts the nearest measured whole-policy serving step time by the per-layer
kernel latency delta. A true per-projection frontier still requires a mixed-
weight serving path.

## Implementation Notes For Real Run

The real implementation should reference the `rollout/exploring` branch for the
serving-code changes used to switch projection implementations and run the
Torch two-stream LoRA path.

The benchmark should use serving-level measurement rather than isolated kernels:

1. Start the server with the selected projection policy.
2. Send `N` client requests for a target decode batch size.
3. Detect or record the point where all prefill has completed.
4. Force exactly 256 generated tokens per request and ignore EOS.
5. Report pure decode throughput:

```text
tokens_per_second = N * 256 / decode_time_seconds
```

The main presentation question is whether selective QLoRA projection choices can
approach the ideal int4 speedup while avoiding the LoRA overhead that hurts some
projection/batch-size regions.

## Real Harness

Files:

- `configs/qwen2.5-14b.json`: current 14B server/sweep config.
- `configs/qwen2.5-32b.json`: archived editable 32B server/sweep config.
- `decoding_client.py`: sends a fixed concurrent batch of streaming requests.
- `run_decoding_sweep.py`: launches one SGLang server per scheme, runs the
  client for each batch size, and writes
  `measurements_14b_current/*.json`.
- `plot_decoding.py`: converts `measurements_14b_current/*.json` into
  `decoding_throughput.csv`, `summary_table.md`, and
  `decoding_throughput.png`.

Directory layout:

- `measurements_14b_current/`: raw JSONs and server logs for the current all-GPU
  14B sweep used by the plots.
- `deprecated/raw_runs/`: older 32B attempts, retry-only data, and superseded
  14B raw sweeps. Do not use these for the current figures.

The decode metric is:

```text
decode_start = min(first streamed token timestamp across all requests)
decode_end   = max(last streamed token timestamp across all requests)
tok/s        = observed generated tokens / (decode_end - decode_start)
```

This intentionally excludes prefill/TTFT before the first emitted decode token,
but keeps the full decode window. Do not use `max(first_token_time)` as the
decode start while still counting all generated tokens: at large batch sizes
SGLang can stagger requests in waves, and that denominator/numerator mismatch
artificially inflates throughput. The harness also reports an
`all_started_decode_tok_s` diagnostic using `max(first_token_time)`, but it only
counts token events after that timestamp.

Important caveat: decode throughput is strongly coupled to KV-cache length, not
only to the projection kernels. As rollout context grows, attention increasingly
streams a large KV cache, which can become memory-bandwidth and capacity
dominant. That long-tail regime can cap or even erase the throughput gain from
faster int4/QLoRA projection paths. The first sweep should therefore report the
prompt/KV length used for every data point, and a follow-up sweep should vary KV
length explicitly. Otherwise the result may only show that model-weight loading
or projection GEMMs are faster at short context, while the target long-CoT or
agentic rollout workload is bounded by KV-cache traffic.

Before running, verify the paths in `configs/qwen2.5-14b.json`:

- `bf16_merged.model_path`: BF16 model with LoRA already merged/effective, or a
  normal BF16 baseline if you want no adapter.
- `qlora_*.model_path`: GPTQ/int4 model served by Marlin.
- `qlora_*.lora_path` and `--lora-paths default=...`: BF16 LoRA adapter path.

Example dry run:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/qlora-decoding-throughput/run_decoding_sweep.py \
  --config .rollout-profile/qlora-decoding-throughput/configs/qwen2.5-14b.json \
  --scheme qlora_torch_twostream \
  --batch-size 8 \
  --gpu 0 \
  --dry-run
```

Example real run for one GPU:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/qlora-decoding-throughput/run_decoding_sweep.py \
  --config .rollout-profile/qlora-decoding-throughput/configs/qwen2.5-14b.json \
  --gpu 0

/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/qlora-decoding-throughput/plot_decoding.py
```

To fan schemes out across multiple GPUs, use `--gpus`; the runner assigns one
scheme per GPU and increments ports from `--port`:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/qlora-decoding-throughput/run_decoding_sweep.py \
  --config .rollout-profile/qlora-decoding-throughput/configs/qwen2.5-14b.json \
  --gpus 0,1,2
```

The `qlora_torch_twostream` scheme runs SGLang with:

```text
--lora-backend torch_native
SGLANG_LORA_TORCH_TWOSTREAM=1
SGLANG_LORA_TWOSTREAM_RESERVE_SMS=1
SGLANG_MARLIN_RESERVE_SMS=1
```

The two-stream layer hook is experimental and opt-in. It is active only for
`torch_native` LoRA backend, active decode LoRA batches, and CUDA tensors. The
LoRA patch is still implemented through torch-native matmuls so it can run under
`--enable-torch-compile` and CUDA graph capture. SM reservation is applied by the
server environment for this scheme only; the forward path does not mutate env
vars. SGLang's default `csgmv` path is unchanged.

Current limitation: this harness can compare executable whole-server policies
(`bf16_merged`, `qlora_csgmv`, `qlora_torch_twostream`). A true single server
that mixes BF16 merged projections and int4 QLoRA projections independently for
`qkv/out/gate_up/down` would require a custom per-layer weight loading/quant
policy, not just a benchmarking flag. The fake `selective_frontier` plots remain
the intended presentation shape for that future mixed-weight implementation.
