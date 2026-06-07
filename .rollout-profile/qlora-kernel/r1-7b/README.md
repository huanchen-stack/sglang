# DeepSeek R1 Distill Qwen 7B Kernel Sweep

This directory contains the single-GPU kernel sweep for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` shapes. The canonical plots in this
directory are generated from `single_gpu/raw_all.json`.

Generated files:

- `qlora_kernel_perf_clean.png`: six-line clean comparison plot.
- `qlora_kernel_perf_clean.json`: clean plotted data.
- `raw/`: earlier one-raw-benchmark-shard-per-projection run.
- `single_gpu/`: source single-GPU rerun used for the canonical R1 plots.

Model shapes:

| Projection | Shape |
|---|---:|
| qkv | `3584 x 4608` |
| o | `3584 x 3584` |
| gate_up | `3584 x 37888` |
| down | `18944 x 3584` |

Clean plot lines:

- `bf16`
- `bf16 + sequential csgmv LoRA`
- `bf16 + two-stream Torch LoRA`
- `int4`
- `int4 + sequential csgmv LoRA`
- `int4 + two-stream Torch LoRA`

The two-stream line uses the same Torch/cuBLAS LoRA patch path with Marlin SM
reservation applied only to the two-stream Marlin call.

## Summary

The method is useful mainly at low active decode batch sizes for this 7B model.
At higher active batches, BF16 merged rollout catches up and then wins because
the R1 7B projection matrices are much smaller than the Qwen2.5 14B/32B profiling
targets.

Aggregate projection-choice estimate, using `min(bf16, int4 + two-stream Torch
LoRA)` per projection:

| Token rows | Best over BF16 | All QLoRA two-stream over BF16 | Selected projections |
|---:|---:|---:|---|
| 1 | 2.08x | 2.08x | qkv, o, gate_up, down: QLoRA |
| 4 | 1.76x | 1.76x | qkv, o, gate_up, down: QLoRA |
| 8 | 1.89x | 1.89x | qkv, o, gate_up, down: QLoRA |
| 16 | 1.83x | 1.83x | o, gate_up, down: QLoRA; qkv: BF16 |
| 32 | 1.58x | 1.52x | gate_up, down: QLoRA; qkv, o: BF16 |
| 64 | 1.37x | 1.25x | gate_up, down: QLoRA; qkv, o: BF16 |
| 128 | 1.01x | 0.95x | down: QLoRA; qkv, o, gate_up: BF16 |
| 256 | 1.00x | 0.70x | all BF16 |
| 512 | 1.00x | 0.70x | all BF16 |

Selected per-projection measurements:

| Tokens | Projection | BF16 | INT4 | INT4 + csgmv | INT4 + two-stream | BF16 / two-stream | csgmv / two-stream |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | qkv | 43.0 us | 18.4 us | 39.9 us | 34.8 us | 1.24x | 1.15x |
| 1 | o | 32.8 us | 18.4 us | 33.8 us | 23.6 us | 1.39x | 1.43x |
| 1 | gate_up | 193.5 us | 68.6 us | 81.9 us | 74.8 us | 2.59x | 1.10x |
| 1 | down | 110.6 us | 44.0 us | 93.2 us | 49.2 us | 2.25x | 1.90x |
| 128 | qkv | 45.1 us | 49.2 us | 87.0 us | 60.4 us | 0.75x | 1.44x |
| 128 | o | 36.9 us | 36.9 us | 61.4 us | 45.1 us | 0.82x | 1.36x |
| 128 | gate_up | 231.4 us | 186.4 us | 313.3 us | 237.6 us | 0.97x | 1.32x |
| 128 | down | 119.8 us | 104.4 us | 163.3 us | 114.7 us | 1.04x | 1.42x |
| 512 | qkv | 105.5 us | 113.2 us | 235.5 us | 147.5 us | 0.72x | 1.60x |
| 512 | o | 84.0 us | 81.9 us | 135.7 us | 104.4 us | 0.80x | 1.30x |
| 512 | gate_up | 545.8 us | 714.8 us | 1180.7 us | 882.7 us | 0.62x | 1.34x |
| 512 | down | 323.6 us | 346.1 us | 449.5 us | 377.9 us | 0.86x | 1.19x |

Interpretation:

- For low active batches, `int4 + two-stream Torch LoRA` is faster than BF16
  merged for all four projections.
- The two-stream method consistently improves over SGLang csgmv sequential
  QLoRA.
- For active batches around 128, only the down projection remains slightly
  competitive in the single-GPU rerun.
- For active batches 256 and 512, BF16 merged is the best kernel-level choice
  for all projections on this 7B shape.
