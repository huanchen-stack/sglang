# Presentation Plots

This directory contains presentation-only plots with token rows extended to
`512`.

Generated plots:

- `csgmv_lora_extended.png`: compares plain `bf16` and
  `int4 + sequential csgmv LoRA`.
- `torch_twostream_lora_extended.png`: compares plain `bf16` and
  `int4 + two-stream Torch LoRA`.
- `int4_base_vs_bf16_extended.png`: compares plain `bf16` and plain `int4`
  with no LoRA patch. This is the theoretical best-speedup reference for an
  int4 rollout path without LoRA overhead.

The BF16 line is base-only, representing merged/effective BF16 rollout weights
without a separate LoRA patch kernel. The base-only `int4` line is intentionally
excluded.

## Rollout Interpretation

For BF16 PEFT rollout, using the base-only BF16 line assumes merged/effective
rollout weights:

```text
W_eff = W_base + scale * (B @ A)
```

This is valid because rollout is inference-only. In systems such as verl, the
default LoRA rollout mode is adapter mode (`model.lora.merge=False`), but merged
rollout can be enabled explicitly. Merging is a stronger BF16 rollout baseline
when long CoT or agentic rollouts generate enough tokens to amortize the
one-time merge/sync cost after each policy update.

The int4 QLoRA path is not treated the same way. Merging the BF16 LoRA delta into
int4 weights would require dequantization plus re-quantization/calibration of
the modified weights, which is too expensive and approximate to be worthwhile
per rollout update. The practical int4 QLoRA serving workload is therefore int4
base weights plus a separate BF16 LoRA patch path.

Each subplot includes a batch-size-8 indicator labeled as `xN.N`, computed as
`plain bf16 latency / int4+LoRA latency`.

Both plots cover:

- Models: `qwen2.5-14b`, `qwen2.5-32b`
- Projections: `qkv`, `o`, `gate_up`, `down`
- Token rows: `1,4,8,16,32,64,128,256,512`

The `512` points come from `extended_512_1024_raw.json`; lower batch-size
points come from the active clean benchmark data. The `1024` points are omitted
from the rendered presentation plots.

The current backing datasets are:

- `csgmv_vs_bf16_to512.json`
- `torch_twostream_vs_bf16_to512.json`
- `int4_base_vs_bf16_to512.json`
