# QLoRA Kernel Profiling

Active files in this directory:

- `benchmark.py`: main CUDA Graph benchmark harness.
- `build_clean_data.py`: rebuilds the clean six-line comparison JSON from archived raw inputs.
- `lineplot.py`: renders `qlora_kernel_perf_clean.png` from `qlora_kernel_perf_clean.json` by default.
- `qlora_kernel_perf_clean.json`: active clean comparison data.
- `qlora_kernel_perf_clean.png`: active clean comparison plot.
- `configs/`: model/projection shape configs.
- `reserve_sweep_torch/`: focused SM-reservation sweep for the Torch LoRA two-stream path.
- `deliverable/`: presentation plots and their backing datasets.
- `nsys/peek_nsys_graph.py`: helper for rendering Nsight Systems kernel timeline peeks.

Clean plot line set:

- `bf16`
- `bf16 + sequential csgmv LoRA`
- `bf16 + two-stream Torch LoRA`
- `int4`
- `int4 + sequential csgmv LoRA`
- `int4 + two-stream Torch LoRA`

Archived/intermediate files:

- `archive/raw-inputs/`: full raw benchmark data and older intermediate JSON/PNG outputs used to build the clean plot.
- `archive/old-shards/shards/`: per-model/per-projection shard benchmark outputs from earlier runs.
- `archive/smoke/`: fake and smoke-test outputs.
- `tp-splitk(deprecate)/`: deprecated TP-style split-K probe and associated profiles.

To rebuild the active clean plot:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python .rollout-profile/qlora-kernel/build_clean_data.py
/data/huanchen/miniforge3/envs/sglang/bin/python .rollout-profile/qlora-kernel/lineplot.py --linear-y
```

## Rollout LoRA Interpretation

In PEFT training, LoRA normally stays as a separate adapter branch:

```text
y = x W_base^T + scale * x A^T B^T
```

The frozen base weight is not permanently modified during training, because
gradients should update only LoRA parameters.

For rollout, the forward pass is inference-only, so BF16 PEFT has two valid
serving representations:

- `adapter mode`: keep the base BF16 weight and compute the LoRA patch during
  rollout.
- `merged mode`: materialize `W_eff = W_base + scale * (B @ A)` and run rollout
  as a normal BF16 model without separate LoRA kernels.

For verl, the default is adapter mode. The config default is
`actor_rollout_ref.model.lora.merge: False`, and code paths treat missing
`merge` as false via `model_config.lora.get("merge", False)`. Merged rollout
must be enabled explicitly, for example with:

```bash
+actor_rollout_ref.model.lora.merge=True
```

The presentation BF16 line is therefore interpreted as the stronger
`bf16 PEFT rollout, merged mode` baseline. This is especially relevant for long
CoT or agentic RL, where many rollout tokens can amortize the one-time
merge/sync overhead after a policy update.

The int4 QLoRA path is different. Merging a BF16 LoRA delta into int4 weights is
not a cheap representation change: it would require dequantizing and then
re-quantizing/calibrating the modified weights, which is too expensive and
approximate to do per rollout policy update. Keeping int4 base weights plus a
separate BF16 LoRA patch is the practical serving workload for QLoRA rollout.
