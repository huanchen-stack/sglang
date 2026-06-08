# Rollout Dynamic Precision Implementation

This directory tracks the implementation work for decode-only dynamic precision
switching in RL rollout serving.

## Current Integration

The SGLang runtime now accepts a rollout precision policy:

```bash
python -m sglang.launch_server \
  --enable-lora \
  --lora-paths adapter=/path/to/adapter \
  --rollout-precision-policy .rollout-impl-codex/policies/qwen2.5-14b-eurus-demo.json \
  --rollout-precision-assume-merged-bf16
```

Key semantics:

- Prefill never uses the rollout precision policy. It always receives a disabled
  decision, so it stays on the base BF16 path.
- Decode batches select a policy window by live batch size.
- Projection groups are `qkv`, `o`, `up`, and `down`.
- `int4_torch_twostream` enables the existing torch-native two-stream LoRA path
  for selected projection groups.
- `bf16_merged` skips LoRA compute only when
  `--rollout-precision-assume-merged-bf16` is set. Without that flag, LoRA math
  is retained for correctness.
- Providing a policy switches `--lora-backend` to `torch_native` by default,
  because the target optimized path depends on torch GEMM. Pass
  `--no-rollout-precision-force-torch-lora` to keep the configured backend.

## Backend Boundary

The full dual-resident BF16 + INT4 base-weight loader is intentionally not hidden
inside the LoRA wrappers. The current patch adds the policy/control plane and
decode-only LoRA routing. The actual INT4 base-weight replacement should be wired
at the model-loader or linear-method level so that each captured decode graph can
bind the intended base-weight implementation.

## Tests

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python -m pytest \
  test/manual/lora/test_rollout_precision_policy.py -q
```

