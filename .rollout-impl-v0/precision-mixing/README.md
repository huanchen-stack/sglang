# Rollout Precision Mixing

This deliverable validates rollout decode precision selection while keeping
both model copies resident in VRAM:

- Prefill always uses the primary BF16 model.
- Decode qkv/o use the primary BF16 projection.
- Decode up/down use the INT4 shadow projection plus the direct compiled
  Torch2S LoRA patch.

The default policy is:

```json
{
  "0-8": {
    "qkv": "bf16",
    "o": "bf16",
    "up": "int4+torch2s",
    "down": "int4+torch2s"
  }
}
```

Run Qwen2.5 14B:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python .rollout-impl-v0/precision-mixing/run_precision_mixing_batch.py \
  --bf16-model-path Qwen/Qwen2.5-14B-Instruct \
  --int4-model-path Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
  --lora-path default \
  --lora-startup-arg default=/data/huanchen/._delete/adapters/qwen2.5-14b_rank16_zero_bf16 \
  --gpu 0 \
  --port 30080 \
  --ready-timeout-s 1800 \
  --decode-tokens 32 \
  --out-dir .rollout-impl-v0/precision-mixing/results/qwen2.5-14b-tp1 \
  --extra-server-arg=--max-running-requests \
  --extra-server-arg=8 \
  --extra-server-arg=--attention-backend \
  --extra-server-arg=triton \
  --extra-server-arg=--disable-radix-cache \
  --extra-server-arg=--enable-torch-compile \
  --extra-server-arg=--torch-compile-max-bs \
  --extra-server-arg=8 \
  --extra-server-arg=--max-lora-rank \
  --extra-server-arg=16 \
  --extra-server-arg=--max-loras-per-batch \
  --extra-server-arg=1
```
