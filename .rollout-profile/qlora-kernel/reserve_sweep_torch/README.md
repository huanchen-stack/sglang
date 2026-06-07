# Torch LoRA Two-Stream SM Reservation Sweep

This directory contains the SM-reservation sweep for the Torch/cuBLAS LoRA two-stream path.

## Setup

Shape/profile target:

- Model config: `qwen2.5-32b`
- Projection: `down`
- Batch/token rows: `1`
- Base path: int4 GPTQ Marlin
- LoRA path: BF16 Torch/cuBLAS matmul patch on a side stream
- Scheme: `Torch QLoRA matmul two-stream`
- Warmup: `10` CUDA Graph replays
- Measurement: `100` CUDA Graph replays
- L2 flush: disabled for this focused sweep (`--l2-flush-mib 0`)

Command pattern:

```bash
mkdir -p .rollout-profile/qlora-kernel/reserve_sweep_torch
for r in 1 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 /data/huanchen/miniforge3/envs/sglang/bin/python \
    .rollout-profile/qlora-kernel/benchmark.py \
    --model-config .rollout-profile/qlora-kernel/configs/qwen2.5-32b.json \
    --projection down \
    --scheme "Torch QLoRA matmul two-stream" \
    --tokens 1 \
    --warmup 10 \
    --iters 100 \
    --l2-flush-mib 0 \
    --two-stream-reserve-sms "$r" \
    --output ".rollout-profile/qlora-kernel/reserve_sweep_torch/qwen32b_down_bs1_torch_twostream_reserve${r}.json"
done
```

## Results

| Reserved SMs | Median latency (us) | p20 (us) | p80 (us) |
|---:|---:|---:|---:|
| 1 | 70.656 | 69.632 | 70.656 |
| 2 | 70.656 | 69.632 | 70.656 |
| 4 | 72.704 | 72.704 | 73.728 |
| 8 | 72.704 | 71.680 | 73.728 |
| 16 | 74.752 | 74.752 | 75.776 |

## Conclusion

Use `--two-stream-reserve-sms 1` for the Torch LoRA two-stream path.

Reserving more SMs did not improve latency in this focused sweep. `1` and `2` tied on median, and values above that slightly regressed. The likely reason is that Torch/cuBLAS already exposes enough parallelism for the low-batch LoRA patch, so further reducing Marlin CTAs mostly hurts the base kernel without making the side-stream LoRA patch faster enough to compensate.
