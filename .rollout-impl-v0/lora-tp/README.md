# LoRA TP Patch Profiling

This directory isolates the TP-specific LoRA patch profiling work from the
precision-mixing server validation.

## What It Profiles

The harness profiles Qwen2.5-14B TP4 LoRA patch shapes only:

- `gate_up` column-parallel patch:
  `cat((x @ A_gate.T) @ B_gate_shard.T, (x @ A_up.T) @ B_up_shard.T)`
- `down` row-parallel patch:
  `all_reduce(x_local @ A_shard.T) @ B_replicated.T`

It does not profile the full SGLang server or the INT4 Marlin base projection.

## Regenerate Nsight

```bash
nsys profile \
  --output .rollout-impl-v0/lora-tp/nsys/tp4_lora_patch_tp_parallel \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH=/data/huanchen/sglang/python:/data/huanchen/sglang/sgl-kernel/python \
  MASTER_PORT=29647 \
  /data/huanchen/miniforge3/envs/sglang/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  .rollout-impl-v0/lora-tp/profile_tp_lora_patch.py \
  --batch-size 8 \
  --warmup 2 \
  --iters 8 \
  --profile both \
  --no-compile \
  --cuda-profiler-range \
  --output-json .rollout-impl-v0/lora-tp/nsys/tp4_lora_patch_tp_parallel.json
```

The current generated report is:

- `nsys/tp4_lora_patch_tp_parallel.nsys-rep`
- `nsys/tp4_lora_patch_tp_parallel_summary.json`
- `nsys/tp4_lora_patch_tp_parallel_kernels_cuda_gpu_kern_sum.csv`
