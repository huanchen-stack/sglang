# TP-Scale QLoRA Torch Two-Stream Kernel Profiling

This directory measures Qwen2.5-32B TP-shaped projection kernels for:

- `bf16/tp{1,2,4}`: BF16 dense base projection.
- `marlin/tp{1,2,4}`: int4 GPTQ Marlin base projection.
- `qlora/tp{1,2,4}`: int4 GPTQ Marlin base plus BF16 Torch LoRA patch with flipped two-stream overlap.

The local benchmark copy defaults to the current rollout policy:

- flipped stream layout: Marlin base runs on the side stream, Torch LoRA runs on the current stream;
- `--two-stream-reserve-sms 1`, meaning Marlin leaves one SM available for LoRA progress;
- torch.compile enabled;
- CUDA Graph capture/replay enabled;
- default 96 MiB L2 flush for full JSON measurements.

TP shape convention:

- `qkv` and `gate_up` are column-parallel local shapes, so output slices are divided by TP.
- `o` and `down` are row-parallel local shapes, so input features are divided by TP.
- Full JSON measurements are single-process local-shape measurements.
- Nsight profiles are launched with `torchrun --nproc_per_node=TP`, so TP2 and TP4 reports contain all participating GPU ranks in one `.nsys-rep`/SQLite/PNG.
- The benchmark does not include distributed all-reduce timing; it profiles per-rank projection kernels.

Run everything with GPU fan-out:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-qlora-torch2s-kernel/tp-scale/run_tp_scale.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --force
```

The full measurement JSONs are written under `bf16/tp*/`, `marlin/tp*/`, and `qlora/tp*/`.
Each TP/scheme also gets a focused Nsight profile for `down` with token rows
`1`, exported to SQLite and rendered to a `_peek.png` timeline with
`nsys/peek_nsys_graph.py`.

Nsight profiles use:

```text
--trace=cuda,nvtx,osrt
--trace-fork-before-exec=true
--cuda-graph-trace=node
--force-overwrite=true
--capture-range=cudaProfilerApi
--capture-range-end=stop
```

The peek PNG labels lanes as `gpu <deviceId> stream <streamId>`, so TP2 should
show two GPU lanes and TP4 should show four GPU lanes.
