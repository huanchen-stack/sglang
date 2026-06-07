# QLoRA Decoding Throughput

This directory is a new profiling track for rollout decoding throughput. It is
separate from `.rollout-profile/qlora-kernel/`, which only measures individual
projection kernels.

The current files are fake deliverables used to confirm the experiment shape
before implementing server/client measurement.

## Target

Use `qwen2.5-32b` for the first real evaluation.

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

## Fake Deliverables

`make_fake_deliverable.py` creates synthetic data and plots to demonstrate the
intended output shape:

- `fake_decoding_throughput.json`
- `fake_decoding_throughput.csv`
- `fake_summary_table.md`
- `fake_throughput.png`
- `fake_speedup.png`
- `fake_projection_policy.png`

Run:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python .rollout-profile/qlora-decoding-throughput/make_fake_deliverable.py
```

## Implementation Notes For Real Run

The real implementation should reference the `rollout/explore` branch for the
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
