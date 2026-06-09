# Rollout Decode Throughput

This validation measures the rollout implementation, not the old profiling
setup:

- BF16 primary weights and INT4 shadow weights are both resident in VRAM.
- Prefill uses BF16 only.
- Decode uses the frontier precision policy derived from
  `.rollout-profile/qlora-decoding-throughput/frontier_decoding_throughput.json`,
  whose projection choices came from `.rollout-profile/qlora-kernel`.
- Each batch launches its own server with only that batch captured and compiled:
  `--cuda-graph-bs == --cuda-graph-max-bs == --torch-compile-max-bs`.

Because duplicated weights reduce available KV/cache memory, the default sweep
is progressive and stops at the first failed batch. Use
`--continue-after-failure` to record later failures too.

Default Qwen2.5 14B TP1 run:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-v0/decoding-throughput/run_decoding_throughput.py
```

Useful shorter probe:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-v0/decoding-throughput/run_decoding_throughput.py \
  --batch-sizes 1 4 8 16 32 64 128
```

Parallel TP1 run across all eight local GPUs:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-v0/decoding-throughput/run_decoding_throughput.py \
  --parallel-gpus 0,1,2,3,4,5,6,7
```

Flipped two-stream experiment:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-v0/decoding-throughput/run_decoding_throughput.py \
  --parallel-gpus 0,1,2,3,4,5,6,7 \
  --flip-twostream \
  --out-dir .rollout-impl-v0/decoding-throughput/results/qwen2.5-14b-tp1-colocated-flipped
```

Parallel mode launches one independent TP1 server/client worker per GPU with a
unique port. The parent process merges child `summary.json` files into the main
summary. TorchInductor/Triton caches and `TMPDIR` are redirected to the short
path `/data/huanchen/tmp/rollout-dt` so parallel compilation does not fill
`/tmp` or exceed ZMQ IPC path length limits.

Outputs:

- `frontier_precision_policy.json`: rollout policy consumed by the server.
- `results/qwen2.5-14b-tp1-colocated/summary.json`: aggregate results.
- `results/qwen2.5-14b-tp1-colocated/summary.csv`: compact table.
- `results/qwen2.5-14b-tp1-colocated/rollout_colocated_frontier.bs*.json`:
  per-batch client traces.
- `results/qwen2.5-14b-tp1-colocated/rollout_colocated_frontier.bs*.server.log`:
  server logs with rollout path markers.
