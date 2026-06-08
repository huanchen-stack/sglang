# Dynamic Decode Verification

This directory verifies fixed-length decoding throughput through the actual
dynamic-precision server path implemented in this branch.

It deliberately reuses the same input JSON used by the live rollout runs:

```text
.rollout-impl-codex/real-deliverable/configs/qwen2.5-14b-eurus-dynamic-tp4.json
```

The verification client is the original fixed-decode profiling client from:

```text
.rollout-profile/qlora-decoding-throughput/decoding_client.py
```

What this validates:

- the real server loads BF16 main weights, the GPTQ INT4 shadow model, the LoRA
  adapter, and the rollout precision policy;
- requests go through `/generate` with the configured LoRA name;
- decode uses `max_new_tokens=256` and `ignore_eos=True`, matching the prior
  decode-throughput profiling setup;
- measured throughput can be compared against the prior profiling numbers in
  `.rollout-profile/qlora-decoding-throughput/decoding_throughput.csv`.

What this does not validate:

- long-tail EOS behavior;
- stochastic CoT rollout length changes;
- repeated policy switching as the live batch drains. With `ignore_eos=True`
  and fixed 256-token outputs, the active batch stays nearly constant until all
  requests finish.

Run a dry check:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-codex/decoding-throughput-verification/run_dynamic_decode.py \
  --dry-run
```

Run selected batches:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-codex/decoding-throughput-verification/run_dynamic_decode.py \
  --batch-size 128 --batch-size 256 --batch-size 512 \
  --gpus 0,1,2,3
```

Summarize results:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-codex/decoding-throughput-verification/summarize_dynamic_decode.py
```
