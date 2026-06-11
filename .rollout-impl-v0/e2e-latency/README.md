## E2E Latency

This experiment measures end-to-end request lifetime for two TP4 serving modes
on the same 128-request Eurus batch:

- `bf16`: regular Qwen2.5-14B BF16 serving
- `dynamic`: rollout/impl_v0 weight-colocated serving with BF16 prefill and
  batch-conditioned mixed decode precision from
  `../decoding-throughput/frontier_precision_policy.json`

The harness does three things:

1. ranks Eurus prompts by prompt length
2. screens them at increasing output caps while respecting EOS
3. keeps searching until it finds one 128-request batch where each serving
   mode has at least one request reach the final output cap

The final retained artifacts are:

- one sampled dataset file per mode
- one run directory per mode with `server.log`, `requests.json`,
  `token_events.csv`, `timeline_*.csv`, `lifecycle.json`, and `lifecycle.png`
- one comparison summary and one comparison plot

## Run

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-impl-v0/e2e-latency/run_e2e_latency.py \
  --output-dir .rollout-impl-v0/e2e-latency/results/qwen2.5-14b-eurus-tp4-cap16k
```

Important defaults:

- model: `Qwen/Qwen2.5-14B-Instruct`
- dynamic INT4 shadow: `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4`
- dataset: `../request-lifetime-cot/datasets/eurus-2-rl-qwen2.5-14b-instruct.jsonl`
- TP: `4`
- GPUs: `0,1,2,3` for BF16 and `4,5,6,7` for dynamic
- output cap: `16000`
- EOS: respected
- screening caps: `2048, 4096, 8192, 16000`
- batch validity: at least one cap-hit request per mode in the same 128-request batch
- sampling: `temperature=0.9`, `top_p=0.95`, `top_k=20`
- batch-conditioned dynamic policy:
  `../decoding-throughput/frontier_precision_policy.json`

The dynamic server keeps a startup zero LoRA adapter resident so the INT4 +
Torch2S decode topology is present during CUDA graph capture. Requests sent to
the dynamic server explicitly set `lora_path=default`; BF16 requests do not.
