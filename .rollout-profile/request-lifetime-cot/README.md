# Request Lifetime CoT

This experiment measures CoT request-lifetime traces with the missing
sequence-length information included.

The old lifecycle trace only stored request finish times. This harness records
streaming token events:

```text
request_id, timestamp_s, decoded_tokens
```

That makes it possible to reconstruct both:

- live request count over decode time
- decoded-token/KV pressure over decode time

## Current Target

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Dataset: `PRIME-RL/Eurus-2-RL-Data`, materialized locally as
  `datasets/eurus-2-rl-qwen2.5-14b-instruct.jsonl`
- Tensor parallel settings: `TP1` and `TP4`
- Batch sizes: `128, 256, 512`
- EOS: respected
- Max output length: `32768`
- Server context length: `36864`, so prompt tokens still fit with a full 32K
  generation cap.
- Sampling: `temperature=0.9`, `top_p=0.95`, `top_k=20`

Specifically, for each prompt we set `temperature=0.9` and cap generation at
32K tokens, matching common RL training rollout settings.

For Qwen2.5-14B-Instruct, SGLang derives a 32K context length by default. The
experiment uses `context_length=36864` so prompt tokens still fit with a full
32K generation cap; set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` when
launching these runs.

The old DeepSeek R1 Distill Qwen 7B traces are kept under the grouped results
directory for comparison, but the active dataset/model target is Qwen 14B on
Eurus-2-RL.

## Run

Dry run:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/request-lifetime-cot/run_lifetime_cot.py \
  --config .rollout-profile/request-lifetime-cot/configs/qwen2.5-14b-instruct-eurus-2-rl.json \
  --dry-run
```

Prepare Eurus-2-RL prompts:

```bash
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/request-lifetime-cot/prepare_eurus_dataset.py \
  --output .rollout-profile/request-lifetime-cot/datasets/eurus-2-rl-qwen2.5-14b-instruct.jsonl \
  --limit 2048 \
  --per-ability-limit 1024 \
  --seed 0 \
  --shuffle-buffer 100000
```

Real run:

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/request-lifetime-cot/run_lifetime_cot.py \
  --config .rollout-profile/request-lifetime-cot/configs/qwen2.5-14b-instruct-eurus-2-rl.json \
  --output-dir .rollout-profile/request-lifetime-cot/results \
  --result-group qwen2.5-14b-instruct-eurus-2-rl-tp4
```

Run one batch:

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
/data/huanchen/miniforge3/envs/sglang/bin/python \
  .rollout-profile/request-lifetime-cot/run_lifetime_cot.py \
  --config .rollout-profile/request-lifetime-cot/configs/qwen2.5-14b-instruct-eurus-2-rl.json \
  --batch-size 128 \
  --tp-size 4 \
  --gpus 0,1,2,3 \
  --output-dir .rollout-profile/request-lifetime-cot/results \
  --result-group qwen2.5-14b-instruct-eurus-2-rl-tp4
```

## Outputs

Each batch writes to a grouped result directory:

```text
.rollout-profile/request-lifetime-cot/results/<model-dataset-tpN>/bs<N>/
```

Current groups:

- `qwen2.5-14b-instruct-eurus-2-rl-tp4`
- `qwen2.5-14b-instruct-eurus-2-rl-tp1`
- `deepseek-r1-distill-qwen-7b-reasoning-eos-cap1024-tp4`

Important files:

- `token_events.csv`: exact streaming events, one row per observed decoded-token
  update. This is the source of truth for sequence length at timestamp.
- `requests.json`: per-request start/finish/length/finish-reason summary.
- `lifecycle.json`: compact aggregate summary.
- `timeline_sampled.csv`: sampled live-request and live decoded-token pressure.
- `lifecycle.png` and `lifecycle.pdf`: paper-sized plot.

Raw event CSVs are ignored by git because the bs512 32k cap can produce tens of
millions of events.
