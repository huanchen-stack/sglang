# Rollout Precision Experiment

This directory contains tooling for testing bf16 versus int4 rollout serving
behavior with scheduler request lifecycle traces.

## Instrumentation

Set one of these before starting `sglang.launch_server`:

```bash
export SGLANG_ROLLOUT_TRACE_DIR=/path/to/traces
# or
export SGLANG_ROLLOUT_TRACE_FILE=/path/to/scheduler.jsonl
```

The scheduler emits JSONL events for request `waiting`, `prefill`, and `decode`
spans. Events include batch size, forward mode, TP metadata, model path,
quantization, and the existing CUDA graph eligibility signal
`can_run_cuda_graph`. The tracer is CPU-side only and does not change scheduler
batch construction or CUDA graph gates.

## Typical Flow

1. Prepare custom JSONL datasets:

```bash
python rollout_request_lifetime/prepare_rollout_datasets.py \
  --output-dir /tmp/rollout_precision_datasets \
  --num-prompts 512
```

2. Launch the experiment matrix. Start with dry-run:

```bash
python rollout_request_lifetime/run_rollout_precision_experiments.py \
  --dataset-dir /tmp/rollout_precision_datasets \
  --output-dir /tmp/rollout_precision_results \
  --int4-model llama_dense_8b=/path/to/llama-8b-awq \
  --dry-run
```

Use `--model LABEL=PATH` for bf16 checkpoints and `--int4-model LABEL=PATH`
for their int4/AWQ/GPTQ counterparts. If no `--int4-model` is supplied, the
runner uses the bf16 path with `--quantization awq`, which is only valid when
that checkpoint is actually compatible with the selected quantization backend.

3. Generate plots and a Markdown report:

```bash
python rollout_request_lifetime/plot_rollout_traces.py \
  --results-dir /tmp/rollout_precision_results \
  --report /tmp/rollout_precision_results/report.md \
  --min-decode-cuda-graph-eligible 1.0
```

The runner defaults to batch sizes `128,256,512`, which are common rollout
concurrency targets for high-throughput generation. Override them with
`--batch-sizes` when your RL stack uses a different rollout fanout.
