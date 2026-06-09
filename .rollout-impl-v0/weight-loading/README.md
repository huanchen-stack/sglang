# Rollout Weight Colocation v0

Goal: keep both weight copies resident in VRAM.

- Prefill path: primary BF16 model only.
- Decode path: INT4 shadow base weights + Torch-native two-stream BF16 LoRA.
- CUDA graph policy: follow SGLang batch capture buckets. This v0 mode uses a fixed decode topology for captured batch sizes.

Runtime flags:

```bash
--model-path <bf16-merged-model>
--lora-paths default=<lora-adapter>
--rollout-weight-colocation-int4-model-path <int4-model>
--cuda-graph-bs 8 16
--cuda-graph-max-bs 16
```

Optional:

```bash
--rollout-weight-colocation-int4-load-format <format>
--rollout-weight-colocation-int4-quantization <method>
--no-rollout-weight-colocation-force-torch2s
```

The default behavior forces `torch_native` LoRA and `SGLANG_LORA_TORCH_TWOSTREAM=1` when the INT4 shadow path is provided.

Validation:

```bash
python .rollout-impl-v0/weight-loading/run_weight_colocation_batch.py \
  --bf16-model-path <bf16-merged-model> \
  --int4-model-path <int4-model> \
  --lora-path <lora-adapter> \
  --gpu 0
```

Outputs:

- `results/weight_colocation_server.log`
- `results/weight_colocation_batch8_16.json`

Passing checks:

- server log contains INT4 shadow load confirmation
- server log contains INT4 shadow attachment count
- server log contains `path=bf16_prefill`
- server log contains `path=int4_decode`
- batch 8 and batch 16 requests complete
- JSON contains VRAM snapshots before launch, after ready, after each batch, and after shutdown

v0 limitation: CUDA graph capture requires at least one startup LoRA adapter in `--lora-paths`, because the Torch-native two-stream LoRA path has Python-controlled matmul topology during capture.
