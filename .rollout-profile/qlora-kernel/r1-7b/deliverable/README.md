# R1 7B Deliverable Plots

Presentation-ready single-GPU kernel plots for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.

Generated plots:

- `csgmv_lora_extended.png`: BF16 merged vs `int4 + sequential csgmv LoRA`.
- `torch_twostream_lora_extended.png`: BF16 merged vs `int4 + two-stream Torch LoRA`.
- `int4_base_vs_bf16_extended.png`: BF16 merged vs INT4 Marlin base-only reference.

Each plot uses four projection panels: `qkv`, `o`, `gate_up`, and `down`.
Token rows cover `1,4,8,16,32,64,128,256,512`.

The batch-size-8 marker is `BF16 latency / comparison latency`.

These are generated from `../single_gpu/raw_all.json`. They are kernel-only
measurements and do not accurately model KV-cache effects, scheduler behavior,
or end-to-end serving overhead.
