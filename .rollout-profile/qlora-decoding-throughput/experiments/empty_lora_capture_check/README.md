# Empty LoRA Capture Check

This folder groups the artifacts created to test whether the old
`qlora_torch_twostream` decoding-throughput run was effectively measuring plain
INT4 decode because CUDA graph capture used empty LoRA IDs.

Contents:

- `configs/qwen2.5-14b-int4-no-lora.json`
  - old-profile control config for GPTQ INT4 decode with no LoRA adapter
- `measurements_14b_int4_no_lora/`
  - full no-LoRA control sweep outputs and server logs
- `int4_no_lora_vs_qlora_torch_twostream.csv`
- `int4_no_lora_vs_qlora_torch_twostream.json`
- `int4_no_lora_vs_qlora_torch_twostream.png`
- `int4_no_lora_vs_qlora_torch_twostream_ratio.png`
  - control-vs-old-QLoRA comparisons
- `nsys/`
  - BS16 graph/eager `nsys` traces and the summary note

Key conclusion:

- The no-LoRA INT4 control tracks the old `qlora_torch_twostream` throughput
  closely.
- The `nsys` check shows real LoRA matmul NVTX ranges in eager decode, but not
  in the old CUDA-graph-backed path.
