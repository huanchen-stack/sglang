# Old QLoRA Torch2S BS16 Nsight Check

Files:

- `old_qlora_torch_twostream_bs16_graph.nsys-rep`
- `old_qlora_torch_twostream_bs16_graph.server.log`
- `old_qlora_torch_twostream_bs16_graph.json`
- `old_qlora_torch_twostream_bs16_eager.nsys-rep`
- `old_qlora_torch_twostream_bs16_eager.server.log`
- `old_qlora_torch_twostream_bs16_eager.json`

Setup:

- Both runs use the old `qlora_torch_twostream` serving path on `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4`.
- The synthetic adapter `/data/huanchen/._delete/adapters/qwen2.5-14b_rank16_zero_bf16` is loaded in both runs.
- `SGLANG_LORA_NVTX=1` adds NVTX ranges only around the real LoRA matmul sites in `python/sglang/srt/lora/torch_ops/lora_ops.py`:
  - `lora_a_mm`
  - `lora_b_addmm`

Observation:

- CUDA graph enabled (`old_qlora_torch_twostream_bs16_graph.nsys-rep`)
  - `nsys stats --report nvtx_pushpop_sum` shows only:
    - `CCCL:cub::DeviceScan::InclusiveScan`
  - `lora_a_mm` count: `0`
  - `lora_b_addmm` count: `0`
  - decode throughput: `1800.06 tok/s`

- CUDA graph disabled (`old_qlora_torch_twostream_bs16_eager.nsys-rep`)
  - `nsys stats --report nvtx_pushpop_sum` shows:
    - `lora_a_mm`: `98304` instances
    - `lora_b_addmm`: `172032` instances
  - decode throughput: `86.20 tok/s`

Conclusion:

- The old standard LoRA CUDA graph capture did not execute the real LoRA matmul sites at capture time.
- Since CUDA graph replay can only replay kernels that were present during capture, the old graph-backed `qlora_torch_twostream` decode path was not replaying real LoRA GEMMs.
- The eager control confirms the zero-valued rank-16 adapter does execute real LoRA GEMM work when CUDA graph is not suppressing it.
