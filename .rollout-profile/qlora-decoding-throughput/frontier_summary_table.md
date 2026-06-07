| Batch | qkv | o | gate_up | down | BF16 tok/s | QLoRA tok/s | Mixed est tok/s | Frontier tok/s | Frontier speedup |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|
| 1 | qlora | qlora | qlora | qlora | 53.7 | 116.1 | 116.1 | 116.1 | 2.16x |
| 4 | qlora | qlora | qlora | qlora | 182.8 | 521.6 | 521.6 | 521.6 | 2.85x |
| 8 | qlora | qlora | qlora | qlora | 432.4 | 934.3 | 934.3 | 934.3 | 2.16x |
| 16 | qlora | qlora | qlora | qlora | 855.3 | 1844.3 | 1844.3 | 1844.3 | 2.16x |
| 32 | qlora | qlora | qlora | qlora | 1551.8 | 2877.0 | 2877.0 | 2877.0 | 1.85x |
| 64 | qlora | bf16 | qlora | qlora | 2893.5 | 3897.0 | 3932.3 | 3932.3 | 1.36x |
| 128 | bf16 | bf16 | bf16 | qlora | 4545.7 | 4909.5 | 4610.1 | 4909.5 | 1.08x |
| 256 | bf16 | bf16 | bf16 | bf16 | 5857.7 | 5391.3 | 5857.7 | 5857.7 | 1.00x |
| 512 | qlora | bf16 | bf16 | bf16 | 6484.9 | 5663.8 | 6587.5 | 6587.5 | 1.02x |
