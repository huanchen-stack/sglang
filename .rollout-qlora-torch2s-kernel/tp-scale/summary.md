# TP-Scale Summary

Median latency in microseconds. `sum_qkv_o_gate_up_down` is a simple sum of the four projection-group medians, not an end-to-end layer measurement.

| kind | TP | token rows | summed projection median us |
|---|---:|---:|---:|
| bf16 | 1 | 1 | 715.104 |
| bf16 | 1 | 256 | 1129.632 |
| bf16 | 2 | 1 | 396.960 |
| bf16 | 2 | 256 | 643.520 |
| bf16 | 4 | 1 | 233.152 |
| bf16 | 4 | 256 | 347.136 |
| marlin | 1 | 1 | 247.808 |
| marlin | 1 | 256 | 1474.560 |
| marlin | 2 | 1 | 159.744 |
| marlin | 2 | 256 | 673.792 |
| marlin | 4 | 1 | 116.736 |
| marlin | 4 | 256 | 436.736 |
| qlora | 1 | 1 | 315.904 |
| qlora | 1 | 256 | 1644.032 |
| qlora | 2 | 1 | 223.232 |
| qlora | 2 | 256 | 904.704 |
| qlora | 4 | 1 | 177.664 |
| qlora | 4 | 256 | 579.584 |

Full per-projection values are in `summary.csv` and the per-run JSON files.

## Real SGLang TP up/down profile

Generated on 2026-06-10 with `sglang_tp_linear_benchmark.py`, token rows 1,
warmup 5, iters 10, CUDA Graph enabled, torch.compile enabled, and forced NCCL
all-reduce for TP2/TP4. QLoRA TP1/TP2/TP4 JSON and Nsight artifacts were
refreshed in `sglang-tp/qlora/tp*-up-down/nsys/`.

| TP | BF16 median us | INT4 Marlin median us | QLoRA median us | QLoRA/BF16 | QLoRA/INT4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 539.488 | 172.320 | 227.232 | 0.421x | 1.319x |
| 2 | 308.768 | 139.568 | 197.200 | 0.639x | 1.413x |
| 4 | 200.912 | 95.712 | 163.216 | 0.812x | 1.705x |

| TP | kind | median us | min us | max us | source JSON |
|---:|---|---:|---:|---:|---|
| 1 | bf16 | 539.488 | 535.072 | 606.176 | `sglang-tp/bf16/tp1-up-down/nsys/qwen32b_tp1_bf16_up_down_bs1.json` |
| 1 | marlin | 172.320 | 169.472 | 239.840 | `sglang-tp/marlin/tp1-up-down/nsys/qwen32b_tp1_marlin_up_down_bs1.json` |
| 1 | qlora | 227.232 | 211.840 | 419.680 | `sglang-tp/qlora/tp1-up-down/nsys/qwen32b_tp1_qlora_up_down_bs1.json` |
| 2 | bf16 | 308.768 | 307.040 | 352.896 | `sglang-tp/bf16/tp2-up-down/nsys/qwen32b_tp2_bf16_up_down_bs1.json` |
| 2 | marlin | 139.568 | 126.592 | 176.736 | `sglang-tp/marlin/tp2-up-down/nsys/qwen32b_tp2_marlin_up_down_bs1.json` |
| 2 | qlora | 197.200 | 178.752 | 4846.016 | `sglang-tp/qlora/tp2-up-down/nsys/qwen32b_tp2_qlora_up_down_bs1.json` |
| 4 | bf16 | 200.912 | 189.536 | 379.552 | `sglang-tp/bf16/tp4-up-down/nsys/qwen32b_tp4_bf16_up_down_bs1.json` |
| 4 | marlin | 95.712 | 91.584 | 223.808 | `sglang-tp/marlin/tp4-up-down/nsys/qwen32b_tp4_marlin_up_down_bs1.json` |
| 4 | qlora | 163.216 | 152.288 | 1318.656 | `sglang-tp/qlora/tp4-up-down/nsys/qwen32b_tp4_qlora_up_down_bs1.json` |

Each `source JSON` sits beside the matching `.nsys-rep`, `.sqlite`, and
`_peek.png` Nsight files. TP1 QLoRA's peek image is anchored on Marlin because
there is no NCCL kernel in a single-rank run; TP2/TP4 QLoRA peek images are
anchored on NCCL.

## Actual LoRA wrapper up/down profile

Generated through the real SGLang LoRA wrapper path. TP1 was refreshed on
2026-06-10 after fixing the no-reduce row-parallel flipped path so Marlin is
launched on the side stream between LoRA A and LoRA B instead of after the full
LoRA patch.

| TP | median us | min us | max us | source JSON |
|---:|---:|---:|---:|---|
| 1 | 204.352 | 201.664 | 258.368 | `actual-lora/up_down/tp1/nsys/qwen32b_tp1_actual_qlora_up_down_bs1.json` |
| 2 | 230.848 | 213.472 | 281.152 | `actual-lora/up_down/tp2/nsys/qwen32b_tp2_actual_qlora_up_down_bs1.json` |
| 4 | 202.560 | 179.040 | 273.440 | `actual-lora/up_down/tp4/nsys/qwen32b_tp4_actual_qlora_up_down_bs1.json` |

The refreshed TP1 Nsight peek is
`actual-lora/up_down/tp1/nsys/qwen32b_tp1_actual_qlora_up_down_bs1_peek.png`.
The selected replay shows Marlin on side streams overlapping main-stream LoRA
kernels.
