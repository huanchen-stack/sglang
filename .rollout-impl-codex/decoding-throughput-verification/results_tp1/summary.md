| Batch | Dynamic tok/s | vs BF16 profile | vs Torch2S profile | vs frontier | Frontier label |
|---:|---:|---:|---:|---:|---|
| 128 | 4257.4 | x0.94 | x0.87 | x0.87 | Q(down) |
| 256 | 5892.1 | x1.01 | x1.09 | x1.01 | Q(null) |
| 512 | 6427.0 | x0.99 | x1.13 | x0.98 | Q(qkv) |
| 8 | 435.6 | x1.01 | x0.47 | x0.47 | Q(qkv, out, up, down) |
