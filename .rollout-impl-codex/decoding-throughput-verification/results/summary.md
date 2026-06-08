| Batch | Dynamic tok/s | vs BF16 profile | vs Torch2S profile | vs frontier | Frontier label |
|---:|---:|---:|---:|---:|---|
| 128 | 8080.6 | x1.78 | x1.65 | x1.65 | Q(down) |
| 256 | 13497.9 | x2.30 | x2.50 | x2.30 | Q(null) |
| 512 | 15782.7 | x2.43 | x2.79 | x2.40 | Q(qkv) |
| 8 | 1096.2 | x2.54 | x1.17 | x1.17 | Q(qkv, out, up, down) |
