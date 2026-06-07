#!/usr/bin/env python3
"""Create fake deliverables for QLoRA decoding-throughput planning.

The real experiment will replace this synthetic data with server/client
measurements.  The shape of the output is the important part here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
MODEL = "qwen2.5-32b"
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
PROJECTIONS = ["qkv", "out", "gate_up", "down"]


def fake_rows() -> list[dict]:
    # Synthetic but monotonic-ish values. Decode throughput excludes prefill.
    bf16_tok_s = {
        1: 118,
        4: 442,
        8: 842,
        16: 1510,
        32: 2575,
        64: 4070,
        128: 5700,
        256: 6880,
        512: 7450,
    }
    qlora_tok_s = {
        1: 178,
        4: 646,
        8: 1195,
        16: 2035,
        32: 3280,
        64: 4740,
        128: 5900,
        256: 6500,
        512: 6650,
    }
    frontier_tok_s = {
        1: 185,
        4: 682,
        8: 1280,
        16: 2210,
        32: 3560,
        64: 5200,
        128: 6620,
        256: 7420,
        512: 8060,
    }
    decode_ms = {
        scheme: {
            bs: (bs * 256 * 1000.0) / tok_s
            for bs, tok_s in values.items()
        }
        for scheme, values in {
            "bf16_merged": bf16_tok_s,
            "qlora_torch_twostream": qlora_tok_s,
            "selective_frontier": frontier_tok_s,
        }.items()
    }

    policy = {
        1: {"qkv": "qlora", "out": "qlora", "gate_up": "qlora", "down": "qlora"},
        4: {"qkv": "qlora", "out": "qlora", "gate_up": "qlora", "down": "qlora"},
        8: {"qkv": "qlora", "out": "qlora", "gate_up": "qlora", "down": "qlora"},
        16: {"qkv": "qlora", "out": "qlora", "gate_up": "qlora", "down": "qlora"},
        32: {"qkv": "bf16", "out": "qlora", "gate_up": "qlora", "down": "qlora"},
        64: {"qkv": "bf16", "out": "bf16", "gate_up": "qlora", "down": "qlora"},
        128: {"qkv": "bf16", "out": "bf16", "gate_up": "qlora", "down": "qlora"},
        256: {"qkv": "bf16", "out": "bf16", "gate_up": "qlora", "down": "bf16"},
        512: {"qkv": "bf16", "out": "bf16", "gate_up": "qlora", "down": "bf16"},
    }

    rows = []
    for bs in BATCH_SIZES:
        rows.append(
            {
                "batch_size": bs,
                "model": MODEL,
                "forced_decode_tokens_per_request": 256,
                "prefill_excluded": True,
                "bf16_merged_tok_s": bf16_tok_s[bs],
                "qlora_torch_twostream_tok_s": qlora_tok_s[bs],
                "selective_frontier_tok_s": frontier_tok_s[bs],
                "bf16_merged_decode_ms": round(decode_ms["bf16_merged"][bs], 3),
                "qlora_torch_twostream_decode_ms": round(
                    decode_ms["qlora_torch_twostream"][bs], 3
                ),
                "selective_frontier_decode_ms": round(
                    decode_ms["selective_frontier"][bs], 3
                ),
                "qlora_speedup_vs_bf16": round(qlora_tok_s[bs] / bf16_tok_s[bs], 3),
                "frontier_speedup_vs_bf16": round(
                    frontier_tok_s[bs] / bf16_tok_s[bs], 3
                ),
                **{f"select_{proj}": policy[bs][proj] for proj in PROJECTIONS},
            }
        )
    return rows


def write_data(rows: list[dict]) -> None:
    payload = {
        "metadata": {
            "fake": True,
            "purpose": "Demonstrate intended QLoRA decoding-throughput deliverable shape before implementation.",
            "decode_tokens_per_request": 256,
            "model": MODEL,
            "prefill_excluded": True,
            "batch_sizes": BATCH_SIZES,
            "schemes": [
                "bf16_merged",
                "qlora_torch_twostream",
                "selective_frontier",
            ],
            "projection_choices": PROJECTIONS,
        },
        "rows": rows,
    }
    (OUT_DIR / "fake_decoding_throughput.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with (OUT_DIR / "fake_decoding_throughput.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table_lines = [
        "| Batch | BF16 merged tok/s | QLoRA two-stream tok/s | Selective frontier tok/s | QLoRA speedup | Frontier speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {batch_size} | {bf16_merged_tok_s:.0f} | {qlora_torch_twostream_tok_s:.0f} | "
            "{selective_frontier_tok_s:.0f} | {qlora_speedup_vs_bf16:.2f}x | "
            "{frontier_speedup_vs_bf16:.2f}x |".format(**row)
        )
    (OUT_DIR / "fake_summary_table.md").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )


def plot_throughput(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    xs = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(xs, [r["bf16_merged_tok_s"] for r in rows], marker="o", linewidth=2.4, label="BF16 merged rollout")
    ax.plot(xs, [r["qlora_torch_twostream_tok_s"] for r in rows], marker="s", linewidth=2.4, label="QLoRA all projections, Torch two-stream LoRA")
    ax.plot(xs, [r["selective_frontier_tok_s"] for r in rows], marker="D", linewidth=2.8, label="Selective per-projection frontier")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_ylabel("Decode throughput (tokens/s), prefill excluded")
    ax.set_title(f"Fake QLoRA Rollout Decoding Throughput ({MODEL})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fake_throughput.png", dpi=180)
    plt.close(fig)


def plot_speedup(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    xs = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axhline(1.0, color="#333333", linewidth=1.1, alpha=0.7)
    ax.plot(xs, [r["qlora_speedup_vs_bf16"] for r in rows], marker="s", linewidth=2.4, label="QLoRA all projections / BF16")
    ax.plot(xs, [r["frontier_speedup_vs_bf16"] for r in rows], marker="D", linewidth=2.8, label="Selective frontier / BF16")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_ylabel("Speedup over BF16 merged rollout")
    ax.set_title(f"Fake Decode Throughput Speedup ({MODEL})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fake_speedup.png", dpi=180)
    plt.close(fig)


def plot_policy(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(
        [[1 if row[f"select_{proj}"] == "qlora" else 0 for row in rows] for proj in PROJECTIONS]
    )
    fig, ax = plt.subplots(figsize=(9.8, 3.7))
    ax.imshow(matrix, aspect="auto", cmap=plt.cm.get_cmap("RdYlBu_r", 2), vmin=0, vmax=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(r["batch_size"]) for r in rows])
    ax.set_yticks(range(len(PROJECTIONS)))
    ax.set_yticklabels(PROJECTIONS)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_title("Fake Selective Frontier Projection Policy")
    for y, proj in enumerate(PROJECTIONS):
        for x, row in enumerate(rows):
            label = "QLoRA" if row[f"select_{proj}"] == "qlora" else "BF16"
            ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#111111")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fake_projection_policy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fake_rows()
    write_data(rows)
    plot_throughput(rows)
    plot_speedup(rows)
    plot_policy(rows)
    print(f"Wrote fake decoding-throughput deliverables to {OUT_DIR}")


if __name__ == "__main__":
    main()
