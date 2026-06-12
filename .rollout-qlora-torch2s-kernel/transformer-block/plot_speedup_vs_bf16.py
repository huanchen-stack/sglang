#!/usr/bin/env python3
"""Plot QLoRA speedup versus BF16 from latency benchmark JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "latency-results"
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
SCOPES = ["block", "attn", "mlp"]
SETUPS = [
    ("qwen2.5-32b", "tp1", "Qwen2.5 32B, TP1"),
    ("qwen2.5-32b", "tp4", "Qwen2.5 32B, TP4"),
    ("deepseek-r1-distill-qwen-7b", "tp1", "DeepSeek-R1-Distill-Qwen-7B, TP1"),
]


def load_median_us(model: str, tp: str, batch_size: int, precision: str, scope: str) -> float:
    path = RESULTS_ROOT / model / tp / "kv1024" / f"bs{batch_size}" / precision / f"{scope}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["median_us"])


def speedup_series(model: str, tp: str, scope: str) -> list[float]:
    values = []
    for batch_size in BATCH_SIZES:
        bf16 = load_median_us(model, tp, batch_size, "bf16", scope)
        qlora = load_median_us(model, tp, batch_size, "qlora", scope)
        values.append(bf16 / qlora)
    return values


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    colors = {"block": "#2563eb", "attn": "#d97706", "mlp": "#059669"}
    linestyles = {"block": "-", "attn": "--", "mlp": "--"}
    labels = {"block": "Block QLoRA/BF16", "attn": "Attn QLoRA/BF16", "mlp": "MLP QLoRA/BF16"}

    for ax, (model, tp, title) in zip(axes, SETUPS, strict=True):
        for scope in SCOPES:
            ax.plot(
                BATCH_SIZES,
                speedup_series(model, tp, scope),
                label=labels[scope],
                color=colors[scope],
                linestyle=linestyles[scope],
                linewidth=2.0,
                marker="o",
                markersize=4,
            )

        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES, [str(x) for x in BATCH_SIZES])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speedup vs BF16")
        ax.grid(True, which="major", alpha=0.25)
        ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
        ax.legend(loc="best", frameon=False)

    pdf_path = RESULTS_ROOT / "qlora_speedup_vs_bf16.pdf"
    png_path = RESULTS_ROOT / "qlora_speedup_vs_bf16.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
