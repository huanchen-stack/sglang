#!/usr/bin/env python3
"""Create synthetic rollout precision-switch deliverables.

These plots are intentionally fake. They encode the expected deliverable shape
before the implementation and measurement pipeline exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def add_rollout_end_marker(ax, end_s: float, color: str = "#111111", label: str = "rollout end") -> None:
    ax.axvline(end_s, color=color, linestyle=":", linewidth=1.1)
    ax.scatter([end_s], [0], marker="D", s=24, color=color, edgecolor="white", linewidth=0.6, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(
        end_s,
        ymin + (ymax - ymin) * 0.08,
        label,
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=7,
        color=color,
    )


def live_requests_at_time(t: np.ndarray, batch: int = 512) -> np.ndarray:
    # Synthetic long-tail shape: fast early drain, slow final tail.
    return np.maximum(1, np.ceil(batch * np.exp(-t / 95.0) + 3.5 * np.exp(-t / 620.0))).astype(int)


def make_policy_segments() -> list[dict]:
    return [
        {
            "name": "BF16 merged",
            "batch_start": 512,
            "batch_end": 128,
            "t_start": 0,
            "t_end": 118,
            "color": "#4C72B0",
            "qkv": "bf16",
            "o": "bf16",
            "up": "bf16",
            "down": "bf16",
            "q_short": "B,B,B,B",
            "q_label": "Q(null)",
            "speedup_vs_bf16": 1.00,
            "speedup_vs_csgmv": 0.86,
        },
        {
            "name": "Hybrid mixed",
            "batch_start": 128,
            "batch_end": 64,
            "t_start": 118,
            "t_end": 192,
            "color": "#8172B2",
            "qkv": "bf16",
            "o": "bf16",
            "up": "int4+torch2s",
            "down": "int4+torch2s",
            "q_short": "B,B,I,I",
            "q_label": "Q(up, down)",
            "speedup_vs_bf16": 1.38,
            "speedup_vs_csgmv": 1.92,
        },
        {
            "name": "QLoRA selective",
            "batch_start": 64,
            "batch_end": 32,
            "t_start": 192,
            "t_end": 286,
            "color": "#DD8452",
            "qkv": "int4+torch2s",
            "o": "bf16",
            "up": "int4+torch2s",
            "down": "int4+torch2s",
            "q_short": "I,B,I,I",
            "q_label": "Q(qkv, up, down)",
            "speedup_vs_bf16": 2.30,
            "speedup_vs_csgmv": 3.60,
        },
        {
            "name": "QLoRA tail",
            "batch_start": 32,
            "batch_end": 1,
            "t_start": 286,
            "t_end": 610,
            "color": "#C44E52",
            "qkv": "int4+torch2s",
            "o": "int4+torch2s",
            "up": "int4+torch2s",
            "down": "int4+torch2s",
            "q_short": "I,I,I,I",
            "q_label": "Q(qkv, out, up, down)",
            "speedup_vs_bf16": 3.15,
            "speedup_vs_csgmv": 4.05,
        },
    ]


def write_policy_data(segments: list[dict]) -> None:
    serializable = [{k: v for k, v in row.items() if k != "color"} for row in segments]
    (OUT_DIR / "fake_precision_policy.json").write_text(
        json.dumps({"model": "Qwen2.5-14B-Instruct", "dataset": "Eurus-2-RL", "segments": serializable}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with (OUT_DIR / "fake_precision_policy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def plot_lifetime_policy(segments: list[dict]) -> None:
    t = np.linspace(0, 610, 900)
    live_bf16 = live_requests_at_time(t)
    mean_decoded = 220 + 0.9 * t + 0.0035 * t**2
    mean_decoded[live_bf16 > 128] *= 0.55

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )

    label_y = [468, 400, 332, 264]
    for idx, seg in enumerate(segments):
        for ax in axes:
            ax.axvspan(seg["t_start"], seg["t_end"], color=seg["color"], alpha=0.14, linewidth=0)
        mid = (seg["t_start"] + seg["t_end"]) / 2
        label = (
            f"{seg['q_label']}\n"
            f"x{seg['speedup_vs_bf16']:.2f} / x{seg['speedup_vs_csgmv']:.2f}"
        )
        axes[0].text(
            mid,
            label_y[idx],
            label,
            ha="center",
            va="top",
            fontsize=7.4,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": seg["color"], "linewidth": 0.8},
        )

    axes[0].plot(t, live_bf16, color="#222222", linewidth=2.0, label="BF16 merged baseline lifetime")
    axes[0].set_ylabel("Live requests")
    axes[0].set_ylim(0, 540)
    axes[0].grid(True, alpha=0.25, linewidth=0.6)
    axes[0].legend(loc="lower left", frameon=False)
    add_rollout_end_marker(axes[0], float(t[-1]))

    axes[1].plot(t, mean_decoded, color="#55A868", linewidth=2.0)
    axes[1].set_xlabel("Decode time on BF16 baseline clock (s)")
    axes[1].set_ylabel("Mean decoded length of live requests")
    axes[1].set_ylim(0, max(mean_decoded) * 1.08)
    axes[1].grid(True, alpha=0.25, linewidth=0.6)
    axes[1].axvline(float(t[-1]), color="#111111", linestyle=":", linewidth=1.1)

    for seg in segments[1:]:
        axes[0].axvline(seg["t_start"], color=seg["color"], linestyle="--", linewidth=1.0)
        axes[0].text(seg["t_start"], 18, f"bs {seg['batch_start']}", rotation=90, fontsize=7, va="bottom", ha="right")

    fig.suptitle(
        "Fake Dynamic Precision Rollout Policy, Qwen2.5-14B on Eurus-2-RL\n"
        "Q(...) lists INT4+BF16 LoRA torch2s projections; x=vs BF16 / vs csgmv",
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "fake_precision_lifetime.png", dpi=300)
    fig.savefig(OUT_DIR / "fake_precision_lifetime.pdf")
    plt.close(fig)


def plot_speedup_summary() -> None:
    batch_bins = ["512-128", "128-64", "64-32", "32-1"]
    x = np.arange(len(batch_bins))
    width = 0.22
    tp1 = np.array([1.00, 1.21, 2.05, 2.78])
    tp4 = np.array([1.00, 1.38, 2.30, 3.15])
    csgmv_tp4 = np.array([0.86, 1.92, 3.60, 4.05])

    fig, ax = plt.subplots(figsize=(6.4, 3.15), constrained_layout=True)
    ax.bar(x - width, tp1, width, label="Our policy vs BF16, TP1", color="#4C72B0")
    ax.bar(x, tp4, width, label="Our policy vs BF16, TP4", color="#DD8452")
    ax.bar(x + width, csgmv_tp4, width, label="Our policy vs int4+csgmv, TP4", color="#C44E52")
    ax.axhline(1.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_bins)
    ax.set_ylabel("Window speedup")
    ax.set_xlabel("Live-request window")
    ax.set_ylim(0, 4.6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="x%.2f", fontsize=7, padding=2)
    fig.suptitle("Fake Window Speedup from Dynamic Precision Switching", fontsize=11)
    fig.savefig(OUT_DIR / "fake_speedup_summary.png", dpi=300)
    fig.savefig(OUT_DIR / "fake_speedup_summary.pdf")
    plt.close(fig)


def plot_vram_timeline(segments: list[dict]) -> None:
    t = np.linspace(0, 610, 500)
    bf16_only = np.full_like(t, 63.0)
    dual_resident = np.full_like(t, 75.0)
    offload_future = np.piecewise(
        t,
        [t < 128, (t >= 128) & (t < 260), t >= 260],
        [75.0, lambda x: 75.0 - (x - 128) * 0.055, 67.7],
    )
    kv = 2.0 + 18.0 * live_requests_at_time(t) / 512.0

    fig, ax = plt.subplots(figsize=(6.4, 3.25), constrained_layout=True)
    for seg in segments:
        ax.axvspan(seg["t_start"], seg["t_end"], color=seg["color"], alpha=0.10, linewidth=0)
    ax.plot(t, bf16_only + kv, label="BF16 merged + KV", color="#4C72B0", linewidth=2.0)
    ax.plot(t, dual_resident + kv, label="BF16 + int4 resident + KV", color="#C44E52", linewidth=2.0)
    ax.plot(t, offload_future + kv, label="Future gradual offload sketch", color="#55A868", linewidth=2.0, linestyle="--")
    ax.set_xlabel("Decode time on BF16 baseline clock (s)")
    ax.set_ylabel("Approx VRAM per GPU (GB)")
    ax.set_ylim(55, 98)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Fake VRAM Cost of Dual-Precision Rollout", fontsize=11)
    fig.savefig(OUT_DIR / "fake_vram_timeline.png", dpi=300)
    fig.savefig(OUT_DIR / "fake_vram_timeline.pdf")
    plt.close(fig)


def main() -> None:
    segments = make_policy_segments()
    write_policy_data(segments)
    plot_lifetime_policy(segments)
    plot_speedup_summary()
    plot_vram_timeline(segments)


if __name__ == "__main__":
    main()
