"""Plot LoRA forward evaluation results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = {
    "bf16_full": "BF16 full",
    "marlin_int4_full": "Marlin INT4",
    "bf16_lora_patch": "BF16 LoRA patch",
    "qlora_marlin_bf16": "INT4+LoRA",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_kernel(kernel_jsonl: Path, output_dir: Path) -> None:
    rows = [row for row in read_jsonl(kernel_jsonl) if row.get("error") is None]
    if not rows:
        return
    by_key = {(r["model"], r["projection"], r["provider"]): r for r in rows}
    keys = sorted({(r["model"], r["projection"]) for r in rows})

    labels = [f"{m.replace('qwen2.5-', '')}\n{p}" for m, p in keys]
    x = np.arange(len(keys))
    width = 0.25

    bf16 = np.array([by_key[(m, p, "bf16_full")]["median_us"] for m, p in keys])
    int4 = np.array([by_key[(m, p, "marlin_int4_full")]["median_us"] for m, p in keys])
    lora = np.array([by_key[(m, p, "bf16_lora_patch")]["median_us"] for m, p in keys])

    fig, ax = plt.subplots(figsize=(max(11, len(keys) * 1.2), 5.8))
    b1 = ax.bar(x - width, bf16, width, label="BF16 full", color="#4C78A8")
    b2 = ax.bar(x, int4, width, label="Marlin INT4 full", color="#59A14F")
    ax.bar(x + width, int4, width, label="QLoRA INT4 base", color="#9CD17B")
    ax.bar(
        x + width,
        lora,
        width,
        bottom=int4,
        label="QLoRA bf16 LoRA patch",
        color="#F28E2B",
    )
    for bars, values in ((b1, bf16), (b2, int4)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    for idx, total in enumerate(int4 + lora):
        ax.text(x[idx] + width, total, f"{total:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title("Kernel-Level Forward Time")
    ax.set_ylabel("CUDA Graph replay time with L2 flush (us)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "01_kernel_level_bar.png", dpi=180)
    fig.savefig(output_dir / "01_kernel_level_bar.pdf")
    plt.close(fig)


def plot_kernel_sweep(sweep_jsonl: Path, output_dir: Path) -> None:
    """Line plots of latency vs batch size (tokens), one figure per model.

    If the JSONL contains both cold (hot_activation=False) and hot
    (hot_activation=True) rows, both are overlaid: solid = cold, dashed = hot.
    """
    rows = [r for r in read_jsonl(sweep_jsonl) if r.get("error") is None]
    if not rows:
        return

    provider_colors = {
        "bf16_full": "#4C78A8",
        "marlin_int4_full": "#59A14F",
        "bf16_lora_patch": "#F28E2B",
        "qlora_marlin_bf16": "#9467BD",
        "qlora_parallel_streams": "#17BECF",
    }
    provider_labels = {
        "bf16_full": "BF16 full",
        "marlin_int4_full": "Marlin INT4",
        "bf16_lora_patch": "BF16 LoRA patch",
        "qlora_marlin_bf16": "INT4 + LoRA patch (serial)",
        "qlora_parallel_streams": "INT4 + LoRA patch (parallel streams)",
    }
    projections = ["qkv", "out", "gate_up", "down"]
    providers = list(provider_colors)

    # Detect which activation modes are present
    hot_modes = sorted({r.get("hot_activation", False) for r in rows})
    linestyles = {False: "-", True: "--"}
    markers    = {False: "o", True: "s"}

    # (model, projection, provider, tokens, hot) -> row
    by_key: dict[tuple, dict] = {}
    for r in rows:
        hot = r.get("hot_activation", False)
        by_key[(r["model"], r["projection"], r["provider"], r["tokens"], hot)] = r

    models = sorted({r["model"] for r in rows})
    all_tokens = sorted({r["tokens"] for r in rows})

    for model in models:
        fig, axes = plt.subplots(1, len(projections), figsize=(5.5 * len(projections), 4.5))
        for ax, proj in zip(axes, projections):
            for hot in hot_modes:
                ls = linestyles[hot]
                mk = markers[hot]
                suffix = " (hot act)" if hot else " (cold)"
                for provider in providers:
                    xs = [t for t in all_tokens if (model, proj, provider, t, hot) in by_key]
                    if not xs:
                        continue
                    ys = [by_key[(model, proj, provider, t, hot)]["median_us"] for t in xs]
                    lo = [by_key[(model, proj, provider, t, hot)]["p20_us"] for t in xs]
                    hi = [by_key[(model, proj, provider, t, hot)]["p80_us"] for t in xs]
                    color = provider_colors[provider]
                    label = provider_labels[provider] + (suffix if len(hot_modes) > 1 else "")
                    ax.plot(xs, ys, mk + ls, label=label, color=color)
                    ax.fill_between(xs, lo, hi, alpha=0.12, color=color)
            ax.set_title(proj)
            ax.set_xlabel("Batch size (tokens)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(all_tokens)
            ax.set_xticklabels(all_tokens, fontsize=7)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Latency (µs)")
            ax.grid(alpha=0.25)
        axes[-1].legend(fontsize=7)
        hot_tag = " — cold=solid, hot=dashed" if len(hot_modes) > 1 else ""
        fig.suptitle(f"{model} — Kernel Latency vs Batch Size{hot_tag}")
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = model.replace(".", "_").replace("-", "_")
        fig.savefig(output_dir / f"01_kernel_sweep_{slug}.png", dpi=180)
        fig.savefig(output_dir / f"01_kernel_sweep_{slug}.pdf")
        plt.close(fig)


def load_rollout_experiments(rollout_dir: Path) -> list[dict[str, Any]]:
    experiments = []
    for exp_dir in sorted(rollout_dir.iterdir() if rollout_dir.exists() else []):
        if not exp_dir.is_dir() or exp_dir.name == "adapters":
            continue
        metadata_path = exp_dir / "metadata.json"
        bench_path = exp_dir / "bench_serving.jsonl"
        if not metadata_path.exists() or not bench_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        bench_rows = read_jsonl(bench_path)
        if not bench_rows:
            continue
        summary = bench_rows[-1]
        if "output_throughput" not in summary:
            for row in reversed(bench_rows):
                if "output_throughput" in row:
                    summary = row
                    break
        metadata["summary"] = summary
        metadata["exp_dir"] = str(exp_dir)
        experiments.append(metadata)
    return experiments


def plot_throughput(rollout_dir: Path, output_dir: Path) -> None:
    exps = load_rollout_experiments(rollout_dir)
    if not exps:
        return
    exps = [e for e in exps if "output_throughput" in e["summary"]]
    if not exps:
        return

    keys = sorted(
        {(e["model_label"], e["dataset_category"], e["batch_size"]) for e in exps}
    )
    modes = ["bf16_full", "marlin_int4_full", "qlora_marlin_bf16"]
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    by_key = {
        (e["model_label"], e["dataset_category"], e["batch_size"], e["mode"]): e
        for e in exps
    }
    labels = [f"{m.replace('qwen2.5-', '')}\n{c}\nB{b}" for m, c, b in keys]
    x = np.arange(len(keys))
    width = 0.23

    fig, ax = plt.subplots(figsize=(max(14, len(keys) * 0.55), 6.2))
    for offset, mode, color in zip([-width, 0, width], modes, colors):
        values = [
            by_key.get((m, c, b, mode), {}).get("summary", {}).get("output_throughput", 0)
            for m, c, b in keys
        ]
        bars = ax.bar(x + offset, values, width, label=MODE_LABELS[mode], color=color)
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
    ax.set_title("End-to-End Output Throughput")
    ax.set_ylabel("Output tokens / second")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "02_e2e_throughput_bar.png", dpi=180)
    fig.savefig(output_dir / "02_e2e_throughput_bar.pdf")
    plt.close(fig)


def request_live_times(exp_dir: Path) -> list[float]:
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    for path in sorted((exp_dir / "traces").glob("*.jsonl")):
        for row in read_jsonl(path):
            rid = row.get("rid")
            if not rid:
                continue
            ts = row.get("ts")
            if ts is None:
                continue
            starts[rid] = min(starts.get(rid, ts), ts)
            ends[rid] = max(ends.get(rid, ts), ts)
    return sorted(ends[rid] - starts[rid] for rid in starts.keys() & ends.keys())


def plot_live_time(rollout_dir: Path, output_dir: Path) -> None:
    exps = load_rollout_experiments(rollout_dir)
    if not exps:
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exp in exps:
        grouped[exp["dataset_category"]].append(exp)
    categories = [c for c in ("math", "agentic", "reasoning") if c in grouped]
    if not categories:
        categories = sorted(grouped)

    colors = {
        "bf16_full": "#4C78A8",
        "marlin_int4_full": "#59A14F",
        "qlora_marlin_bf16": "#F28E2B",
    }
    fig, axes = plt.subplots(1, len(categories), figsize=(5.2 * len(categories), 4.8), sharey=True)
    if len(categories) == 1:
        axes = [axes]
    for ax, category in zip(axes, categories):
        for exp in sorted(grouped[category], key=lambda e: (e["model_label"], e["batch_size"], e["mode"])):
            times = request_live_times(Path(exp["exp_dir"]))
            if not times:
                continue
            x = np.linspace(0, 100, len(times))
            label = f"{exp['model_label'].replace('qwen2.5-', '')} {MODE_LABELS[exp['mode']]} B{exp['batch_size']}"
            ax.plot(x, times, label=label, color=colors.get(exp["mode"]), alpha=0.85)
        ax.set_title(category)
        ax.set_xlabel("Completed request percentile")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Request live time (s)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Request Live-Time Long Tail")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "03_request_live_time.png", dpi=180)
    fig.savefig(output_dir / "03_request_live_time.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-jsonl")
    parser.add_argument("--sweep-jsonl")
    parser.add_argument("--rollout-dir")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.kernel_jsonl:
        plot_kernel(Path(args.kernel_jsonl), output_dir)
    if args.sweep_jsonl:
        plot_kernel_sweep(Path(args.sweep_jsonl), output_dir)
    if args.rollout_dir:
        rollout_dir = Path(args.rollout_dir)
        plot_throughput(rollout_dir, output_dir)
        plot_live_time(rollout_dir, output_dir)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
