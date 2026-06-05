"""Grouped bar plot: decode wall tokens/s per batch size for bf16 / int4 / qlora.

bf16 and marlin-int4 data come from the 8gpu sweep dir; qlora data comes from
the mem-0.72 no-qkv rerun (the canonical, crash-free arm).
"""

import argparse
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/data/huanchen/sglang"
BASE = os.path.join(
    REPO, ".codex-reports/lora_forward_eval/decode_throughput_8gpu_bs_sweep_v2_chunk64"
)
QLORA = os.path.join(
    REPO,
    ".codex-reports/lora_forward_eval/decode_throughput_qlora_noqkv_chunk64_mem072",
)
BATCH_SIZES = [8, 16, 32, 64, 128, 512]

METRIC_LABEL = {
    "decode_wall_tokens_per_s": "Decode throughput (wall tokens / s)",
    "decode_forward_tokens_per_s": "Decode throughput (forward tokens / s)",
    "gen_throughput": "Pure-decode output throughput (tokens / s)",
}


def _steady_state_gen_throughput(exp_dir: str):
    """Median of the scheduler's per-step 'gen throughput (token/s)' during
    steady-state decode (drops the first two ramp-up readings)."""
    log = os.path.join(exp_dir, "server.log")
    if not os.path.exists(log):
        return None
    vals = []
    for line in open(log, errors="replace"):
        m = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) <= 3:
        return None
    return statistics.median(vals[2:])


def read_metric(exp_dir: str, metric: str):
    if os.path.exists(os.path.join(exp_dir, "failure.txt")):
        return None
    if metric == "gen_throughput":
        return _steady_state_gen_throughput(exp_dir)
    summary = os.path.join(exp_dir, "decode_summary.json")
    if not os.path.exists(summary):
        return None
    return json.load(open(summary)).get(metric)


def qlora_dir(bs: int):
    # qlora experiment names keep a _gpuN suffix; pick whichever exists.
    cand = os.path.join(QLORA, f"qwen2.5-14b_qlora_parallel_streams_math_bs{bs}")
    if os.path.isdir(cand):
        return cand
    for gpu in range(8):
        d = f"{cand}_gpu{gpu}"
        if os.path.isdir(d):
            return d
    return cand


SCHEMES = {
    "bf16": lambda bs: os.path.join(BASE, f"qwen2.5-14b_bf16_full_math_bs{bs}"),
    "int4 (marlin)": lambda bs: os.path.join(
        BASE, f"qwen2.5-14b_marlin_int4_full_math_bs{bs}"
    ),
    "qlora (int4+BF16 LoRA)": qlora_dir,
}
COLORS = {
    "bf16": "#4C72B0",
    "int4 (marlin)": "#55A868",
    "qlora (int4+BF16 LoRA)": "#C44E52",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        default="gen_throughput",
        choices=list(METRIC_LABEL),
    )
    args = parser.parse_args()
    metric = args.metric

    data = {
        name: [read_metric(path_fn(bs), metric) for bs in BATCH_SIZES]
        for name, path_fn in SCHEMES.items()
    }

    x = np.arange(len(BATCH_SIZES))
    n = len(SCHEMES)
    width = 0.8 / n

    # bf16 is the baseline for the relative (xN) labels.
    baseline = data["bf16"]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (name, vals) in enumerate(data.items()):
        offset = (i - (n - 1) / 2) * width
        plotted = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + offset, plotted, width, label=name, color=COLORS[name])
        for rect, v, base in zip(bars, vals, baseline):
            if v is None:
                label = "NA"
            elif base:
                label = f"{v / base:.2f}×"
            else:
                label = ""
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                (v if v else 0) + 40,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="gray" if v is None else "black",
            )

    ax.set_xlabel("Batch size (max concurrency)", fontsize=11)
    ax.set_ylabel("Pure-decode output tok/s", fontsize=11)
    ax.set_title("Qwen2.5-14B pure-decode throughput", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in BATCH_SIZES])
    ax.tick_params(labelsize=10)
    ax.margins(y=0.12)
    ax.legend(title="Scheme", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    suffix = {
        "decode_wall_tokens_per_s": "wall",
        "decode_forward_tokens_per_s": "forward",
        "gen_throughput": "gen",
    }[metric]
    out = os.path.join(
        REPO,
        f"lora_forward_eval_data/decode_throughput_by_scheme_{suffix}.png",
    )
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}\n")
    header = f"{'bs':>5} | " + " | ".join(f"{n:>22}" for n in SCHEMES)
    print(header)
    for j, bs in enumerate(BATCH_SIZES):
        cells = [
            (f"{data[name][j]:.1f}" if data[name][j] is not None else "NA")
            for name in SCHEMES
        ]
        print(f"{bs:>5} | " + " | ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
