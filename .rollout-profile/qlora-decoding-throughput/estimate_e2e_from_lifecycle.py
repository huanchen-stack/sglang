#!/usr/bin/env python3
"""Estimate rollout E2E gain from request-lifetime traces and frontier speedups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_FRONTIER = OUT_DIR / "frontier_decoding_throughput.json"
DEFAULT_LIFECYCLE_ROOT = Path("/data/huanchen/sglang/rollout_precision_data/lifecycle")
DEFAULT_BATCHES = [128, 256, 512]


def parse_lifecycle_override(raw: str) -> tuple[int, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected BATCH=/path/to/lifecycle.json")
    batch, path = raw.split("=", 1)
    return int(batch), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument(
        "--lifecycle-json",
        type=parse_lifecycle_override,
        action="append",
        default=[],
        help="Override one trace path as BATCH=/path/to/lifecycle.json.",
    )
    parser.add_argument(
        "--scaled-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of lifecycle drain time assumed to scale with decode "
            "frontier speedup. Use less than 1.0 to model KV/cache or scheduler "
            "time that does not improve."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["bin", "scalar"],
        default="bin",
        help=(
            "bin rescales each lifecycle live-request duration bin by the "
            "frontier speedup for that active batch size. scalar rescales the "
            "whole drain window by the initial batch speedup."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUT_DIR / "deepseek_r1_7b_e2e_gain",
    )
    return parser.parse_args()


def load_frontier_speedups(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    return {int(row["batch_size"]): row for row in rows}


def default_lifecycle_path(root: Path, batch_size: int) -> Path:
    return root / f"r1_7b_reasoning_bs{batch_size}_cap32k" / "lifecycle.json"


def parse_live_request_bin(label: str) -> tuple[int, int]:
    hi, lo = label.split("-", 1)
    return int(lo), int(hi)


def nearest_frontier_speedup(frontier: dict[int, dict], active_batch_size: int) -> tuple[int, float]:
    if active_batch_size in frontier:
        key = active_batch_size
    else:
        key = min(frontier, key=lambda candidate: abs(candidate - active_batch_size))
    return key, float(frontier[key]["frontier_speedup_vs_bf16"])


def estimate_from_bins(args: argparse.Namespace, frontier: dict[int, dict], trace: dict) -> tuple[float, list[dict]]:
    labels = trace.get("bin_labels") or []
    durations = trace.get("bin_durations_s") or []
    if not labels or not durations or len(labels) != len(durations):
        raise SystemExit("lifecycle trace is missing matching bin_labels/bin_durations_s")

    estimated_drain_s = 0.0
    segment_rows = []
    for label, duration in zip(labels, durations):
        lo, hi = parse_live_request_bin(label)
        frontier_batch_size, speedup = nearest_frontier_speedup(frontier, hi)
        duration = float(duration)
        scale = (1.0 - args.scaled_fraction) + args.scaled_fraction / speedup
        estimated_segment_s = duration * scale
        estimated_drain_s += estimated_segment_s
        segment_rows.append(
            {
                "live_request_bin": label,
                "live_request_lo": lo,
                "live_request_hi": hi,
                "frontier_batch_size": frontier_batch_size,
                "frontier_speedup_vs_bf16": speedup,
                "baseline_segment_s": duration,
                "estimated_segment_s": estimated_segment_s,
                "estimated_saved_s": duration - estimated_segment_s,
            }
        )
    return estimated_drain_s, segment_rows


def load_lifecycle_inputs(args: argparse.Namespace) -> dict[int, tuple[Path, dict]]:
    batches = args.batch_size or DEFAULT_BATCHES
    overrides = dict(args.lifecycle_json)
    missing = []
    loaded = {}
    for batch_size in batches:
        path = overrides.get(batch_size, default_lifecycle_path(args.lifecycle_root, batch_size))
        if not path.exists():
            missing.append(path)
            continue
        loaded[batch_size] = (path, json.loads(path.read_text(encoding="utf-8")))

    if missing:
        searched = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Missing DeepSeek R1 7B lifecycle JSONs.\n"
            "The rollout/exploring scripts expect these generated files, but "
            "they are not tracked in git.\n"
            f"Searched:\n{searched}"
        )
    return loaded


def build_rows(args: argparse.Namespace) -> list[dict]:
    if not 0.0 <= args.scaled_fraction <= 1.0:
        raise SystemExit("--scaled-fraction must be between 0 and 1")

    frontier = load_frontier_speedups(args.frontier_json)
    lifecycle = load_lifecycle_inputs(args)
    rows = []
    for batch_size, (path, trace) in sorted(lifecycle.items()):
        if batch_size not in frontier:
            raise SystemExit(f"missing frontier speedup for batch size {batch_size}")
        speedup = float(frontier[batch_size]["frontier_speedup_vs_bf16"])
        drain_s = float(trace["total_drain_s"])
        segment_rows = []
        if args.method == "bin":
            estimated_drain_s, segment_rows = estimate_from_bins(args, frontier, trace)
        else:
            scale = (1.0 - args.scaled_fraction) + args.scaled_fraction / speedup
            estimated_drain_s = drain_s * scale
        n = int(trace["n"])
        output_len_mean = float(trace.get("output_len_mean", 0.0))
        estimated_total_output_tokens = n * output_len_mean if output_len_mean else None

        rows.append(
            {
                "batch_size": batch_size,
                "n_requests": n,
                "lifecycle_json": str(path),
                "baseline_drain_s": drain_s,
                "frontier_speedup_vs_bf16": speedup,
                "estimation_method": args.method,
                "scaled_fraction": args.scaled_fraction,
                "estimated_drain_s": estimated_drain_s,
                "estimated_saved_s": drain_s - estimated_drain_s,
                "estimated_e2e_throughput_gain": drain_s / estimated_drain_s,
                "baseline_requests_per_s": n / drain_s,
                "estimated_requests_per_s": n / estimated_drain_s,
                "output_len_p50": trace.get("output_len_p50"),
                "output_len_p99": trace.get("output_len_p99"),
                "output_len_max": trace.get("output_len_max"),
                "estimated_total_output_tokens": estimated_total_output_tokens,
                "finish_reasons": json.dumps(trace.get("finish_reasons", {}), sort_keys=True),
                "segments": segment_rows,
            }
        )
    return rows


def write_outputs(rows: list[dict], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{k: v for k, v in row.items() if k != "segments"} for row in rows]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {"metadata": {"source": "frontier speedups applied to lifecycle drain time"}, "rows": rows}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| Batch | baseline drain s | frontier speedup | estimated drain s | E2E throughput gain | baseline req/s | estimated req/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {batch_size} | {baseline_drain_s:.1f} | {frontier_speedup_vs_bf16:.2f}x | "
            "{estimated_drain_s:.1f} | {estimated_e2e_throughput_gain:.2f}x | "
            "{baseline_requests_per_s:.4f} | {estimated_requests_per_s:.4f} |".format(**row)
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot(rows, prefix.with_suffix(".png"))


def plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.markersize": 4.5,
        }
    )
    nrows = len(rows)
    fig_height = max(3.1, 2.25 * nrows + 0.55)
    fig, axes = plt.subplots(nrows, 1, figsize=(6.8, fig_height), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        segments = row.get("segments") or []
        labels = [segment["live_request_bin"] for segment in segments]
        baseline = np.array([segment["baseline_segment_s"] for segment in segments], dtype=float)
        estimated = np.array([segment["estimated_segment_s"] for segment in segments], dtype=float)
        xs = np.arange(len(labels))
        width = 0.38

        ax.bar(
            xs - width / 2,
            baseline,
            width=width,
            color="#55A868",
            label="Measured BF16 merged rollout",
        )
        ax.bar(
            xs + width / 2,
            estimated,
            width=width,
            color="#C44E52",
            label="Estimated frontier rollout",
        )

        for x, base, est in zip(xs, baseline, estimated):
            if est <= 0:
                continue
            speedup = base / est
            ax.text(
                x + width / 2,
                est,
                f"x{speedup:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#7A1F2B",
            )

        total_speedup = row["estimated_e2e_throughput_gain"]
        ax.text(
            0.99,
            0.92,
            f"overall x{total_speedup:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB"},
        )
        ax.set_title(f"Batch {row['batch_size']}: Lifecycle-Bin E2E Estimate")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel("Duration (s)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    axes[-1].set_xlabel("Live-request bin")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Estimated DeepSeek R1 7B Rollout E2E Gain From Lifecycle Bars", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Estimated speedups rescale decode lifecycle bars by measured frontier ratios; KV-cache effects are not accurately modeled.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_outputs(rows, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
