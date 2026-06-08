#!/usr/bin/env python3
"""Create rollout precision deliverables from measured traces."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FRONTIER_JSON = ROOT / ".rollout-profile/qlora-decoding-throughput/frontier_decoding_throughput.json"
DECODING_CSV = ROOT / ".rollout-profile/qlora-decoding-throughput/decoding_throughput.csv"
LIFECYCLE_ROOT = ROOT / ".rollout-profile/request-lifetime-cot/results"
LIVE_DYNAMIC_ROOT = (
    OUT_DIR / "live-runs/dynamic-qwen2.5-14b-eurus-tp4"
)
RUN_PREFIX = "qwen2.5-14b-instruct-eurus-2-rl"
BATCHES = [128, 256, 512]
TPS = [1, 4]

COLORS = {
    "bf16": "#4C72B0",
    "frontier": "#C44E52",
    "live": "#222222",
    "mean": "#55A868",
    "window": "#8172B2",
}

POLICY_COLORS = {
    "BBB": "#B8C0CC",
    "IBB": "#7B6FD0",
    "BBI": "#3BA76D",
    "IBI": "#E28E2C",
    "III": "#D64B3C",
}


@dataclass(frozen=True)
class SegmentEstimate:
    tp: int
    batch_size: int
    live_request_bin: str
    live_request_hi: int
    live_request_lo: int
    frontier_batch_size: int
    frontier_speedup_vs_bf16: float
    frontier_speedup_vs_csgmv: float
    baseline_segment_s: float
    estimated_segment_s: float
    policy: str
    policy_short: str
    qlora_projection_label: str

    @property
    def speedup(self) -> float:
        if self.estimated_segment_s <= 0:
            return 1.0
        return self.baseline_segment_s / self.estimated_segment_s

    @property
    def csgmv_segment_s(self) -> float:
        return self.estimated_segment_s * self.frontier_speedup_vs_csgmv


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.9,
            "ytick.labelsize": 7.9,
            "legend.fontsize": 7.9,
            "figure.titlesize": 9.8,
            "lines.linewidth": 1.8,
            "axes.linewidth": 0.8,
        }
    )


def add_rollout_end_marker(
    ax,
    end_s: float,
    color: str = "#111111",
    label: str = "rollout end",
    y: float = 0.0,
) -> None:
    ax.axvline(end_s, color=color, linestyle=":", linewidth=1.1)
    ax.scatter(
        [end_s],
        [y],
        marker="D",
        s=24,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ymin, ymax = ax.get_ylim()
    ax.text(
        end_s,
        ymin + (ymax - ymin) * 0.08,
        label,
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=6.9,
        color=color,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frontier() -> dict[int, dict]:
    payload = load_json(FRONTIER_JSON)
    rows = payload["rows"]
    frontier = {int(row["batch_size"]): row for row in rows}
    csgmv = load_decoding_throughput("qlora_csgmv")
    for batch_size, row in frontier.items():
        csgmv_tok_s = csgmv.get(batch_size)
        if csgmv_tok_s and csgmv_tok_s > 0:
            row["frontier_speedup_vs_csgmv"] = float(row["kernel_guided_frontier_tok_s"]) / csgmv_tok_s
        else:
            row["frontier_speedup_vs_csgmv"] = 1.0
    return frontier


def load_decoding_throughput(scheme: str) -> dict[int, float]:
    values = {}
    with DECODING_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scheme"] != scheme:
                continue
            values[int(row["batch_size"])] = float(row["decode_tok_s"])
    return values


def nearest_frontier(frontier: dict[int, dict], active_batch_size: int) -> tuple[int, dict]:
    key = min(frontier, key=lambda candidate: abs(candidate - active_batch_size))
    return key, frontier[key]


def parse_live_request_bin(label: str) -> tuple[int, int]:
    hi, lo = label.split("-", 1)
    return int(hi), int(lo)


def policy_label(row: dict) -> str:
    choices = [row[f"select_{name}"] for name in ("qkv", "o", "gate_up", "down")]
    short = ["I" if choice == "qlora" else "B" for choice in choices]
    return f"({','.join(short)})"


def compact_policy_label(row: dict) -> str:
    choices = [row[f"select_{name}"] for name in ("qkv", "gate_up", "down")]
    return "".join("I" if choice == "qlora" else "B" for choice in choices)


def qlora_projection_label(row: dict) -> str:
    selected = []
    for key, label in (
        ("qkv", "qkv"),
        ("o", "out"),
        ("gate_up", "up"),
        ("down", "down"),
    ):
        if row[f"select_{key}"] == "qlora":
            selected.append(label)
    return f"Q({', '.join(selected)})" if selected else "Q(null)"


def tile_projection_label(label: str) -> str:
    compact = label.replace(", ", ",")
    if compact == "Q(qkv,out,up,down)":
        return "Q(qkv,out,\nup,down)"
    if compact == "Q(qkv,up,down)":
        return "Q(qkv,up,\ndown)"
    return compact


def run_dir(tp: int, batch_size: int) -> Path:
    return LIFECYCLE_ROOT / f"{RUN_PREFIX}-tp{tp}" / f"bs{batch_size}"


def load_lifecycle(tp: int, batch_size: int) -> dict:
    return load_json(run_dir(tp, batch_size) / "lifecycle.json")


def live_dynamic_run_dir(batch_size: int) -> Path:
    return LIVE_DYNAMIC_ROOT / f"bs{batch_size}"


def load_live_dynamic_lifecycle(batch_size: int) -> dict:
    return load_json(live_dynamic_run_dir(batch_size) / "lifecycle.json")


def load_live_dynamic_timeline(batch_size: int) -> list[dict]:
    rows = []
    path = live_dynamic_run_dir(batch_size) / "timeline_sampled.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "time_since_decode_start_s": float(row["time_since_decode_start_s"]),
                    "live_requests": int(float(row["live_requests"])),
                    "decoded_token_mass": float(row["decoded_token_mass"]),
                    "mean_decoded_len": float(row["mean_decoded_len"]),
                }
            )
    return rows


def load_timeline(tp: int, batch_size: int) -> list[dict]:
    rows = []
    with (run_dir(tp, batch_size) / "timeline_sampled.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "time_since_decode_start_s": float(row["time_since_decode_start_s"]),
                    "live_requests": int(float(row["live_requests"])),
                    "decoded_token_mass": float(row["decoded_token_mass"]),
                    "mean_decoded_len": float(row["mean_decoded_len"]),
                }
            )
    return rows


def estimate_segments(frontier: dict[int, dict], tp: int, batch_size: int) -> list[SegmentEstimate]:
    trace = load_lifecycle(tp, batch_size)
    segments = []
    for label, duration in zip(trace["bin_labels"], trace["bin_durations_s"]):
        hi, lo = parse_live_request_bin(label)
        frontier_batch_size, frontier_row = nearest_frontier(frontier, hi)
        speedup = float(frontier_row["frontier_speedup_vs_bf16"])
        duration = float(duration)
        estimated = duration / speedup if speedup > 0 else duration
        segments.append(
            SegmentEstimate(
                tp=tp,
                batch_size=batch_size,
                live_request_bin=label,
                live_request_hi=hi,
                live_request_lo=lo,
                frontier_batch_size=frontier_batch_size,
                frontier_speedup_vs_bf16=speedup,
                frontier_speedup_vs_csgmv=float(frontier_row["frontier_speedup_vs_csgmv"]),
                baseline_segment_s=duration,
                estimated_segment_s=estimated,
                policy=policy_label(frontier_row),
                policy_short=compact_policy_label(frontier_row),
                qlora_projection_label=qlora_projection_label(frontier_row),
            )
        )
    return segments


def build_all_estimates(frontier: dict[int, dict]) -> tuple[list[dict], list[SegmentEstimate]]:
    summary_rows = []
    all_segments = []
    for tp in TPS:
        for batch_size in BATCHES:
            trace = load_lifecycle(tp, batch_size)
            segments = estimate_segments(frontier, tp, batch_size)
            all_segments.extend(segments)
            baseline = float(trace["total_drain_s"])
            estimated = sum(segment.estimated_segment_s for segment in segments)
            summary_rows.append(
                {
                    "tp": tp,
                    "batch_size": batch_size,
                    "baseline_drain_s": baseline,
                    "estimated_dynamic_drain_s": estimated,
                    "estimated_e2e_speedup": baseline / estimated if estimated > 0 else 1.0,
                    "decode_output_tok_s": float(trace.get("decode_output_tok_s", 0.0)),
                    "output_len_mean": float(trace.get("output_len_mean", 0.0)),
                    "output_len_p99": float(trace.get("output_len_p99", 0.0)),
                    "output_len_max": int(trace.get("output_len_max", 0)),
                    "finish_reasons": json.dumps(trace.get("finish_reasons", {}), sort_keys=True),
                }
            )
    return summary_rows, all_segments


def write_tables(summary_rows: list[dict], segments: list[SegmentEstimate]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "real_e2e_estimates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    segment_rows = [segment.__dict__ | {"segment_speedup": segment.speedup} for segment in segments]
    with (OUT_DIR / "real_window_estimates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(segment_rows[0]))
        writer.writeheader()
        writer.writerows(segment_rows)

    payload = {
        "metadata": {
            "model": "Qwen2.5-14B-Instruct",
            "dataset": "Eurus-2-RL",
            "source": "Measured lifecycle traces plus measured QLoRA frontier speedups",
            "caveat": "Estimated dynamic-switch speedup; not a live dual-weight serving measurement.",
        },
        "summary": summary_rows,
        "segments": segment_rows,
    }
    (OUT_DIR / "real_estimates.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "| TP | Batch | BF16 drain s | Estimated dynamic drain s | Estimated speedup | output p99 | output max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['tp']} | {row['batch_size']} | {row['baseline_drain_s']:.1f} | "
            f"{row['estimated_dynamic_drain_s']:.1f} | x{row['estimated_e2e_speedup']:.2f} | "
            f"{row['output_len_p99']:.0f} | {row['output_len_max']} |"
        )
    (OUT_DIR / "real_estimates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def live_summary_rows() -> list[dict]:
    rows = []
    for batch_size in BATCHES:
        baseline = load_lifecycle(4, batch_size)
        dynamic = load_live_dynamic_lifecycle(batch_size)
        baseline_drain = float(baseline["total_drain_s"])
        dynamic_drain = float(dynamic["total_drain_s"])
        rows.append(
            {
                "tp": 4,
                "batch_size": batch_size,
                "baseline_bf16_drain_s": baseline_drain,
                "live_dynamic_drain_s": dynamic_drain,
                "live_dynamic_speedup_vs_bf16": (
                    baseline_drain / dynamic_drain if dynamic_drain > 0 else 1.0
                ),
                "baseline_output_len_p99": float(baseline.get("output_len_p99", 0.0)),
                "baseline_output_len_max": int(baseline.get("output_len_max", 0)),
                "dynamic_output_len_p99": float(dynamic.get("output_len_p99", 0.0)),
                "dynamic_output_len_max": int(dynamic.get("output_len_max", 0)),
                "dynamic_finish_reasons": json.dumps(
                    dynamic.get("finish_reasons", {}), sort_keys=True
                ),
            }
        )
    return rows


def write_live_tables(rows: list[dict]) -> None:
    with (OUT_DIR / "real_live_dynamic_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| TP | Batch | BF16 drain s | Live dynamic drain s | Speedup vs BF16 | BF16 max len | Dynamic max len |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tp']} | {row['batch_size']} | "
            f"{row['baseline_bf16_drain_s']:.1f} | {row['live_dynamic_drain_s']:.1f} | "
            f"x{row['live_dynamic_speedup_vs_bf16']:.2f} | "
            f"{row['baseline_output_len_max']} | {row['dynamic_output_len_max']} |"
        )
    (OUT_DIR / "real_live_dynamic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def cumulative_boundaries(segments: list[SegmentEstimate]) -> list[tuple[float, SegmentEstimate]]:
    t = 0.0
    out = []
    for segment in segments:
        t += segment.baseline_segment_s
        out.append((t, segment))
    return out


def cumulative_starts_and_ends(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(np.array(values, dtype=float))
    starts = np.concatenate(([0.0], ends[:-1]))
    return starts, ends


def remap_timeline_clock(
    timeline: list[dict],
    segments: list[SegmentEstimate],
    target_durations: list[float],
) -> np.ndarray:
    baseline_starts, baseline_ends = cumulative_starts_and_ends(
        [segment.baseline_segment_s for segment in segments]
    )
    target_starts, _ = cumulative_starts_and_ends(target_durations)
    target_durations_np = np.array(target_durations, dtype=float)
    baseline_durations_np = np.array([segment.baseline_segment_s for segment in segments], dtype=float)

    out = []
    for row in timeline:
        t = float(row["time_since_decode_start_s"])
        idx = int(np.searchsorted(baseline_ends, t, side="left"))
        idx = min(max(idx, 0), len(segments) - 1)
        denom = baseline_durations_np[idx]
        frac = 0.0 if denom <= 0 else (t - baseline_starts[idx]) / denom
        frac = min(max(frac, 0.0), 1.0)
        out.append(float(target_starts[idx] + frac * target_durations_np[idx]))
    return np.array(out, dtype=float)


def policy_tile_text(segment: SegmentEstimate) -> str:
    return (
        f"{segment.live_request_bin}\n"
        f"{tile_projection_label(segment.qlora_projection_label)}=\n"
        f"x{segment.frontier_speedup_vs_bf16:.2f}/x{segment.frontier_speedup_vs_csgmv:.2f}"
    )


def add_policy_tiles(ax, segments: list[SegmentEstimate]) -> None:
    tile_count = len(segments)
    for idx, segment in enumerate(segments):
        color = POLICY_COLORS.get(segment.policy_short, COLORS["window"])
        ax.barh(
            0,
            0.92,
            left=idx - 0.46,
            height=0.78,
            color=color,
            alpha=0.26,
            edgecolor=color,
            linewidth=1.0,
        )
        ax.text(
            idx,
            0,
            policy_tile_text(segment),
            ha="center",
            va="center",
            fontsize=5.8,
            color="#111111",
        )
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.55, tile_count - 0.45)
    ax.set_yticks([])
    ax.set_ylabel("Policy")
    ax.set_xticks([])
    ax.set_xlabel("Live-request bin policy; ratios are speedups vs BF16 / vs INT4 CSGMV")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def lifetime_inputs(frontier: dict[int, dict]) -> tuple[int, int, dict, list[dict], list[SegmentEstimate]]:
    tp = 4
    batch_size = 512
    lifecycle = load_lifecycle(tp, batch_size)
    timeline = load_timeline(tp, batch_size)
    segments = estimate_segments(frontier, tp, batch_size)
    return tp, batch_size, lifecycle, timeline, segments


def plot_baseline_lifetime(frontier: dict[int, dict]) -> None:
    tp, batch_size, lifecycle, timeline, _ = lifetime_inputs(frontier)
    times = np.array([row["time_since_decode_start_s"] for row in timeline])
    live = np.array([row["live_requests"] for row in timeline])
    mean_len = np.array([row["mean_decoded_len"] for row in timeline])
    mean_len = np.where(live > 0, mean_len, np.nan)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5.95, 3.45),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9]},
        constrained_layout=True,
    )
    axes[0].plot(times, live, color=COLORS["live"], label="BF16 merged measured")
    axes[0].set_ylabel("Live requests")
    axes[0].set_ylim(0, 540)
    axes[0].grid(True, alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper right")
    add_rollout_end_marker(
        axes[0],
        float(lifecycle["total_drain_s"]),
        color=COLORS["bf16"],
        label="BF16 end",
    )

    axes[1].plot(times, mean_len, color=COLORS["mean"])
    axes[1].set_xlabel("BF16 measured decode clock (s)")
    axes[1].set_ylabel("Mean decoded length")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.25, linewidth=0.6)
    axes[1].axvline(float(lifecycle["total_drain_s"]), color=COLORS["bf16"], linestyle=":", linewidth=1.1)

    fig.suptitle(
        f"Baseline BF16 Rollout Lifetime\n"
        f"Qwen2.5-14B-Instruct Eurus-2-RL, TP{tp}, bs{batch_size}; "
        f"drain={lifecycle['total_drain_s']:.0f}s"
    )
    fig.savefig(OUT_DIR / "real_baseline_lifetime.png", dpi=300)
    fig.savefig(OUT_DIR / "real_baseline_lifetime.pdf")
    plt.close(fig)


def plot_precision_lifetime(frontier: dict[int, dict]) -> None:
    tp, batch_size, lifecycle, timeline, segments = lifetime_inputs(frontier)
    baseline_drain_s = float(lifecycle["total_drain_s"])
    estimated_drain_s = sum(segment.estimated_segment_s for segment in segments)
    estimated_csgmv_drain_s = sum(segment.csgmv_segment_s for segment in segments)
    overall_speedup = baseline_drain_s / estimated_drain_s if estimated_drain_s > 0 else 1.0
    overall_speedup_vs_csgmv = (
        estimated_csgmv_drain_s / estimated_drain_s if estimated_drain_s > 0 else 1.0
    )
    dynamic_durations = [segment.estimated_segment_s for segment in segments]
    times = remap_timeline_clock(timeline, segments, dynamic_durations)
    live = np.array([row["live_requests"] for row in timeline])
    mean_len = np.array([row["mean_decoded_len"] for row in timeline])
    mean_len = np.where(live > 0, mean_len, np.nan)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.45, 4.55),
        sharex=False,
        gridspec_kw={"height_ratios": [1.05, 0.9, 0.46]},
        constrained_layout=True,
    )

    boundaries = []
    running = 0.0
    for segment in segments:
        running += segment.estimated_segment_s
        boundaries.append((running, segment))
    prev = 0.0
    for idx, (end, segment) in enumerate(boundaries):
        color = POLICY_COLORS.get(segment.policy_short, COLORS["window"])
        for ax in axes[:2]:
            ax.axvspan(prev, end, color=color, alpha=0.13, linewidth=0)
        prev = end

    axes[0].plot(times, live, color=COLORS["live"], label="Estimated dynamic lifetime")
    axes[0].set_ylabel("Live requests")
    axes[0].set_ylim(0, 540)
    axes[0].grid(True, alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper right")
    add_rollout_end_marker(
        axes[0],
        estimated_drain_s,
        color=COLORS["frontier"],
        label="dynamic end",
    )
    axes[0].text(
        0.985,
        0.73,
        "Overall decode estimate\n"
        f"BF16: {baseline_drain_s:.0f}s -> {estimated_drain_s:.0f}s, x{overall_speedup:.2f}\n"
        f"INT4 CSGMV: {estimated_csgmv_drain_s:.0f}s -> {estimated_drain_s:.0f}s, "
        f"x{overall_speedup_vs_csgmv:.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#333333",
            "linewidth": 0.75,
            "alpha": 0.94,
        },
    )

    axes[1].plot(times, mean_len, color=COLORS["mean"], label="Mean decoded length")
    axes[1].set_xlabel("Estimated dynamic precision decode clock (s)")
    axes[1].set_ylabel("Mean decoded length")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.25, linewidth=0.6)
    axes[1].axvline(estimated_drain_s, color=COLORS["frontier"], linestyle=":", linewidth=1.1)
    axes[0].set_xlim(times.min(), max(times.max(), estimated_drain_s) * 1.02)
    axes[1].set_xlim(times.min(), max(times.max(), estimated_drain_s) * 1.02)

    add_policy_tiles(axes[2], segments)

    fig.suptitle(
        f"Our Estimated Dynamic-Precision Rollout Lifetime, TP{tp}, bs{batch_size}\n"
        "Q(...) lists INT4+Torch2S LoRA projections; speedup is vs BF16 / vs INT4 CSGMV",
    )
    fig.savefig(OUT_DIR / "real_precision_lifetime.png", dpi=300)
    fig.savefig(OUT_DIR / "real_precision_lifetime.pdf")
    plt.close(fig)


def plot_lifetime_comparison(frontier: dict[int, dict]) -> None:
    tp, batch_size, lifecycle, timeline, segments = lifetime_inputs(frontier)
    baseline_times = np.array([row["time_since_decode_start_s"] for row in timeline])
    live = np.array([row["live_requests"] for row in timeline])
    dynamic_times = remap_timeline_clock(
        timeline, segments, [segment.estimated_segment_s for segment in segments]
    )
    csgmv_times = remap_timeline_clock(
        timeline, segments, [segment.csgmv_segment_s for segment in segments]
    )

    fig, ax = plt.subplots(figsize=(5.75, 2.95), constrained_layout=True)
    ax.plot(baseline_times, live, color=COLORS["bf16"], label="BF16 merged measured")
    ax.plot(csgmv_times, live, color="#DD8452", linestyle="--", label="INT4 CSGMV estimated")
    ax.plot(dynamic_times, live, color=COLORS["frontier"], label="Our dynamic estimate")
    ax.set_xlabel("Decode clock (s)")
    ax.set_ylabel("Live requests")
    ax.set_ylim(0, 540)
    ax.set_xlim(0, max(float(lifecycle["total_drain_s"]), csgmv_times.max(), dynamic_times.max()) * 1.03)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    add_rollout_end_marker(
        ax,
        float(lifecycle["total_drain_s"]),
        color=COLORS["bf16"],
        label="BF16 end",
    )
    add_rollout_end_marker(
        ax,
        float(csgmv_times.max()),
        color="#DD8452",
        label="CSGMV end",
    )
    add_rollout_end_marker(
        ax,
        float(dynamic_times.max()),
        color=COLORS["frontier"],
        label="dynamic end",
    )
    fig.suptitle(
        f"Rollout Lifetime Comparison, Qwen2.5-14B-Instruct Eurus-2-RL, TP{tp}, bs{batch_size}"
    )
    fig.savefig(OUT_DIR / "real_lifetime_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "real_lifetime_comparison.pdf")
    plt.close(fig)


def plot_live_dynamic_summary(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 2.8), constrained_layout=True)
    labels = [f"bs{row['batch_size']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    baseline = np.array([row["baseline_bf16_drain_s"] for row in rows], dtype=float)
    dynamic = np.array([row["live_dynamic_drain_s"] for row in rows], dtype=float)
    bars_a = ax.bar(
        x - width / 2,
        baseline,
        width,
        color=COLORS["bf16"],
        label="BF16 merged measured",
    )
    bars_b = ax.bar(
        x + width / 2,
        dynamic,
        width,
        color=COLORS["frontier"],
        label="Live dynamic BF16/INT4",
    )
    ax.set_ylabel("Decode drain time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(baseline.max(), dynamic.max()) * 1.16)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    for bar, row in zip(bars_b, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.04,
            f"x{row['live_dynamic_speedup_vs_bf16']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
        )
    ax.bar_label(
        bars_a,
        labels=[f"{b.get_height():.0f}" for b in bars_a],
        fontsize=6.8,
        label_type="center",
        color="white",
    )
    ax.bar_label(
        bars_b,
        labels=[f"{b.get_height():.0f}" for b in bars_b],
        fontsize=6.8,
        label_type="center",
        color="white",
    )
    fig.suptitle("Actual Dynamic-Precision Serving, Qwen2.5-14B Eurus-2-RL TP4")
    fig.savefig(OUT_DIR / "real_live_dynamic_summary.png", dpi=300)
    fig.savefig(OUT_DIR / "real_live_dynamic_summary.pdf")
    plt.close(fig)


def plot_live_dynamic_lifetime() -> None:
    batch_size = 512
    baseline = load_lifecycle(4, batch_size)
    dynamic = load_live_dynamic_lifecycle(batch_size)
    baseline_timeline = load_timeline(4, batch_size)
    dynamic_timeline = load_live_dynamic_timeline(batch_size)

    baseline_times = np.array(
        [row["time_since_decode_start_s"] for row in baseline_timeline], dtype=float
    )
    baseline_live = np.array([row["live_requests"] for row in baseline_timeline], dtype=float)
    dynamic_times = np.array(
        [row["time_since_decode_start_s"] for row in dynamic_timeline], dtype=float
    )
    dynamic_live = np.array([row["live_requests"] for row in dynamic_timeline], dtype=float)

    fig, ax = plt.subplots(figsize=(5.25, 2.95), constrained_layout=True)
    ax.plot(
        baseline_times,
        baseline_live,
        color=COLORS["bf16"],
        label=f"BF16 merged measured ({baseline['total_drain_s']:.0f}s)",
    )
    ax.plot(
        dynamic_times,
        dynamic_live,
        color=COLORS["frontier"],
        label=f"Live dynamic measured ({dynamic['total_drain_s']:.0f}s)",
    )
    ax.set_xlabel("Decode clock (s)")
    ax.set_ylabel("Live requests")
    ax.set_ylim(0, 540)
    ax.set_xlim(
        0,
        max(float(baseline["total_drain_s"]), float(dynamic["total_drain_s"])) * 1.03,
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    add_rollout_end_marker(
        ax,
        float(baseline["total_drain_s"]),
        color=COLORS["bf16"],
        label="BF16 end",
    )
    add_rollout_end_marker(
        ax,
        float(dynamic["total_drain_s"]),
        color=COLORS["frontier"],
        label="dynamic end",
    )
    speedup = float(baseline["total_drain_s"]) / float(dynamic["total_drain_s"])
    ax.text(
        0.985,
        0.68,
        f"Actual live speedup: x{speedup:.2f}\n"
        f"Dynamic max len: {int(dynamic['output_len_max'])}\n"
        "Prefill BF16; decode switches by policy",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#333333",
            "linewidth": 0.75,
            "alpha": 0.94,
        },
    )
    fig.suptitle(
        "Actual Rollout Lifetime: BF16 Baseline vs Dynamic Precision, TP4 bs512"
    )
    fig.savefig(OUT_DIR / "real_live_dynamic_lifetime.png", dpi=300)
    fig.savefig(OUT_DIR / "real_live_dynamic_lifetime.pdf")
    plt.close(fig)


def plot_tp1_debug_breakdown() -> None:
    baseline = load_lifecycle(1, 128)
    dynamic = load_json(
        OUT_DIR
        / "live-runs/dynamic-qwen2.5-14b-eurus-tp1/bs128/lifecycle.json"
    )
    labels = baseline["bin_labels"]
    baseline_bins = np.array(baseline["bin_durations_s"], dtype=float)
    dynamic_by_label = {
        label: float(duration)
        for label, duration in zip(dynamic["bin_labels"], dynamic["bin_durations_s"])
    }
    dynamic_bins = np.array([dynamic_by_label[label] for label in labels], dtype=float)
    ratios = np.divide(
        baseline_bins,
        dynamic_bins,
        out=np.ones_like(baseline_bins),
        where=dynamic_bins > 0,
    )

    with (OUT_DIR / "real_tp1_debug_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "live_request_bin",
                "bf16_tp1_s",
                "dynamic_tp1_s",
                "speedup_vs_bf16",
            ],
        )
        writer.writeheader()
        for label, b, d, r in zip(labels, baseline_bins, dynamic_bins, ratios):
            writer.writerow(
                {
                    "live_request_bin": label,
                    "bf16_tp1_s": b,
                    "dynamic_tp1_s": d,
                    "speedup_vs_bf16": r,
                }
            )

    y = np.arange(len(labels))
    height = 0.34
    fig, ax = plt.subplots(figsize=(5.35, 3.05), constrained_layout=True)
    bars_a = ax.barh(
        y - height / 2,
        baseline_bins,
        height,
        color=COLORS["bf16"],
        label="BF16 TP1 measured",
    )
    bars_b = ax.barh(
        y + height / 2,
        dynamic_bins,
        height,
        color=COLORS["frontier"],
        label="Live dynamic TP1 measured",
    )
    ax.set_xlabel("Time in live-request bin (s)")
    ax.set_ylabel("Live-request bin")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(baseline_bins.max(), dynamic_bins.max()) * 1.35)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    for idx, (b, d, r) in enumerate(zip(baseline_bins, dynamic_bins, ratios)):
        ax.text(
            max(b, d) + ax.get_xlim()[1] * 0.015,
            idx,
            f"{b:.1f}->{d:.1f}s  x{r:.2f}",
            ha="left",
            va="center",
            fontsize=6.8,
        )
    ax.bar_label(
        bars_a,
        labels=[f"{b.get_width():.1f}" for b in bars_a],
        fontsize=6.6,
        label_type="center",
        color="white",
    )
    ax.bar_label(
        bars_b,
        labels=[f"{b.get_width():.1f}" for b in bars_b],
        fontsize=6.6,
        label_type="center",
        color="white",
    )
    overall = float(baseline["total_drain_s"]) / float(dynamic["total_drain_s"])
    fig.suptitle(
        f"TP1 bs128 Debug Breakdown, Qwen2.5-14B Eurus-2-RL, x{overall:.2f} overall"
    )
    fig.savefig(OUT_DIR / "real_tp1_debug_breakdown.png", dpi=300)
    fig.savefig(OUT_DIR / "real_tp1_debug_breakdown.pdf")
    plt.close(fig)


def plot_speedup_summary(summary_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.75, 2.95), constrained_layout=True)
    labels = [f"TP{row['tp']} bs{row['batch_size']}" for row in summary_rows]
    x = np.arange(len(summary_rows))
    width = 0.36
    baseline = np.array([row["baseline_drain_s"] for row in summary_rows])
    estimated = np.array([row["estimated_dynamic_drain_s"] for row in summary_rows])
    bars_a = ax.bar(x - width / 2, baseline, width, color=COLORS["bf16"], label="Measured BF16 merged")
    bars_b = ax.bar(x + width / 2, estimated, width, color=COLORS["frontier"], label="Estimated dynamic precision")
    ax.set_ylabel("Decode drain time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0, max(baseline.max(), estimated.max()) * 1.18)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    for bar, row in zip(bars_b, summary_rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.025,
            f"x{row['estimated_e2e_speedup']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    for bars in (bars_a, bars_b):
        ax.bar_label(bars, labels=[f"{b.get_height():.0f}" for b in bars], fontsize=6.7, padding=1)
    fig.suptitle("Estimated Decode-Drain Gain from Real Lifecycle Bins")
    fig.savefig(OUT_DIR / "real_e2e_speedup_bars.png", dpi=300)
    fig.savefig(OUT_DIR / "real_e2e_speedup_bars.pdf")
    plt.close(fig)


def plot_window_breakdown(segments: list[SegmentEstimate]) -> None:
    selected = [s for s in segments if s.tp == 4 and s.batch_size == 512]
    labels = [s.live_request_bin for s in selected]
    y = np.arange(len(selected))
    height = 0.34
    baseline = np.array([s.baseline_segment_s for s in selected])
    estimated = np.array([s.estimated_segment_s for s in selected])

    fig, ax = plt.subplots(figsize=(5.65, 3.15), constrained_layout=True)
    bars_a = ax.barh(y - height / 2, baseline, height, color=COLORS["bf16"], label="Measured BF16 bin time")
    bars_b = ax.barh(y + height / 2, estimated, height, color=COLORS["frontier"], label="Estimated dynamic bin time")
    ax.set_xlabel("Time in live-request bin (s)")
    ax.set_ylabel("Live-request bin")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(baseline.max(), estimated.max()) * 1.45)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")

    for segment, base, est, y_text in zip(selected, baseline, estimated, y):
        x = max(base, est) + ax.get_xlim()[1] * 0.015
        ax.text(
            x,
            y_text,
            f"{base:.1f}->{est:.1f}s  x{segment.speedup:.2f} {segment.policy}",
            ha="left",
            va="center",
            fontsize=6.6,
        )
    fig.suptitle("TP4 bs512 Tail Bins with Frontier Rescaling")
    fig.savefig(OUT_DIR / "real_window_breakdown_tp4_bs512.png", dpi=300)
    fig.savefig(OUT_DIR / "real_window_breakdown_tp4_bs512.pdf")
    plt.close(fig)


def plot_frontier_throughput(frontier: dict[int, dict]) -> None:
    rows = [frontier[key] for key in sorted(frontier)]
    bs = np.array([row["batch_size"] for row in rows], dtype=int)
    bf16 = np.array([row["bf16_merged_tok_s"] for row in rows], dtype=float)
    qlora = np.array([row["qlora_torch_twostream_tok_s"] for row in rows], dtype=float)
    dynamic = np.array([row["kernel_guided_frontier_tok_s"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.25, 2.85), constrained_layout=True)
    ax.plot(bs, bf16, marker="o", color=COLORS["bf16"], label="BF16 merged")
    ax.plot(bs, qlora, marker="s", color="#DD8452", label="INT4 + Torch2S LoRA")
    ax.plot(bs, dynamic, marker="^", color=COLORS["frontier"], label="Projection frontier")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(x) for x in bs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size")
    ax.set_ylabel("Measured / estimated tokens per second")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Qwen2.5-14B Decode Frontier")
    fig.savefig(OUT_DIR / "real_frontier_throughput.png", dpi=300)
    fig.savefig(OUT_DIR / "real_frontier_throughput.pdf")
    plt.close(fig)


def write_readme(summary_rows: list[dict], live_rows: list[dict]) -> None:
    best = max(summary_rows, key=lambda row: row["estimated_e2e_speedup"])
    best_live = max(live_rows, key=lambda row: row["live_dynamic_speedup_vs_bf16"])
    text = f"""# Real-Data Rollout Precision Deliverables

These figures combine the earlier estimator with actual live dynamic-precision
serving traces from this branch.

Inputs:

- Real request lifecycle traces: Qwen2.5-14B-Instruct on Eurus-2-RL, TP1/TP4, batch sizes 128/256/512.
- Real decode frontier: `.rollout-profile/qlora-decoding-throughput/frontier_decoding_throughput.json`.
- The estimator rescales each live-request bin independently by the frontier speedup for the nearest active batch size.
- Actual live dynamic traces: `.rollout-impl-codex/real-deliverable/live-runs/dynamic-qwen2.5-14b-eurus-tp4`.

Important caveat: the live dynamic traces are raw serving runs with stochastic
sampling. The bs512 dynamic run had one request hit the 32K length cap, so the
raw drain-time comparison is dominated by a different sampled long tail. The
estimator plots remain useful for controlled bin-by-bin projection of the BF16
lifecycle, while `real_live_dynamic_*` shows what the implemented system did.

In lifetime plots, the dotted vertical line plus diamond marker denotes the end
of the corresponding rollout trace.

Generated plots:

- `real_baseline_lifetime.png`: measured BF16 TP4 bs512 lifetime.
- `real_precision_lifetime.png`: estimated dynamic-precision TP4 bs512 lifetime on the recomputed clock.
- `real_lifetime_comparison.png`: BF16 measured, INT4 CSGMV estimated, and our dynamic estimate.
- `real_live_dynamic_lifetime.png`: actual BF16 baseline vs actual live dynamic TP4 bs512 trace.
- `real_live_dynamic_summary.png`: actual live dynamic drain time for TP4 bs128/256/512.
- `real_tp1_debug_breakdown.png`: actual one-GPU TP1 bs128 bin-by-bin debug comparison.
- `real_window_breakdown_tp4_bs512.png`: measured vs estimated time per live-request bin.
- `real_e2e_speedup_bars.png`: estimated full-drain speedup for TP1/TP4 and bs128/256/512.
- `real_frontier_throughput.png`: measured BF16/QLoRA serving frontier used by the estimator.

Best estimated row in this batch of data: TP{best['tp']} bs{best['batch_size']}, x{best['estimated_e2e_speedup']:.2f} decode-drain speedup.
Best actual live dynamic row: TP{best_live['tp']} bs{best_live['batch_size']}, x{best_live['live_dynamic_speedup_vs_bf16']:.2f} versus the prior BF16 trace.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_matplotlib()
    frontier = load_frontier()
    summary_rows, segments = build_all_estimates(frontier)
    live_rows = live_summary_rows()
    write_tables(summary_rows, segments)
    write_live_tables(live_rows)
    plot_baseline_lifetime(frontier)
    plot_precision_lifetime(frontier)
    plot_lifetime_comparison(frontier)
    plot_live_dynamic_lifetime()
    plot_live_dynamic_summary(live_rows)
    plot_tp1_debug_breakdown()
    plot_window_breakdown(segments)
    plot_speedup_summary(summary_rows)
    plot_frontier_throughput(frontier)
    write_readme(summary_rows, live_rows)


if __name__ == "__main__":
    main()
