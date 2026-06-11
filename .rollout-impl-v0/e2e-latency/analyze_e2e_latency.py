#!/usr/bin/env python3
"""Analyze request-lifetime traces for the rollout e2e latency experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def safe_finish_reason(value):
    if isinstance(value, dict):
        return value.get("type") or value.get("reason") or json.dumps(value, sort_keys=True)
    return value if value is not None else "unknown"


def load_requests(run_dir: Path) -> tuple[dict, list[dict]]:
    payload = json.loads((run_dir / "requests.json").read_text(encoding="utf-8"))
    return payload.get("metadata", {}), payload["requests"]


def load_token_events(run_dir: Path) -> list[tuple[int, float, int]]:
    events = []
    with (run_dir / "token_events.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events.append(
                (
                    int(row["request_id"]),
                    float(row["timestamp_s"]),
                    int(row["decoded_tokens"]),
                )
            )
    events.sort(key=lambda item: item[1])
    return events


def halving_bins(n: int) -> list[tuple[int, int]]:
    bins = []
    hi = n
    while hi > 8:
        lo = hi // 2
        bins.append((lo, hi))
        hi = lo
    bins.append((0, hi))
    return bins


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def build_lifecycle(run_dir: Path, sample_points: int) -> dict:
    metadata, requests = load_requests(run_dir)
    successful = [
        r
        for r in requests
        if r.get("success") and r.get("first_token_s") is not None and r.get("last_token_s") is not None
    ]
    if not successful:
        raise RuntimeError(f"no successful requests in {run_dir}")
    events = load_token_events(run_dir)
    if not events:
        raise RuntimeError(f"no token events in {run_dir}")

    n = len(successful)
    t0 = min(float(r["first_token_s"]) for r in successful)
    finish_items = sorted(
        (float(r.get("last_token_s") or r.get("end_s")), int(r["request_id"]))
        for r in successful
    )
    finish_ts = [ts for ts, _ in finish_items]
    finish_rel = [t - t0 for t in finish_ts]
    total_drain_s = finish_rel[-1] if finish_rel else 0.0
    output_lens = [int(r.get("output_len") or 0) for r in successful]
    completion_latencies = [
        float((r.get("last_token_s") or r.get("end_s")) - r["start_s"]) for r in successful
    ]
    ttft_latencies = [
        float(r["first_token_s"] - r["start_s"])
        for r in successful
        if r.get("first_token_s") is not None
    ]

    reasons: dict[str, int] = {}
    for request in successful:
        reason = safe_finish_reason(request.get("finish_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1

    for request in successful:
        events.append(
            (
                int(request["request_id"]),
                float(request.get("last_token_s") or request.get("end_s")),
                int(request.get("output_len") or 0),
            )
        )
    events.sort(key=lambda item: item[1])

    bins = halving_bins(n)
    durations = [0.0] * len(bins)
    prev = 0.0
    live = n
    for rel_t in finish_rel:
        dur = rel_t - prev
        for i, (lo, hi) in enumerate(bins):
            if lo < live <= hi:
                durations[i] += dur
                break
        prev = rel_t
        live -= 1

    latest: dict[int, int] = {}
    finished_ids: set[int] = set()
    running_token_mass = 0
    peak_decoded_token_mass = 0
    peak_mean_decoded_len = 0.0
    timeline_events = []
    finish_idx = 0
    for request_id, ts, decoded_tokens in events:
        rel_t = ts - t0
        if rel_t < 0:
            continue
        if request_id not in finished_ids:
            running_token_mass += decoded_tokens - latest.get(request_id, 0)
            latest[request_id] = decoded_tokens
        while finish_idx < len(finish_ts) and finish_ts[finish_idx] < ts:
            finished_request_id = finish_items[finish_idx][1]
            if finished_request_id not in finished_ids:
                running_token_mass -= latest.get(finished_request_id, 0)
                finished_ids.add(finished_request_id)
            finish_idx += 1
        live_requests = max(n - finish_idx, 0)
        decoded_token_mass = max(running_token_mass, 0)
        peak_decoded_token_mass = max(peak_decoded_token_mass, decoded_token_mass)
        mean_decoded_len = decoded_token_mass / live_requests if live_requests else 0.0
        peak_mean_decoded_len = max(peak_mean_decoded_len, mean_decoded_len)
        timeline_events.append((rel_t, live_requests, decoded_token_mass, mean_decoded_len))

    sample_rows = sample_timeline(timeline_events, total_drain_s, sample_points)
    write_timeline(run_dir, timeline_events, sample_rows)

    return {
        "metadata": metadata,
        "n": n,
        "t0_perf": t0,
        "total_drain_s": total_drain_s,
        "rel_finish": finish_rel,
        "bin_labels": [f"{hi}-{lo}" for lo, hi in bins],
        "bin_durations_s": durations,
        "finish_reasons": reasons,
        "output_len_mean": float(np.mean(output_lens)) if output_lens else 0.0,
        "output_len_p50": percentile(output_lens, 50),
        "output_len_p90": percentile(output_lens, 90),
        "output_len_p99": percentile(output_lens, 99),
        "output_len_max": int(max(output_lens)) if output_lens else 0,
        "first_token_spread_s": max(float(r["first_token_s"]) for r in successful) - t0,
        "peak_decoded_token_mass": peak_decoded_token_mass,
        "peak_mean_decoded_len": peak_mean_decoded_len,
        "total_observed_output_tokens": int(sum(output_lens)),
        "decode_output_tok_s": (sum(output_lens) / total_drain_s) if total_drain_s > 0 else 0.0,
        "timeline_sample_points": len(sample_rows),
        "request_latency_p50_s": percentile(completion_latencies, 50),
        "request_latency_p90_s": percentile(completion_latencies, 90),
        "request_latency_p99_s": percentile(completion_latencies, 99),
        "ttft_p50_s": percentile(ttft_latencies, 50),
        "ttft_p90_s": percentile(ttft_latencies, 90),
        "ttft_p99_s": percentile(ttft_latencies, 99),
    }


def sample_timeline(
    timeline_events: list[tuple[float, int, int, float]],
    total_drain_s: float,
    sample_points: int,
) -> list[tuple[float, int, int, float]]:
    if not timeline_events:
        return []
    if total_drain_s <= 0:
        return [timeline_events[-1]]
    targets = np.linspace(0.0, total_drain_s, sample_points)
    rows = []
    idx = 0
    current = timeline_events[0]
    for target in targets:
        while idx + 1 < len(timeline_events) and timeline_events[idx + 1][0] <= target:
            idx += 1
            current = timeline_events[idx]
        rows.append((float(target), current[1], current[2], current[3]))
    return rows


def write_timeline(
    run_dir: Path,
    timeline_events: list[tuple[float, int, int, float]],
    sample_rows: list[tuple[float, int, int, float]],
) -> None:
    header = ["time_since_decode_start_s", "live_requests", "decoded_token_mass", "mean_decoded_len"]
    with (run_dir / "timeline_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(timeline_events)
    with (run_dir / "timeline_sampled.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sample_rows)


def plot_lifecycle(
    run_dir: Path,
    lifecycle: dict,
    model_label: str,
    dataset_label: str,
    mode_label: str,
) -> None:
    import matplotlib.pyplot as plt

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

    sample_t, sample_live, sample_mean = [], [], []
    with (run_dir / "timeline_sampled.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_t.append(float(row["time_since_decode_start_s"]))
            sample_live.append(int(float(row["live_requests"])))
            sample_mean.append(float(row["mean_decoded_len"]))

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), constrained_layout=True)
    axes[0].plot(sample_t, sample_live, color="#4C72B0", linewidth=1.8)
    axes[0].set_xlabel("Decode time (s)")
    axes[0].set_ylabel("Live requests")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.25, linewidth=0.6)

    sample_mean_live = [mean if live > 0 else float("nan") for live, mean in zip(sample_live, sample_mean)]
    axes[1].plot(sample_t, sample_mean_live, color="#C44E52", linewidth=1.8)
    axes[1].set_xlabel("Decode time (s)")
    axes[1].set_ylabel("Mean decoded length")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.25, linewidth=0.6)

    axes[2].bar(lifecycle["bin_labels"], lifecycle["bin_durations_s"], color="#55A868")
    axes[2].set_xlabel("Live-request bin")
    axes[2].set_ylabel("Duration (s)")
    axes[2].tick_params(axis="x", rotation=40)
    axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)

    meta = lifecycle.get("metadata", {})
    batch = meta.get("batch_size", lifecycle["n"])
    temp = meta.get("temperature", "?")
    cap = meta.get("max_new_tokens", "?")
    fig.suptitle(
        f"{model_label} on {dataset_label}, {mode_label}, bs{batch}, temp={temp}, "
        f"{cap} cap (drain {lifecycle['total_drain_s']:.0f}s, p50 {lifecycle['output_len_p50']:.0f}, "
        f"p99 {lifecycle['output_len_p99']:.0f})",
        fontsize=10,
    )
    fig.savefig(run_dir / "lifecycle.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=1000)
    parser.add_argument("--model-label", default="Qwen2.5-14B-Instruct")
    parser.add_argument("--dataset-label", default="Eurus-2-RL")
    parser.add_argument("--mode-label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lifecycle = build_lifecycle(args.run_dir, args.sample_points)
    (args.run_dir / "lifecycle.json").write_text(
        json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8"
    )
    plot_lifecycle(
        args.run_dir,
        lifecycle,
        model_label=args.model_label,
        dataset_label=args.dataset_label,
        mode_label=args.mode_label,
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "n": lifecycle["n"],
                "drain_s": lifecycle["total_drain_s"],
                "output_len_p50": lifecycle["output_len_p50"],
                "output_len_p99": lifecycle["output_len_p99"],
                "output_len_max": lifecycle["output_len_max"],
                "finish_reasons": lifecycle["finish_reasons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
