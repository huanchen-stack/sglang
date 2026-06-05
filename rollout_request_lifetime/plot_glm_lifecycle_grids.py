"""Create rollout lifecycle grids from long scheduler traces.

The generic report plotter renders one image per experiment. This script is for
side-by-side inspection of batch size, TP, and precision settings. It can filter
fixed-length benchmark traces by output cap, or EOS-cap traces by dropping health
checks and warmup requests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from matplotlib.collections import LineCollection

from plot_rollout_traces import (
    PHASE_COLORS,
    Span,
    collapse_lifecycle_spans,
    pair_spans,
    read_jsonl,
    request_order,
)


def load_metadata(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.setdefault("name", exp_dir.name)
    return metadata


def benchmark_output_len(exp_dir: Path, fallback: int) -> int:
    path = exp_dir / "bench_serving.jsonl"
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get("sharegpt_output_len")
            if isinstance(value, (int, float)) and math.isfinite(value):
                return int(value)
    return fallback


def load_events(exp_dir: Path) -> list[dict[str, Any]]:
    events = []
    for path in sorted((exp_dir / "traces").glob("*.jsonl")):
        events.extend(read_jsonl(path))
    events.sort(key=lambda row: row.get("ts", 0.0))
    return events


def fixed_output_request_ids(
    events: list[dict[str, Any]], expected_output_len: int
) -> set[str]:
    rids = set()
    for event in events:
        rid = event.get("rid")
        if not rid or str(rid).startswith("HEALTH_CHECK_"):
            continue
        if (
            event.get("event") == "request_complete"
            and event.get("max_new_tokens") == expected_output_len
        ):
            rids.add(rid)
    return rids


def eos_cap_request_ids(
    events: list[dict[str, Any]],
    warmup_max_new_tokens: set[int],
    min_benchmark_max_new_tokens: int,
) -> set[str]:
    rids = set()
    for event in events:
        rid = event.get("rid")
        if not rid or str(rid).startswith("HEALTH_CHECK_"):
            continue
        if event.get("event") != "request_complete":
            continue
        max_new_tokens = event.get("max_new_tokens")
        if not isinstance(max_new_tokens, int):
            continue
        if max_new_tokens in warmup_max_new_tokens:
            continue
        if max_new_tokens < min_benchmark_max_new_tokens:
            continue
        rids.add(rid)
    return rids


def filtered_spans(
    exp_dir: Path,
    expected_output_len: int,
    filter_mode: str,
    warmup_max_new_tokens: set[int],
    min_benchmark_max_new_tokens: int,
) -> list[Span]:
    events = load_events(exp_dir)
    if filter_mode == "eos":
        keep_rids = eos_cap_request_ids(
            events,
            warmup_max_new_tokens=warmup_max_new_tokens,
            min_benchmark_max_new_tokens=min_benchmark_max_new_tokens,
        )
    else:
        keep_rids = fixed_output_request_ids(events, expected_output_len)
    events = [event for event in events if event.get("rid") in keep_rids]
    return pair_spans(events)


def plot_lifecycle_axes(
    ax,
    spans: list[Span],
    title: str,
    max_requests: int,
) -> None:
    if not spans:
        ax.text(0.5, 0.5, "no successful trace", ha="center", va="center")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    lifecycle_spans = collapse_lifecycle_spans(spans)
    base = min(span.start for span in lifecycle_spans)
    ordered = request_order(lifecycle_spans)[:max_requests]
    y_index = {rid: idx for idx, rid in enumerate(ordered)}

    for phase, color in PHASE_COLORS.items():
        segments = []
        for span in lifecycle_spans:
            if span.phase != phase or span.rid not in y_index:
                continue
            y = y_index[span.rid]
            segments.append([(span.start - base, y), (span.end - base, y)])
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linewidths=2.2,
                    capstyle="butt",
                )
            )

    first_decode = min(
        (span.start for span in lifecycle_spans if span.phase == "decode"),
        default=None,
    )
    if first_decode is not None:
        ax.axvline(first_decode - base, color="#111827", linestyle="--", linewidth=0.8)

    x_max = max(span.end - base for span in lifecycle_spans)
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(-1, len(ordered))
    ax.set_title(title, fontsize=8)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.18)


def experiment_index(results_dir: Path, model_label: str) -> dict[tuple[str, int, int, str], Path]:
    index = {}
    for exp_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metadata = load_metadata(exp_dir)
        if metadata.get("model_label") != model_label:
            continue
        if not (exp_dir / "bench_serving.jsonl").exists():
            continue
        key = (
            metadata.get("dataset_category"),
            int(metadata.get("batch_size")),
            int(metadata.get("tp_size")),
            metadata.get("precision"),
        )
        index[key] = exp_dir
    return index


def write_grid(
    results_dir: Path,
    output_dir: Path,
    model_label: str,
    category: str,
    batch_sizes: list[int],
    tp_sizes: list[int],
    precisions: list[str],
    fallback_output_len: int,
    max_requests: int,
    filter_mode: str,
    warmup_max_new_tokens: set[int],
    min_benchmark_max_new_tokens: int,
) -> Path:
    import matplotlib.pyplot as plt

    index = experiment_index(results_dir, model_label)
    ncols = len(tp_sizes) * len(precisions)
    nrows = len(batch_sizes)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.4 * nrows),
        sharex=False,
        squeeze=False,
    )

    for row, batch_size in enumerate(batch_sizes):
        for tp_col, tp_size in enumerate(tp_sizes):
            for prec_col, precision in enumerate(precisions):
                col = tp_col * len(precisions) + prec_col
                ax = axes[row][col]
                exp_dir = index.get((category, batch_size, tp_size, precision))
                title = f"bs={batch_size} tp={tp_size} {precision}"
                if exp_dir is None:
                    plot_lifecycle_axes(ax, [], title, max_requests)
                    continue
                expected_output_len = benchmark_output_len(exp_dir, fallback_output_len)
                spans = filtered_spans(
                    exp_dir=exp_dir,
                    expected_output_len=expected_output_len,
                    filter_mode=filter_mode,
                    warmup_max_new_tokens=warmup_max_new_tokens,
                    min_benchmark_max_new_tokens=min_benchmark_max_new_tokens,
                )
                plot_lifecycle_axes(ax, spans, title, max_requests)
                ax.text(
                    0.01,
                    0.92,
                    f"n={len(request_order(spans))}",
                    transform=ax.transAxes,
                    fontsize=7,
                    ha="left",
                    va="top",
                    color="#374151",
                )
        axes[row][0].set_ylabel(category)

    handles = [
        plt.Line2D([0], [0], color=color, lw=5, label=phase)
        for phase, color in PHASE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3)
    fig.suptitle(
        f"{model_label} lifecycle bars, filtered to benchmark requests only",
        y=0.995,
        fontsize=12,
    )
    fig.supxlabel("time since first benchmark request event (s)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_label}_{category}_lifecycle_grid.png"
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def write_manifest(output_dir: Path, paths: list[Path], filter_mode: str) -> None:
    filter_note = (
        "EOS-cap mode keeps non-health completed requests whose `max_new_tokens` "
        "is not a warmup cap."
        if filter_mode == "eos"
        else "Fixed-output mode keeps requests whose completion event has "
        "`max_new_tokens == sharegpt_output_len`."
    )
    lines = [
        "# Lifecycle Grid Plots",
        "",
        "These plots filter out server warmup, `HEALTH_CHECK_*`, and benchmark warmup requests.",
        filter_note,
        "",
    ]
    for path in paths:
        lines.append(f"- `{path}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="/data/huanchen/sglang/rollout_precision_data/main_moe",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/huanchen/sglang/rollout_precision_data/glm_lifecycle_grids",
    )
    parser.add_argument("--model-label", default="glm_moe_30b")
    parser.add_argument("--categories", nargs="+", default=["math", "reasoning", "agentic"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--precisions", nargs="+", default=["bf16", "int4"])
    parser.add_argument("--fallback-output-len", type=int, default=256)
    parser.add_argument("--max-requests-per-cell", type=int, default=512)
    parser.add_argument(
        "--filter-mode",
        choices=["fixed_output", "eos"],
        default="fixed_output",
        help="How to identify real benchmark requests in the scheduler trace.",
    )
    parser.add_argument(
        "--warmup-max-new-tokens",
        nargs="+",
        type=int,
        default=[1, 8, 32],
        help="EOS mode: completion caps treated as health/warmup requests.",
    )
    parser.add_argument(
        "--min-benchmark-max-new-tokens",
        type=int,
        default=64,
        help="EOS mode: minimum request cap for benchmark prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    paths = []
    for category in args.categories:
        paths.append(
            write_grid(
                results_dir=results_dir,
                output_dir=output_dir,
                model_label=args.model_label,
                category=category,
                batch_sizes=args.batch_sizes,
                tp_sizes=args.tp_sizes,
                precisions=args.precisions,
                fallback_output_len=args.fallback_output_len,
                max_requests=args.max_requests_per_cell,
                filter_mode=args.filter_mode,
                warmup_max_new_tokens=set(args.warmup_max_new_tokens),
                min_benchmark_max_new_tokens=args.min_benchmark_max_new_tokens,
            )
        )
    write_manifest(output_dir, paths, args.filter_mode)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
