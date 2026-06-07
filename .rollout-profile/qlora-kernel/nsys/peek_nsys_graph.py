#!/usr/bin/env python3
"""Render a compact Nsight Systems CUDA-kernel peek from an exported SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def fetch_kernels(sqlite_path: Path) -> list[dict]:
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        select
          k.start,
          k.end,
          k.streamId,
          coalesce(s.value, cast(k.shortName as text)) as name,
          k.gridX,
          k.gridY,
          k.gridZ,
          k.blockX,
          k.blockY,
          k.blockZ,
          k.graphNodeId
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on k.shortName = s.id
        order by k.start
        """
    ).fetchall()
    con.close()
    kernels = []
    for row in rows:
        item = dict(row)
        item["duration_us"] = (item["end"] - item["start"]) / 1000.0
        kernels.append(item)
    return kernels


def summarize(kernels: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    samples: dict[tuple, dict] = {}
    for kernel in kernels:
        key = (
            kernel["name"],
            kernel["streamId"],
            kernel["gridX"],
            kernel["gridY"],
            kernel["gridZ"],
            kernel["blockX"],
            kernel["blockY"],
            kernel["blockZ"],
        )
        grouped[key].append(kernel["duration_us"])
        samples[key] = kernel

    rows = []
    for key, durations in grouped.items():
        sample = samples[key]
        rows.append(
            {
                "name": sample["name"],
                "stream": sample["streamId"],
                "count": len(durations),
                "median_us": statistics.median(durations),
                "min_us": min(durations),
                "max_us": max(durations),
                "mean_us": statistics.fmean(durations),
                "grid": [sample["gridX"], sample["gridY"], sample["gridZ"]],
                "block": [sample["blockX"], sample["blockY"], sample["blockZ"]],
            }
        )
    rows.sort(key=lambda item: item["median_us"], reverse=True)
    return rows


def pick_replay(
    kernels: list[dict],
    replay_index: int,
    *,
    anchor_substring: str,
    window_us: float | None,
) -> list[dict]:
    anchors = [k for k in kernels if anchor_substring in k["name"]]
    if not anchors:
        raise ValueError(f"Could not find anchors containing {anchor_substring!r}")
    if replay_index < 0:
        replay_index = len(anchors) + replay_index
    if replay_index < 0 or replay_index >= len(anchors):
        raise ValueError(f"replay index out of range: {replay_index}")

    start = anchors[replay_index]["start"]
    if window_us is not None:
        end = start + int(window_us * 1000)
    elif replay_index + 1 < len(anchors):
        end = anchors[replay_index + 1]["start"]
    else:
        end = start + 250_000

    selected = [
        kernel
        for kernel in kernels
        if kernel["start"] >= start - 10_000 and kernel["start"] < end
    ]
    if not selected:
        raise ValueError("No kernels selected for replay window")
    return selected


def draw_timeline(
    selected: list[dict],
    summary: list[dict],
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    streams = sorted({kernel["streamId"] for kernel in selected})
    stream_to_y = {stream: idx for idx, stream in enumerate(streams)}
    t0 = min(kernel["start"] for kernel in selected)
    t1 = max(kernel["end"] for kernel in selected)

    colors = {
        "Marlin": "#E15759",
        "_sgemm_lora_a_kernel": "#4C78A8",
        "_sgemm_lora_b_kernel": "#59A14F",
        "vectorized_elementwise_kernel": "#B07AA1",
    }

    fig, (ax, table_ax) = plt.subplots(
        2,
        1,
        figsize=(12.5, 6.8),
        gridspec_kw={"height_ratios": [2.0, 1.25]},
    )

    for kernel in selected:
        start_us = (kernel["start"] - t0) / 1000.0
        dur_us = kernel["duration_us"]
        y = stream_to_y[kernel["streamId"]]
        ax.broken_barh(
            [(start_us, dur_us)],
            (y - 0.34, 0.68),
            facecolors=colors.get(kernel["name"], "#9C755F"),
            edgecolors="#222222",
            linewidth=0.35,
        )
        label = kernel["name"].replace("_kernel", "")
        if dur_us >= 2.0:
            ax.text(
                start_us + dur_us / 2,
                y,
                f"{label}\n{dur_us:.1f} us",
                ha="center",
                va="center",
                fontsize=7,
                color="#111111",
            )

    ax.set_title(title)
    ax.set_xlabel("Time within selected replay (us)")
    ax.set_yticks(list(stream_to_y.values()))
    ax.set_yticklabels([f"stream {stream}" for stream in streams])
    ax.set_xlim(0, (t1 - t0) / 1000.0 * 1.04)
    ax.set_ylim(-0.8, len(streams) - 0.2)
    ax.grid(True, axis="x", alpha=0.25)

    table_ax.axis("off")
    top = summary[:6]
    cell_text = [
        [
            item["name"],
            item["stream"],
            item["count"],
            f"{item['median_us']:.3f}",
            f"{item['min_us']:.3f}",
            f"{item['max_us']:.3f}",
            f"{tuple(item['grid'])}",
            f"{tuple(item['block'])}",
        ]
        for item in top
    ]
    table = table_ax.table(
        cellText=cell_text,
        colLabels=["kernel", "stream", "n", "median us", "min us", "max us", "grid", "block"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--replay-index", type=int, default=-1)
    parser.add_argument("--anchor-substring", default="_sgemm_lora_a_kernel")
    parser.add_argument(
        "--window-us",
        type=float,
        default=None,
        help="Select this many microseconds after the chosen anchor instead of stopping at the next anchor.",
    )
    parser.add_argument("--title", default="Nsight Systems CUDA Graph Replay Peek")
    args = parser.parse_args()

    kernels = fetch_kernels(args.sqlite)
    summary = summarize(kernels)
    selected = pick_replay(
        kernels,
        args.replay_index,
        anchor_substring=args.anchor_substring,
        window_us=args.window_us,
    )
    draw_timeline(selected, summary, args.output, title=args.title)

    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(
                {
                    "sqlite": str(args.sqlite),
                    "kernel_count": len(kernels),
                    "selected_kernel_count": len(selected),
                    "summary": summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {args.output}")
    if args.summary_json:
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
