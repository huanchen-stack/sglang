"""Plot scheduler rollout traces and write a precision comparison report."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASE_COLORS = {
    "waiting": "#9ca3af",
    "prefill": "#2563eb",
    "decode": "#f59e0b",
}


@dataclass
class Span:
    rid: str
    phase: str
    start: float
    end: float
    batch_size: int | None
    batch_id: int | None
    forward_mode: str | None
    input_len: int
    output_len: int
    can_run_cuda_graph: bool | None

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"Skipping malformed trace line {path}:{lineno}: {exc}",
                        file=sys.stderr,
                    )
    return rows


def load_experiment(exp_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = exp_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    events = []
    for path in sorted((exp_dir / "traces").glob("*.jsonl")):
        events.extend(read_jsonl(path))
    events.sort(key=lambda row: row.get("ts", 0.0))
    return metadata, events


def pair_spans(events: list[dict[str, Any]]) -> list[Span]:
    starts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    spans = []
    for event in events:
        rid = event.get("rid")
        phase = event.get("phase")
        if not rid or phase not in PHASE_COLORS:
            continue
        key = (rid, phase)
        if event.get("event") == "phase_start":
            starts[key].append(event)
        elif event.get("event") == "phase_end" and starts[key]:
            start = starts[key].pop()
            spans.append(
                Span(
                    rid=rid,
                    phase=phase,
                    start=start["ts"],
                    end=event["ts"],
                    batch_size=event.get("batch_size", start.get("batch_size")),
                    batch_id=event.get("batch_id", start.get("batch_id")),
                    forward_mode=event.get("forward_mode", start.get("forward_mode")),
                    input_len=event.get("input_len", start.get("input_len", 0)),
                    output_len=event.get("output_len", start.get("output_len", 0)),
                    can_run_cuda_graph=event.get("can_run_cuda_graph"),
                )
            )
    return spans


def request_order(spans: list[Span]) -> list[str]:
    first_ts = {}
    for span in spans:
        first_ts[span.rid] = min(first_ts.get(span.rid, span.start), span.start)
    return [rid for rid, _ in sorted(first_ts.items(), key=lambda item: item[1])]


def live_counts(spans: list[Span], phase: str) -> tuple[list[float], list[int]]:
    points = []
    for span in spans:
        if span.phase == phase:
            points.append((span.start, 1))
            points.append((span.end, -1))
    points.sort()
    xs = []
    ys = []
    cur = 0
    for ts, delta in points:
        cur += delta
        xs.append(ts)
        ys.append(cur)
    return xs, ys


def collapse_lifecycle_spans(spans: list[Span]) -> list[Span]:
    """Collapse per-step scheduler spans into one visual span per request phase."""
    grouped: dict[tuple[str, str], list[Span]] = defaultdict(list)
    for span in spans:
        grouped[(span.rid, span.phase)].append(span)

    collapsed = []
    for (rid, phase), phase_spans in grouped.items():
        first = min(phase_spans, key=lambda span: span.start)
        last = max(phase_spans, key=lambda span: span.end)
        collapsed.append(
            Span(
                rid=rid,
                phase=phase,
                start=first.start,
                end=last.end,
                batch_size=last.batch_size,
                batch_id=last.batch_id,
                forward_mode=last.forward_mode,
                input_len=first.input_len,
                output_len=last.output_len,
                can_run_cuda_graph=last.can_run_cuda_graph,
            )
        )
    return collapsed


def plot_experiment(
    exp_dir: Path,
    metadata: dict[str, Any],
    spans: list[Span],
    output_path: Path,
    max_requests: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not spans:
        return
    base = min(span.start for span in spans)
    lifecycle_spans = collapse_lifecycle_spans(spans)
    ordered = request_order(lifecycle_spans)[:max_requests]
    y_index = {rid: idx for idx, rid in enumerate(ordered)}

    fig, (ax, live_ax) = plt.subplots(
        2,
        1,
        figsize=(16, max(7, min(24, len(ordered) * 0.22))),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

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
                    linewidths=5.0,
                    capstyle="butt",
                )
            )
    ax.set_ylim(-1, len(ordered))

    first_decode = min(
        (span.start for span in spans if span.phase == "decode"),
        default=None,
    )
    if first_decode is not None:
        ax.axvline(
            first_decode - base,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label="first decode",
        )

    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=phase)
        for phase, color in PHASE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_ylabel("request")
    ax.set_yticks([])
    ax.set_title(
        " ".join(
            [
                metadata.get("name", exp_dir.name),
                metadata.get("precision", ""),
                f"bs={metadata.get('batch_size', '')}",
                f"tp={metadata.get('tp_size', '')}",
            ]
        )
    )

    for phase, color in (("prefill", PHASE_COLORS["prefill"]), ("decode", PHASE_COLORS["decode"])):
        xs, ys = live_counts(spans, phase)
        if xs:
            live_ax.step([x - base for x in xs], ys, where="post", color=color, label=phase)
    live_ax.set_ylabel("live")
    live_ax.set_xlabel("time since first event (s)")
    live_ax.legend(loc="upper right")
    live_ax.grid(True, axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def summarize_spans(spans: list[Span]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in PHASE_COLORS:
        durations = [span.duration_ms for span in spans if span.phase == phase]
        if durations:
            summary[f"{phase}_median_ms"] = statistics.median(durations)
            summary[f"{phase}_p90_ms"] = statistics.quantiles(durations, n=10)[8]
            summary[f"{phase}_count"] = len(durations)

    decode_by_bs: dict[int, list[float]] = defaultdict(list)
    for span in spans:
        if span.phase == "decode" and span.batch_size is not None:
            decode_by_bs[int(span.batch_size)].append(span.duration_ms)
    summary["decode_median_ms_by_batch_size"] = {
        bs: statistics.median(values) for bs, values in sorted(decode_by_bs.items())
    }
    cuda_values = [
        span.can_run_cuda_graph
        for span in spans
        if span.phase == "decode" and span.can_run_cuda_graph is not None
    ]
    if cuda_values:
        summary["decode_cuda_graph_eligible_fraction"] = sum(cuda_values) / len(cuda_values)
    return summary


def experiment_key(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        metadata.get("model_label"),
        metadata.get("dataset_category"),
        metadata.get("batch_size"),
        metadata.get("tp_size"),
    )


def write_report(
    report_path: Path,
    rows: list[tuple[Path, dict[str, Any], dict[str, Any], Path]],
    failures: list[tuple[Path, dict[str, Any], str]],
    min_decode_cuda_graph_eligible: float,
) -> None:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for _, metadata, summary, _ in rows:
        by_key[experiment_key(metadata)][metadata.get("precision", "unknown")] = summary

    lines = ["# Rollout Precision Report", ""]
    lines.append("## Experiments")
    lines.append("")
    lines.append(
        "| experiment | precision | prefill median ms | decode median ms | "
        "decode CUDA graph eligible | plot |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for exp_dir, metadata, summary, plot_path in rows:
        plot = plot_path.name if plot_path.exists() else "n/a"
        lines.append(
            "| {name} | {precision} | {prefill:.3f} | {decode:.3f} | {cuda} | {plot} |".format(
                name=metadata.get("name", exp_dir.name),
                precision=metadata.get("precision", ""),
                prefill=summary.get("prefill_median_ms", float("nan")),
                decode=summary.get("decode_median_ms", float("nan")),
                cuda=(
                    f"{summary['decode_cuda_graph_eligible_fraction']:.3f}"
                    if "decode_cuda_graph_eligible_fraction" in summary
                    else "n/a"
                ),
                plot=plot,
            )
        )

    if failures:
        lines.extend(["", "## Failed Experiments", ""])
        lines.append("| experiment | precision | reason |")
        lines.append("|---|---:|---|")
        for exp_dir, metadata, failure in failures:
            reason = next(
                (line.strip() for line in reversed(failure.splitlines()) if line.strip()),
                "unknown failure",
            )
            reason = reason.replace("|", "\\|")
            lines.append(
                "| {name} | {precision} | {reason} |".format(
                    name=metadata.get("name", exp_dir.name),
                    precision=metadata.get("precision", ""),
                    reason=reason,
                )
            )

    lines.extend(["", "## bf16 vs int4 Decode Crossover", ""])
    lines.append(
        "| model | category | batch | tp | live decode batch sizes where int4 is faster | note |"
    )
    lines.append("|---|---|---:|---:|---|---|")
    for key, precisions in sorted(by_key.items()):
        bf16 = precisions.get("bf16")
        int4 = precisions.get("int4")
        if not bf16 or not int4:
            continue
        bf16_by_bs = bf16.get("decode_median_ms_by_batch_size", {})
        int4_by_bs = int4.get("decode_median_ms_by_batch_size", {})
        faster = []
        for bs, int4_ms in int4_by_bs.items():
            bf16_ms = bf16_by_bs.get(bs)
            if bf16_ms is not None and int4_ms < bf16_ms:
                faster.append(str(bs))
        note = (
            "candidate mid-decode switch"
            if faster
            else "naive prefill/decode split only supported by this trace"
        )
        model, category, batch, tp = key
        lines.append(
            f"| {model} | {category} | {batch} | {tp} | {', '.join(faster) or 'none'} | {note} |"
        )

    lines.extend(
        [
            "",
            "## CUDA Graph Guardrail",
            "",
            "The instrumentation does not disable CUDA graph. Any experiment with a low "
            "`decode CUDA graph eligible` fraction should be treated as a regression "
            "candidate and inspected before drawing precision conclusions.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    failed = []
    for exp_dir, metadata, summary, _ in rows:
        value = summary.get("decode_cuda_graph_eligible_fraction")
        if value is not None and value < min_decode_cuda_graph_eligible:
            failed.append((metadata.get("name", exp_dir.name), value))
    if failed:
        details = ", ".join(f"{name}={value:.3f}" for name, value in failed)
        raise RuntimeError(
            "Decode CUDA graph eligibility below threshold "
            f"{min_decode_cuda_graph_eligible:.3f}: {details}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-requests-per-plot", type=int, default=512)
    parser.add_argument(
        "--min-decode-cuda-graph-eligible",
        type=float,
        default=0.0,
        help="Fail if any experiment has a lower decode CUDA graph eligible fraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    report_path = Path(args.report)
    rows = []
    failures = []
    for exp_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metadata, events = load_experiment(exp_dir)
        metadata.setdefault("name", exp_dir.name)
        failure_path = exp_dir / "failure.txt"
        if failure_path.exists():
            failures.append(
                (
                    exp_dir,
                    metadata,
                    failure_path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        spans = pair_spans(events)
        summary = summarize_spans(spans)
        plot_path = exp_dir / "lifecycle.png"
        plot_experiment(
            exp_dir,
            metadata,
            spans,
            plot_path,
            max_requests=args.max_requests_per_plot,
        )
        rows.append((exp_dir, metadata, summary, plot_path))

    write_report(
        report_path,
        rows,
        failures,
        min_decode_cuda_graph_eligible=args.min_decode_cuda_graph_eligible,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
