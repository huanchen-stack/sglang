"""Plot decode-only throughput for LoRA forwarding serving experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


MODE_LABELS = {
    "bf16_full": "BF16 weights",
    "marlin_int4_full": "INT4 Marlin weights",
    "qlora_parallel_streams": "INT4 weights + BF16 LoRA patch",
    "qlora_marlin_bf16": "INT4 weights + BF16 LoRA patch",
    "oracle": "Oracle precision switch",
}

COLORS = {
    "bf16_full": "#4C78A8",
    "marlin_int4_full": "#59A14F",
    "qlora_parallel_streams": "#17BECF",
    "qlora_marlin_bf16": "#17BECF",
    "oracle": "#D62728",
}

MARKERS = {
    "bf16_full": "o",
    "marlin_int4_full": "s",
    "qlora_parallel_streams": "^",
    "qlora_marlin_bf16": "^",
    "oracle": "D",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_results(rollout_dir: Path, throughput_key: str) -> list[dict[str, Any]]:
    results = []
    for exp_dir in sorted(rollout_dir.iterdir() if rollout_dir.exists() else []):
        if not exp_dir.is_dir():
            continue
        metadata_path = exp_dir / "metadata.json"
        decode_path = exp_dir / "decode_summary.json"
        validation_path = exp_dir / "forced_graph_validation.json"
        failure_path = exp_dir / "failure.txt"
        if not metadata_path.exists() or not decode_path.exists() or failure_path.exists():
            continue
        metadata = load_json(metadata_path)
        decode = load_json(decode_path)
        validation = load_json(validation_path) if validation_path.exists() else {}
        throughput = decode.get(throughput_key)
        if throughput is None:
            continue
        row = {
            **metadata,
            **{f"decode_{k}": v for k, v in decode.items()},
            "throughput": throughput,
            "validation_ok": validation.get("ok"),
            "exp_dir": str(exp_dir),
        }
        results.append(row)
    return results


def choose_qlora_mode(rows: list[dict[str, Any]]) -> str | None:
    modes = {row["mode"] for row in rows}
    if "qlora_parallel_streams" in modes:
        return "qlora_parallel_streams"
    if "qlora_marlin_bf16" in modes:
        return "qlora_marlin_bf16"
    return None


def fmt_tps(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def plot_group(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model_label: str,
    dataset_category: str,
) -> None:
    qlora_mode = choose_qlora_mode(rows)
    modes = ["bf16_full", "marlin_int4_full"]
    if qlora_mode is not None:
        modes.append(qlora_mode)

    by_key = {(row["mode"], row["batch_size"]): row for row in rows}
    batch_sizes = sorted({row["batch_size"] for row in rows})
    if not batch_sizes:
        return

    oracle_values: list[tuple[int, float, str, float | None, float | None]] = []
    for batch_size in batch_sizes:
        bf16 = by_key.get(("bf16_full", batch_size), {}).get("throughput")
        qlora = (
            by_key.get((qlora_mode, batch_size), {}).get("throughput")
            if qlora_mode is not None
            else None
        )
        candidates = []
        if bf16 is not None:
            candidates.append(("bf16_full", bf16))
        if qlora is not None:
            candidates.append((qlora_mode, qlora))
        if not candidates:
            continue
        chosen_mode, chosen_value = max(candidates, key=lambda item: item[1])
        oracle_values.append((batch_size, chosen_value, chosen_mode, bf16, qlora))

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for mode in modes:
        xs = [b for b in batch_sizes if (mode, b) in by_key]
        ys = [by_key[(mode, b)]["throughput"] for b in xs]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker=MARKERS[mode],
            color=COLORS[mode],
            linewidth=2.4,
            markersize=7,
            label=MODE_LABELS[mode],
        )
        for x, y in zip(xs, ys):
            ax.annotate(
                fmt_tps(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=COLORS[mode],
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            )

    if oracle_values:
        xs = [item[0] for item in oracle_values]
        ys = [item[1] for item in oracle_values]
        ax.plot(
            xs,
            ys,
            marker=MARKERS["oracle"],
            color=COLORS["oracle"],
            linewidth=2.8,
            linestyle="--",
            markersize=7,
            label=MODE_LABELS["oracle"],
            zorder=5,
        )
        for x, y, chosen_mode, bf16, qlora in oracle_values:
            parts = [fmt_tps(y), MODE_LABELS[chosen_mode].split()[0]]
            if bf16:
                parts.append(f"{y / bf16:.2f}x BF16")
            if qlora:
                parts.append(f"{y / qlora:.2f}x QLoRA")
            ax.annotate(
                "\n".join(parts),
                (x, y),
                textcoords="offset points",
                xytext=(0, -42 if y > 0 else 10),
                ha="center",
                fontsize=7,
                color=COLORS["oracle"],
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=COLORS["oracle"], alpha=0.9),
            )

    ax.set_title(f"Decode Throughput vs Batch Size - {model_label} / {dataset_category}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode throughput (output tokens / second, log scale)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(x) for x in batch_sizes])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: fmt_tps(y)))
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{model_label}_{dataset_category}".replace(".", "_").replace("-", "_")
    fig.savefig(output_dir / f"decode_throughput_{slug}.png", dpi=220)
    fig.savefig(output_dir / f"decode_throughput_{slug}.pdf")
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], output_dir: Path, throughput_key: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "decode_throughput_summary.csv"
    fields = [
        "model_label",
        "dataset_category",
        "batch_size",
        "mode",
        "throughput_key",
        "throughput",
        "decode_decode_completion_tokens",
        "decode_decode_wall_s",
        "decode_decode_forward_tokens_per_s",
        "decode_decode_cuda_graph_fraction",
        "validation_ok",
        "exp_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            out["throughput_key"] = throughput_key
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--throughput-key",
        default="decode_wall_tokens_per_s",
        choices=["decode_wall_tokens_per_s", "decode_forward_tokens_per_s"],
    )
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    output_dir = Path(args.output_dir)
    rows = load_results(rollout_dir, args.throughput_key)
    write_csv(rows, output_dir, args.throughput_key)

    groups = sorted({(row["model_label"], row["dataset_category"]) for row in rows})
    for model_label, dataset_category in groups:
        group_rows = [
            row
            for row in rows
            if row["model_label"] == model_label
            and row["dataset_category"] == dataset_category
        ]
        plot_group(group_rows, output_dir, model_label, dataset_category)
    print(f"Wrote decode throughput plots to {output_dir}")


if __name__ == "__main__":
    main()
