#!/usr/bin/env python3
"""Render QLoRA kernel-latency line plots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = OUT_DIR / "qlora_kernel_perf_clean.json"
DEFAULT_FAKE_DATA = OUT_DIR / "qlora_kernel_perf_fake.json"
DEFAULT_OUTPUT = OUT_DIR / "qlora_kernel_perf_clean.png"


COLORS = {
    "bf16": "#4C78A8",
    "int4": "#F28E2B",
    "bf16 + sequential csgmv LoRA": "#4C78A8",
    "bf16 + two-stream Torch LoRA": "#9467BD",
    "int4 + sequential csgmv LoRA": "#F28E2B",
    "int4 + two-stream Torch LoRA": "#E15759",
    "int4 + sequential Torch LoRA": "#EDC948",
    "bf16 + sequential Torch LoRA": "#9467BD",
    "bf16 dense base": "#4C78A8",
    "int4 Marlin base": "#F28E2B",
    "SGLang csgmv LoRA patch": "#B07AA1",
    "SGLang triton LoRA patch": "#9C755F",
    "Torch matmul LoRA patch": "#76B7B2",
    "bf16 dense + csgmv sequential": "#4C78A8",
    "bf16 dense + triton sequential": "#9467BD",
    "bf16 dense + torch matmul sequential": "#9467BD",
    "SGLang QLoRA csgmv sequential": "#F28E2B",
    "SGLang QLoRA csgmv two-stream": "#17BECF",
    "SGLang QLoRA triton sequential": "#E15759",
    "SGLang QLoRA triton two-stream": "#E15759",
    "Torch QLoRA matmul sequential": "#EDC948",
    "Torch QLoRA matmul two-stream": "#86BCB6",
}

LINESTYLES = {
    "bf16": "-",
    "int4": "-",
    "bf16 + sequential csgmv LoRA": "--",
    "bf16 + two-stream Torch LoRA": "--",
    "int4 + sequential csgmv LoRA": "--",
    "int4 + two-stream Torch LoRA": "--",
    "int4 + sequential Torch LoRA": "--",
    "bf16 + sequential Torch LoRA": "--",
    "bf16 dense base": "-",
    "int4 Marlin base": "-",
    "bf16 dense + csgmv sequential": "--",
    "bf16 dense + triton sequential": "--",
    "bf16 dense + torch matmul sequential": "--",
    "SGLang QLoRA csgmv sequential": "--",
    "SGLang QLoRA triton two-stream": "--",
    "Torch QLoRA matmul sequential": "--",
    "Torch QLoRA matmul two-stream": "--",
}

MARKERS = {
    "bf16": "o",
    "int4": "s",
    "bf16 + sequential csgmv LoRA": "h",
    "bf16 + two-stream Torch LoRA": "X",
    "int4 + sequential csgmv LoRA": "^",
    "int4 + two-stream Torch LoRA": "d",
    "int4 + sequential Torch LoRA": "<",
    "bf16 + sequential Torch LoRA": "X",
    "bf16 dense base": "o",
    "int4 Marlin base": "s",
    "SGLang csgmv LoRA patch": "P",
    "SGLang triton LoRA patch": "X",
    "Torch matmul LoRA patch": "*",
    "bf16 dense + csgmv sequential": "h",
    "bf16 dense + triton sequential": "X",
    "bf16 dense + torch matmul sequential": "X",
    "SGLang QLoRA csgmv sequential": "^",
    "SGLang QLoRA csgmv two-stream": "D",
    "SGLang QLoRA triton sequential": "v",
    "SGLang QLoRA triton two-stream": "d",
    "Torch QLoRA matmul sequential": "<",
    "Torch QLoRA matmul two-stream": ">",
}


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def successful_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload["measurements"]:
        if row.get("error"):
            continue
        latency = row.get("median_us", row.get("latency_us"))
        if latency is None:
            continue
        rows.append(row)
    return rows


def projection_order(payload: dict) -> list[str]:
    configured = []
    for model_config in payload["metadata"].get("model_configs", []):
        for projection in model_config.get("projections", []):
            name = projection["name"]
            if name not in configured:
                configured.append(name)
    return configured or ["qkv", "o", "gate_up", "down"]


def annotate_ratio(ax, values_by_scheme: dict[str, list[tuple[int, float]]], token: int) -> None:
    if len(values_by_scheme) != 2:
        return
    schemes = list(values_by_scheme)
    bf16_scheme = next((scheme for scheme in schemes if scheme.startswith("bf16")), None)
    int4_scheme = next((scheme for scheme in schemes if scheme.startswith("int4")), None)
    if bf16_scheme is None or int4_scheme is None:
        return

    bf16_values = dict(values_by_scheme[bf16_scheme])
    int4_values = dict(values_by_scheme[int4_scheme])
    if token not in bf16_values or token not in int4_values:
        return

    bf16_y = bf16_values[token]
    int4_y = int4_values[token]
    if bf16_y <= 0 or int4_y <= 0:
        return

    ratio = bf16_y / int4_y
    y_low = min(bf16_y, int4_y)
    y_high = max(bf16_y, int4_y)
    text_y = (y_low * y_high) ** 0.5
    ax.vlines(token, y_low, y_high, colors="#222222", linewidth=1.1, alpha=0.75)
    ax.scatter([token, token], [y_low, y_high], s=16, color="#222222", zorder=5)
    ax.annotate(
        f"x{ratio:.1f}",
        xy=(token, text_y),
        xytext=(7, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#222222",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#777777", "alpha": 0.92},
    )


def render_projection_grid(
    payload: dict,
    output_path: Path,
    *,
    log_y: bool = True,
    annotate_ratio_token: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    metadata = payload["metadata"]
    scheme_order = metadata["scheme_order"]
    rows = successful_rows(payload)
    if not rows:
        raise ValueError("No successful measurements found")

    grouped: dict[tuple[str, str], dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: {scheme: [] for scheme in scheme_order}
    )
    model_order = []
    for row in rows:
        model = row.get("model") or "model"
        projection = row.get("projection") or "projection"
        if model not in model_order:
            model_order.append(model)
        grouped[(model, projection)][row["scheme"]].append(
            (row["token_rows"], row.get("median_us", row["latency_us"]))
        )

    projections = [
        projection
        for projection in projection_order(payload)
        if any((model, projection) in grouped for model in model_order)
    ]
    fig, axes = plt.subplots(
        len(model_order),
        len(projections),
        figsize=(max(12.0, 2.55 * len(projections)), max(5.0, 2.7 * len(model_order))),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for row_idx, model in enumerate(model_order):
        for col_idx, projection in enumerate(projections):
            ax = axes[row_idx][col_idx]
            series = grouped.get((model, projection))
            if not series:
                ax.axis("off")
                continue
            all_x = []
            plotted_values = {}
            for scheme in scheme_order:
                values = sorted(series[scheme], key=lambda item: item[0])
                if not values:
                    continue
                plotted_values[scheme] = values
                xs = [x for x, _ in values]
                ys = [y for _, y in values]
                all_x.extend(xs)
                ax.plot(
                    xs,
                    ys,
                    label=scheme,
                    color=COLORS[scheme],
                    linestyle=LINESTYLES.get(scheme, "-"),
                    marker=MARKERS[scheme],
                    linewidth=1.8,
                    markersize=4.5,
                )
            ax.set_title(f"{model} / {projection}", fontsize=9)
            ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log", base=10)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(True, which="both", alpha=0.22)
            if all_x:
                xticks = sorted(set(all_x))
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(x) for x in xticks], fontsize=7)
            if col_idx == 0:
                ax.set_ylabel("us", fontsize=8)
            if row_idx == len(model_order) - 1:
                ax.set_xlabel("tokens", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            if annotate_ratio_token is not None:
                annotate_ratio(ax, plotted_values, annotate_ratio_token)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncols=min(4, len(labels)),
            fontsize=9,
        )

    cfg = metadata["config"]
    ranks = sorted(
        {
            str(model_config.get("lora_rank", cfg["lora_rank"]))
            for model_config in metadata.get("model_configs", [])
        }
    )
    rank_label = ",".join(ranks) if ranks else str(cfg["lora_rank"])
    fig.suptitle(
        "SGLang QLoRA Kernel Latency by Model Projection\n"
        f"rank={rank_label}, quant={cfg['quant_type']}, warmup={cfg['warmup_iters']}, "
        f"iters={cfg['measure_iters']}, L2 flush={cfg['l2_flush_mib']} MiB",
        fontsize=12,
        y=0.955,
    )
    fig.text(
        0.01,
        0.012,
        "Uses torch.compile, CUDA Graph replay, activation prewarm, and L2 flush. LoRA backend is indicated by each line label.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_single_axis(payload: dict, output_path: Path, *, log_y: bool = True) -> None:
    import matplotlib.pyplot as plt

    metadata = payload["metadata"]
    scheme_order = metadata["scheme_order"]
    series = {scheme: [] for scheme in scheme_order}
    for row in successful_rows(payload):
        series[row["scheme"]].append((row["token_rows"], row.get("median_us", row["latency_us"])))
    for values in series.values():
        values.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    for scheme, values in series.items():
        if not values:
            continue
        ax.plot(
            [x for x, _ in values],
            [y for _, y in values],
            label=scheme,
            color=COLORS[scheme],
            linestyle=LINESTYLES.get(scheme, "-"),
            marker=MARKERS[scheme],
            linewidth=2.4,
            markersize=6.5,
        )

    all_x = [x for values in series.values() for x, _ in values]
    if not all_x:
        raise ValueError("No successful measurements found")

    cfg = metadata["config"]
    base = cfg.get("base_quant") or cfg.get("quant_type", "int4_marlin")
    title_kind = "Mock" if metadata.get("fake") else "Measured"
    ylabel_kind = "synthetic" if metadata.get("fake") else "CUDA Graph median"
    ax.set_title(
        f"{title_kind} SGLang QLoRA Single-Linear Kernel Latency\n"
        f"H={cfg['hidden_size']}, O={cfg['output_size']}, rank={cfg['lora_rank']}, base={base}"
    )
    ax.set_xlabel("Token rows / decode batch size")
    ax.set_ylabel(f"Latency per linear forward (us, {ylabel_kind})")
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log", base=10)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticks(sorted(set(all_x)))
    ax.set_xticklabels([str(x) for x in sorted(set(all_x))])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.text(
        0.01,
        0.012,
        "LoRA backend is indicated by each line label.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_matplotlib(
    data_path: Path,
    output_path: Path,
    *,
    log_y: bool = True,
    annotate_ratio_token: int | None = None,
) -> None:
    payload = load_payload(data_path)
    if not payload["metadata"].get("fake") and any(row.get("projection") for row in payload["measurements"]):
        render_projection_grid(
            payload,
            output_path,
            log_y=log_y,
            annotate_ratio_token=annotate_ratio_token,
        )
    else:
        render_single_axis(payload, output_path, log_y=log_y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--linear-y", action="store_true", help="Use a linear y-axis.")
    parser.add_argument(
        "--annotate-ratio-token",
        type=int,
        default=None,
        help="Annotate bf16/int4 latency ratio at this token row for two-line plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data
    if data_path is None:
        data_path = DEFAULT_DATA if DEFAULT_DATA.exists() else DEFAULT_FAKE_DATA
    render_matplotlib(
        data_path,
        args.output,
        log_y=not args.linear_y,
        annotate_ratio_token=args.annotate_ratio_token,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
