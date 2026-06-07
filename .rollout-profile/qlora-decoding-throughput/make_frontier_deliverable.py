#!/usr/bin/env python3
"""Create kernel-guided frontier deliverables for QLoRA decoding throughput.

Inputs:
- ``decoding_throughput.csv``: measured serving throughput for whole-model
  BF16 merged and whole-model int4 + BF16 LoRA Torch two-stream.
- ``../qlora-kernel/deliverable/torch_twostream_vs_bf16_to512.json``:
  per-projection kernel latencies for BF16 versus int4 two-stream LoRA.

The per-projection policy is selected from kernel latency. The mixed throughput
line is an estimate, not a serving measurement, because the current server does
not yet support per-projection mixed BF16/int4 weights in one model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SERVING_CSV = OUT_DIR / "decoding_throughput.csv"
KERNEL_JSON = (
    OUT_DIR.parent / "qlora-kernel" / "deliverable" / "torch_twostream_vs_bf16_to512.json"
)
MODEL = "qwen2.5-14b"
LAYERS = 48
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
PROJECTIONS = ["qkv", "o", "gate_up", "down"]
BF16 = "bf16"
QLORA = "int4 + two-stream Torch LoRA"


def apply_paper_style() -> None:
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


def load_serving_rows() -> dict[tuple[str, int], float]:
    measured: dict[tuple[str, int], float] = {}
    with SERVING_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["success"] != "True":
                continue
            scheme = row["scheme"]
            if scheme not in {"bf16_merged", "qlora_torch_twostream"}:
                continue
            measured[(scheme, int(row["batch_size"]))] = float(row["decode_tok_s"])
    return measured


def load_kernel_rows() -> dict[tuple[int, str], dict[str, float]]:
    payload = json.loads(KERNEL_JSON.read_text(encoding="utf-8"))
    rows: dict[tuple[int, str], dict[str, float]] = {}
    for row in payload["measurements"]:
        if row.get("model") != MODEL:
            continue
        scheme = row.get("scheme")
        if scheme not in {BF16, QLORA}:
            continue
        key = (int(row["token_rows"]), row["projection"])
        rows.setdefault(key, {})[scheme] = float(row["latency_us"])
    return rows


def require_inputs(
    serving: dict[tuple[str, int], float],
    kernel: dict[tuple[int, str], dict[str, float]],
) -> None:
    missing_serving = [
        (scheme, bs)
        for scheme in ["bf16_merged", "qlora_torch_twostream"]
        for bs in BATCH_SIZES
        if (scheme, bs) not in serving
    ]
    missing_kernel = [
        (bs, proj, scheme)
        for bs in BATCH_SIZES
        for proj in PROJECTIONS
        for scheme in [BF16, QLORA]
        if scheme not in kernel.get((bs, proj), {})
    ]
    if missing_serving:
        raise SystemExit(f"missing serving rows in {SERVING_CSV}: {missing_serving}")
    if missing_kernel:
        raise SystemExit(f"missing kernel rows in {KERNEL_JSON}: {missing_kernel}")


def estimate_mixed_tok_s(
    *,
    batch_size: int,
    serving_bf16_tok_s: float,
    serving_qlora_tok_s: float,
    bf16_kernel_us: float,
    qlora_kernel_us: float,
    mixed_kernel_us: float,
    qlora_projection_count: int,
) -> float:
    bf16_step_ms = batch_size * 1000.0 / serving_bf16_tok_s
    qlora_step_ms = batch_size * 1000.0 / serving_qlora_tok_s

    if qlora_projection_count == 0:
        return serving_bf16_tok_s
    if qlora_projection_count == len(PROJECTIONS):
        return serving_qlora_tok_s

    bf16_distance = qlora_projection_count
    qlora_distance = len(PROJECTIONS) - qlora_projection_count
    if bf16_distance <= qlora_distance:
        mixed_step_ms = bf16_step_ms + (mixed_kernel_us - bf16_kernel_us) * LAYERS / 1000.0
    else:
        mixed_step_ms = qlora_step_ms + (mixed_kernel_us - qlora_kernel_us) * LAYERS / 1000.0
    return batch_size * 1000.0 / mixed_step_ms


def build_rows() -> list[dict]:
    serving = load_serving_rows()
    kernel = load_kernel_rows()
    require_inputs(serving, kernel)

    rows = []
    for bs in BATCH_SIZES:
        projection_policy = {}
        bf16_kernel_us = 0.0
        qlora_kernel_us = 0.0
        mixed_kernel_us = 0.0

        for proj in PROJECTIONS:
            lat = kernel[(bs, proj)]
            bf16_us = lat[BF16]
            qlora_us = lat[QLORA]
            pick = "qlora" if qlora_us < bf16_us else "bf16"
            projection_policy[proj] = pick
            bf16_kernel_us += bf16_us
            qlora_kernel_us += qlora_us
            mixed_kernel_us += min(bf16_us, qlora_us)

        bf16_tok_s = serving[("bf16_merged", bs)]
        qlora_tok_s = serving[("qlora_torch_twostream", bs)]
        qlora_projection_count = sum(
            1 for mode in projection_policy.values() if mode == "qlora"
        )
        mixed_tok_s = estimate_mixed_tok_s(
            batch_size=bs,
            serving_bf16_tok_s=bf16_tok_s,
            serving_qlora_tok_s=qlora_tok_s,
            bf16_kernel_us=bf16_kernel_us,
            qlora_kernel_us=qlora_kernel_us,
            mixed_kernel_us=mixed_kernel_us,
            qlora_projection_count=qlora_projection_count,
        )
        whole_policy_tok_s = max(bf16_tok_s, qlora_tok_s)
        frontier_tok_s = max(whole_policy_tok_s, mixed_tok_s)

        rows.append(
            {
                "batch_size": bs,
                "model": MODEL,
                "bf16_merged_tok_s": bf16_tok_s,
                "qlora_torch_twostream_tok_s": qlora_tok_s,
                "kernel_mixed_estimated_tok_s": mixed_tok_s,
                "whole_policy_best_tok_s": whole_policy_tok_s,
                "kernel_guided_frontier_tok_s": frontier_tok_s,
                "qlora_speedup_vs_bf16": qlora_tok_s / bf16_tok_s,
                "mixed_estimated_speedup_vs_bf16": mixed_tok_s / bf16_tok_s,
                "frontier_speedup_vs_bf16": frontier_tok_s / bf16_tok_s,
                "bf16_projection_kernel_us": bf16_kernel_us,
                "qlora_projection_kernel_us": qlora_kernel_us,
                "mixed_projection_kernel_us": mixed_kernel_us,
                "qlora_projection_count": qlora_projection_count,
                **{f"select_{proj}": projection_policy[proj] for proj in PROJECTIONS},
            }
        )
    return rows


def write_data(rows: list[dict]) -> None:
    payload = {
        "metadata": {
            "model": MODEL,
            "decode_tokens_per_request": 256,
            "prefill_excluded": True,
            "serving_source_csv": str(SERVING_CSV),
            "kernel_source_json": str(KERNEL_JSON),
            "batch_sizes": BATCH_SIZES,
            "projection_choices": PROJECTIONS,
            "selection_rule": (
                "For each batch and projection, choose BF16 merged if its kernel "
                "latency is lower, otherwise choose int4 + BF16 LoRA Torch "
                "two-stream. Throughput for mixed policies is estimated by "
                "adjusting the nearest measured whole-policy serving step time "
                "with the per-layer kernel latency delta."
            ),
            "not_a_new_serving_experiment": True,
        },
        "rows": rows,
    }
    (OUT_DIR / "frontier_decoding_throughput.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with (OUT_DIR / "frontier_decoding_throughput.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Batch | qkv | o | gate_up | down | BF16 tok/s | QLoRA tok/s | Mixed est tok/s | Frontier tok/s | Frontier speedup |",
        "|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {batch_size} | {select_qkv} | {select_o} | {select_gate_up} | {select_down} | "
            "{bf16_merged_tok_s:.1f} | {qlora_torch_twostream_tok_s:.1f} | "
            "{kernel_mixed_estimated_tok_s:.1f} | {kernel_guided_frontier_tok_s:.1f} | "
            "{frontier_speedup_vs_bf16:.2f}x |".format(**row)
        )
    (OUT_DIR / "frontier_summary_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_throughput(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    xs = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.plot(xs, [r["bf16_merged_tok_s"] for r in rows], marker="o", linewidth=1.9, label="BF16 merged")
    ax.plot(xs, [r["qlora_torch_twostream_tok_s"] for r in rows], marker="s", linewidth=1.9, label="All int4 + two-stream LoRA")
    ax.plot(xs, [r["kernel_guided_frontier_tok_s"] for r in rows], marker="D", linewidth=2.1, label="Best available frontier")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_ylabel("Decode throughput (tokens/s), prefill excluded")
    ax.set_title(f"Kernel-Guided QLoRA Decoding Frontier ({MODEL})")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT_DIR / "frontier_throughput.png", dpi=300)
    fig.savefig(OUT_DIR / "frontier_throughput.pdf")
    plt.close(fig)


def plot_speedup(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    xs = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.axhline(1.0, color="#333333", linewidth=1.1, alpha=0.7)
    ax.plot(xs, [r["qlora_speedup_vs_bf16"] for r in rows], marker="s", linewidth=1.9, label="All int4 two-stream / BF16")
    ax.plot(xs, [r["frontier_speedup_vs_bf16"] for r in rows], marker="D", linewidth=2.1, label="Frontier / BF16")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_ylabel("Speedup over BF16 merged rollout")
    ax.set_title(f"Kernel-Guided Decode Speedup ({MODEL})")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT_DIR / "frontier_speedup.png", dpi=300)
    fig.savefig(OUT_DIR / "frontier_speedup.pdf")
    plt.close(fig)


def plot_policy(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_style()
    matrix = np.array(
        [
            [1 if row[f"select_{proj}"] == "qlora" else 0 for row in rows]
            for proj in PROJECTIONS
        ]
    )
    fig, ax = plt.subplots(figsize=(6.6, 2.8), constrained_layout=True)
    ax.imshow(matrix, aspect="auto", cmap=plt.colormaps["RdYlBu_r"], vmin=0, vmax=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(r["batch_size"]) for r in rows])
    ax.set_yticks(range(len(PROJECTIONS)))
    ax.set_yticklabels(PROJECTIONS)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_title("Kernel-Guided Per-Projection Precision Policy")
    for y, proj in enumerate(PROJECTIONS):
        for x, row in enumerate(rows):
            label = "Int4+LoRA" if row[f"select_{proj}"] == "qlora" else "BF16"
            ax.text(x, y, label, ha="center", va="center", fontsize=6.5, color="#111111")
    fig.savefig(OUT_DIR / "frontier_projection_policy.png", dpi=300)
    fig.savefig(OUT_DIR / "frontier_projection_policy.pdf")
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_data(rows)
    plot_throughput(rows)
    plot_speedup(rows)
    plot_policy(rows)
    print(f"Wrote kernel-guided frontier deliverables to {OUT_DIR}")


if __name__ == "__main__":
    main()
