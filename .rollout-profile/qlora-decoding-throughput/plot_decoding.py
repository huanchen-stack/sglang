#!/usr/bin/env python3
"""Build decoding-throughput plots and summary tables from real sweep JSONs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = OUT_DIR / "measurements_14b_current"
SCHEME_LABELS = {
    "bf16_merged": "BF16 merged/no adapter patch",
    "qlora_csgmv": "Int4 + BF16 LoRA, SGLang csgmv",
    "qlora_torch_twostream": "Int4 + BF16 LoRA, Torch two-stream",
}


def recompute_summary(payload: dict, source: Path) -> dict:
    summary = payload["summary"]
    traces = payload["requests"]
    successful = [
        trace
        for trace in traces
        if trace.get("success")
        and trace.get("first_token_s") is not None
        and trace.get("last_token_s") is not None
    ]
    success = bool(summary.get("success")) and len(successful) == len(traces)
    row = {
        "scheme": payload["scheme"],
        "batch_size": int(summary["batch_size"]),
        "decode_tokens_per_request": int(summary["decode_tokens_per_request"]),
        "success": success,
        "decode_time_s": None,
        "decode_tok_s": None,
        "full_decode_time_s": None,
        "full_decode_tok_s": None,
        "all_started_decode_time_s": None,
        "all_started_decode_tok_s": None,
        "tokens_after_all_started": None,
        "observed_tokens": None,
        "first_token_spread_s": None,
        "legacy_decode_tok_s": summary.get("decode_tok_s"),
        "completed_requests": len(successful),
        "source": str(source),
    }
    if not success:
        return row

    first_decode_start = min(trace["first_token_s"] for trace in successful)
    all_started_decode_start = max(trace["first_token_s"] for trace in successful)
    decode_end = max(trace["last_token_s"] for trace in successful)
    full_decode_time_s = max(decode_end - first_decode_start, 0.0)
    all_started_decode_time_s = max(decode_end - all_started_decode_start, 0.0)
    observed_tokens = sum(int(trace.get("output_len") or 0) for trace in successful)
    tokens_after_all_started = sum(
        1
        for trace in successful
        for _, timestamp in trace.get("token_timestamps_s", [])
        if timestamp >= all_started_decode_start
    )

    full_decode_tok_s = (
        observed_tokens / full_decode_time_s if full_decode_time_s > 0 else None
    )
    all_started_decode_tok_s = (
        tokens_after_all_started / all_started_decode_time_s
        if all_started_decode_time_s > 0
        else None
    )
    row.update(
        {
            "decode_time_s": full_decode_time_s,
            "decode_tok_s": full_decode_tok_s,
            "full_decode_time_s": full_decode_time_s,
            "full_decode_tok_s": full_decode_tok_s,
            "all_started_decode_time_s": all_started_decode_time_s,
            "all_started_decode_tok_s": all_started_decode_tok_s,
            "tokens_after_all_started": tokens_after_all_started,
            "observed_tokens": observed_tokens,
            "first_token_spread_s": all_started_decode_start - first_decode_start,
        }
    )
    return row


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.bs*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(recompute_summary(payload, path))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "scheme",
        "batch_size",
        "decode_tokens_per_request",
        "success",
        "decode_time_s",
        "decode_tok_s",
        "full_decode_time_s",
        "full_decode_tok_s",
        "all_started_decode_time_s",
        "all_started_decode_tok_s",
        "tokens_after_all_started",
        "observed_tokens",
        "first_token_spread_s",
        "legacy_decode_tok_s",
        "completed_requests",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table(rows: list[dict], path: Path) -> None:
    by_batch: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_batch.setdefault(row["batch_size"], {})[row["scheme"]] = row

    lines = [
        "| Batch | BF16 full tok/s | csgmv QLoRA full tok/s | Torch two-stream QLoRA full tok/s | Best QLoRA speedup | Torch all-started tok/s | Torch first-token spread s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bs in sorted(by_batch):
        group = by_batch[bs]
        bf16 = (group.get("bf16_merged") or {}).get("decode_tok_s")
        csgmv = (group.get("qlora_csgmv") or {}).get("decode_tok_s")
        torch_ts = (group.get("qlora_torch_twostream") or {}).get("decode_tok_s")
        torch_all_started = (group.get("qlora_torch_twostream") or {}).get(
            "all_started_decode_tok_s"
        )
        torch_spread = (group.get("qlora_torch_twostream") or {}).get(
            "first_token_spread_s"
        )
        qlora_best = max([v for v in [csgmv, torch_ts] if v is not None], default=None)
        speedup = qlora_best / bf16 if bf16 and qlora_best else None
        lines.append(
            "| {bs} | {bf16} | {csgmv} | {torch_ts} | {speedup} | {torch_all_started} | {torch_spread} |".format(
                bs=bs,
                bf16=f"{bf16:.1f}" if bf16 else "",
                csgmv=f"{csgmv:.1f}" if csgmv else "",
                torch_ts=f"{torch_ts:.1f}" if torch_ts else "",
                speedup=f"{speedup:.2f}x" if speedup else "",
                torch_all_started=(
                    f"{torch_all_started:.1f}"
                    if torch_all_started is not None
                    else ""
                ),
                torch_spread=f"{torch_spread:.2f}" if torch_spread is not None else "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict], out_path: Path) -> None:
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
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    schemes = sorted({row["scheme"] for row in rows})
    for scheme in schemes:
        scheme_rows = sorted(
            [row for row in rows if row["scheme"] == scheme], key=lambda r: r["batch_size"]
        )
        ax.plot(
            [row["batch_size"] for row in scheme_rows],
            [row["decode_tok_s"] for row in scheme_rows],
            marker="o",
            linewidth=1.9,
            label=SCHEME_LABELS.get(scheme, scheme),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["batch_size"] for row in rows}))
    ax.set_xticklabels([str(x) for x in sorted({row["batch_size"] for row in rows})])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_ylabel("Decode throughput (tokens/s), full decode window")
    ax.set_title("QLoRA Rollout Decoding Throughput")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    rows = load_rows(DEFAULT_RESULTS)
    if not rows:
        raise SystemExit(f"no result JSONs found in {DEFAULT_RESULTS}")
    write_csv(rows, OUT_DIR / "decoding_throughput.csv")
    write_table(rows, OUT_DIR / "summary_table.md")
    plot(rows, OUT_DIR / "decoding_throughput.png")
    print(f"Wrote {OUT_DIR / 'decoding_throughput.png'}")


if __name__ == "__main__":
    main()
