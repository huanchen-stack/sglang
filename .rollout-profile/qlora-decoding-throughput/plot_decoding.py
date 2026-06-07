#!/usr/bin/env python3
"""Build decoding-throughput plots and summary tables from real sweep JSONs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = OUT_DIR / "results"
SCHEME_LABELS = {
    "bf16_merged": "BF16 merged/no adapter patch",
    "qlora_csgmv": "Int4 + BF16 LoRA, SGLang csgmv",
    "qlora_torch_twostream": "Int4 + BF16 LoRA, Torch two-stream",
}


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.bs*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        rows.append(
            {
                "scheme": payload["scheme"],
                "batch_size": int(summary["batch_size"]),
                "decode_tokens_per_request": int(
                    summary["decode_tokens_per_request"]
                ),
                "success": bool(summary["success"]),
                "decode_time_s": summary.get("decode_time_s"),
                "decode_tok_s": summary.get("decode_tok_s"),
                "decode_tok_s_excluding_first_token": summary.get(
                    "decode_tok_s_excluding_first_token"
                ),
                "completed_requests": summary.get("completed_requests", 0),
                "source": str(path),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(rows: list[dict], path: Path) -> None:
    by_batch: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_batch.setdefault(row["batch_size"], {})[row["scheme"]] = row

    lines = [
        "| Batch | BF16 tok/s | csgmv QLoRA tok/s | Torch two-stream QLoRA tok/s | Best QLoRA speedup |",
        "|---:|---:|---:|---:|---:|",
    ]
    for bs in sorted(by_batch):
        group = by_batch[bs]
        bf16 = (group.get("bf16_merged") or {}).get("decode_tok_s")
        csgmv = (group.get("qlora_csgmv") or {}).get("decode_tok_s")
        torch_ts = (group.get("qlora_torch_twostream") or {}).get("decode_tok_s")
        qlora_best = max([v for v in [csgmv, torch_ts] if v is not None], default=None)
        speedup = qlora_best / bf16 if bf16 and qlora_best else None
        lines.append(
            "| {bs} | {bf16} | {csgmv} | {torch_ts} | {speedup} |".format(
                bs=bs,
                bf16=f"{bf16:.1f}" if bf16 else "",
                csgmv=f"{csgmv:.1f}" if csgmv else "",
                torch_ts=f"{torch_ts:.1f}" if torch_ts else "",
                speedup=f"{speedup:.2f}x" if speedup else "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    schemes = sorted({row["scheme"] for row in rows})
    for scheme in schemes:
        scheme_rows = sorted(
            [row for row in rows if row["scheme"] == scheme], key=lambda r: r["batch_size"]
        )
        ax.plot(
            [row["batch_size"] for row in scheme_rows],
            [row["decode_tok_s"] for row in scheme_rows],
            marker="o",
            linewidth=2.3,
            label=SCHEME_LABELS.get(scheme, scheme),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["batch_size"] for row in rows}))
    ax.set_xticklabels([str(x) for x in sorted({row["batch_size"] for row in rows})])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Decode batch size / concurrent requests")
    ax.set_ylabel("Decode throughput (tokens/s), prefill excluded")
    ax.set_title("QLoRA Rollout Decoding Throughput")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
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
