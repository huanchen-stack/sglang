#!/usr/bin/env python3
"""Summarize dynamic-policy fixed decode verification results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PROFILE_CSV = REPO_ROOT / ".rollout-profile/qlora-decoding-throughput/decoding_throughput.csv"
FRONTIER_JSON = REPO_ROOT / ".rollout-profile/qlora-decoding-throughput/frontier_decoding_throughput.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile() -> dict[tuple[str, int], dict]:
    rows = {}
    with PROFILE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["scheme"], int(row["batch_size"]))] = row
    return rows


def load_frontier() -> dict[int, dict]:
    payload = load_json(FRONTIER_JSON)
    return {int(row["batch_size"]): row for row in payload["rows"]}


def policy_label(frontier_row: dict) -> str:
    selected = []
    for key, label in (
        ("qkv", "qkv"),
        ("o", "out"),
        ("gate_up", "up"),
        ("down", "down"),
    ):
        if frontier_row.get(f"select_{key}") == "qlora":
            selected.append(label)
    return f"Q({', '.join(selected)})" if selected else "Q(null)"


def summarize(results_dir: Path) -> list[dict]:
    profile = load_profile()
    frontier = load_frontier()
    rows = []
    for run_path in sorted(results_dir.glob("bs*/dynamic_policy.json")):
        batch_size = int(run_path.parent.name.removeprefix("bs"))
        dynamic = load_json(run_path)
        summary = dynamic["summary"]
        bf16 = profile.get(("bf16_merged", batch_size), {})
        qlora = profile.get(("qlora_torch_twostream", batch_size), {})
        csgmv = profile.get(("qlora_csgmv", batch_size), {})
        frontier_row = frontier.get(batch_size, {})
        dynamic_tok_s = float(summary["decode_tok_s"])
        bf16_tok_s = float(bf16["decode_tok_s"]) if bf16 else 0.0
        qlora_tok_s = float(qlora["decode_tok_s"]) if qlora else 0.0
        csgmv_tok_s = float(csgmv["decode_tok_s"]) if csgmv else 0.0
        frontier_tok_s = float(frontier_row.get("kernel_guided_frontier_tok_s", 0.0))
        rows.append(
            {
                "batch_size": batch_size,
                "dynamic_decode_tok_s": dynamic_tok_s,
                "profile_bf16_tok_s": bf16_tok_s,
                "profile_qlora_torch_tok_s": qlora_tok_s,
                "profile_qlora_csgmv_tok_s": csgmv_tok_s,
                "profile_frontier_tok_s": frontier_tok_s,
                "dynamic_vs_bf16": dynamic_tok_s / bf16_tok_s if bf16_tok_s else 0.0,
                "dynamic_vs_profile_qlora_torch": dynamic_tok_s / qlora_tok_s if qlora_tok_s else 0.0,
                "dynamic_vs_profile_frontier": dynamic_tok_s / frontier_tok_s if frontier_tok_s else 0.0,
                "frontier_policy_label": policy_label(frontier_row) if frontier_row else "",
                "completed_requests": int(summary["completed_requests"]),
                "observed_tokens": int(summary["observed_tokens"]),
                "decode_time_s": float(summary["decode_time_s"]),
                "first_token_spread_s": float(summary["first_token_spread_s"]),
            }
        )
    return rows


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"no dynamic_policy.json files found under {out_dir}")
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Batch | Dynamic tok/s | vs BF16 profile | vs Torch2S profile | vs frontier | Frontier label |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['dynamic_decode_tok_s']:.1f} | "
            f"x{row['dynamic_vs_bf16']:.2f} | "
            f"x{row['dynamic_vs_profile_qlora_torch']:.2f} | "
            f"x{row['dynamic_vs_profile_frontier']:.2f} | "
            f"{row['frontier_policy_label']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize(args.results_dir)
    write_outputs(rows, args.results_dir)
    print((args.results_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
