#!/usr/bin/env python3
"""Build clean QLoRA comparison data.

Sequential LoRA uses SGLang csgmv.  The two-stream overlap line uses the
Torch/cuBLAS LoRA patch path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = OUT_DIR / "archive"
DEFAULT_INPUTS = [
    ARCHIVE_DIR / "raw-inputs" / "qlora_kernel_perf.json",
    ARCHIVE_DIR / "raw-inputs" / "qlora_kernel_perf_bf16_torch_twostream.json",
]
DEFAULT_OUTPUT = OUT_DIR / "qlora_kernel_perf_clean.json"

ALIASES = {
    "bf16 dense base": "bf16",
    "int4 Marlin base": "int4",
    "bf16 dense + csgmv sequential": "bf16 + sequential csgmv LoRA",
    "bf16 dense + torch matmul two-stream": "bf16 + two-stream Torch LoRA",
    "SGLang QLoRA csgmv sequential": "int4 + sequential csgmv LoRA",
    "Torch QLoRA matmul two-stream": "int4 + two-stream Torch LoRA",
}

SCHEME_ORDER = [
    "bf16",
    "bf16 + sequential csgmv LoRA",
    "bf16 + two-stream Torch LoRA",
    "int4",
    "int4 + sequential csgmv LoRA",
    "int4 + two-stream Torch LoRA",
]

SUMMARY = [
    "Clean comparison plot with sequential LoRA from SGLang csgmv and overlapped LoRA from Torch/cuBLAS.",
    "Both BF16 and int4 bases include a csgmv sequential LoRA line and a Torch two-stream LoRA line.",
    "Triton LoRA rows are intentionally excluded from this clean plot.",
    "The int4 two-stream Torch LoRA series uses Marlin SM reservation only inside the two-stream Marlin callable.",
]


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: dict) -> tuple:
    return (
        row.get("model"),
        row.get("projection"),
        row.get("projection_kind"),
        row.get("in_features"),
        row.get("out_features"),
        row.get("token_rows"),
        row.get("scheme"),
    )


def build(inputs: list[Path]) -> dict:
    if not inputs:
        raise ValueError("at least one input data file is required")

    base = load_payload(inputs[0])
    rows_by_key = {}
    sources = []
    for path in inputs:
        payload = load_payload(path)
        sources.append(str(path))
        for raw in payload.get("measurements", []):
            alias = ALIASES.get(raw.get("scheme"))
            if alias is None or raw.get("error"):
                continue
            row = dict(raw)
            row["scheme"] = alias
            row["note"] = f"Clean comparison alias of {raw.get('scheme')}"
            rows_by_key[row_key(row)] = row

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            row.get("model") or "",
            row.get("projection") or "",
            row.get("token_rows") or 0,
            SCHEME_ORDER.index(row["scheme"]),
        ),
    )
    payload = {
        "metadata": {
            **base.get("metadata", {}),
            "created_unix": time.time(),
            "scheme_order": SCHEME_ORDER,
            "research_summary": SUMMARY,
            "source_data": sources,
        },
        "measurements": rows,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def default_inputs() -> list[Path]:
    shard_dir = ARCHIVE_DIR / "old-shards" / "shards"
    return DEFAULT_INPUTS + sorted(shard_dir.glob("*_bf16_csgmv_seq.json"))


def main() -> None:
    args = parse_args()
    inputs = args.input or default_inputs()
    payload = build(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(payload['measurements'])} rows")


if __name__ == "__main__":
    main()
