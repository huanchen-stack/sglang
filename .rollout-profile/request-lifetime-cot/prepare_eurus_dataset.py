#!/usr/bin/env python3
"""Materialize Eurus-2-RL prompts as local JSONL for lifetime profiling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def normalize_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        messages = []
        for item in prompt:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("value")
            if not isinstance(content, str) or not content.strip():
                continue
            role = str(item.get("role") or item.get("from") or "user").lower()
            if role in {"human", "user"}:
                role = "user"
            elif role in {"assistant", "gpt"}:
                role = "assistant"
            elif role != "system":
                role = "user"
            messages.append({"role": role, "content": content.strip()})
        if messages:
            return messages
    raise ValueError(f"unsupported prompt format: {type(prompt)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PRIME-RL/Eurus-2-RL-Data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument(
        "--per-ability-limit",
        type=int,
        default=None,
        help="Collect up to this many rows per ability before writing.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    count = 0
    scanned = 0
    ability_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in ds:
        scanned += 1
        ability = str(row.get("ability", "unknown"))
        if args.per_ability_limit is not None and ability_counts.get(ability, 0) >= args.per_ability_limit:
            if sum(ability_counts.values()) >= args.limit:
                break
            continue
        messages = normalize_messages(row["prompt"])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        record = {
            "prompt": prompt,
            "dataset": args.dataset,
            "data_source": row.get("data_source"),
            "ability": row.get("ability"),
            "extra_info": row.get("extra_info"),
        }
        records.append(record)
        source = str(row.get("data_source", "unknown"))
        ability_counts[ability] = ability_counts.get(ability, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        count += 1
        if count >= args.limit:
            break

    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "output": str(args.output),
        "limit": args.limit,
        "written": count,
        "scanned": scanned,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "per_ability_limit": args.per_ability_limit,
        "ability_counts": ability_counts,
        "top_data_sources": sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:20],
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Some versions of streaming datasets leave non-daemon workers that abort
    # during interpreter teardown in this environment. We have written and
    # flushed all outputs, so exit without running those destructors.
    os._exit(0)


if __name__ == "__main__":
    main()
