"""Prepare custom JSONL prompt datasets for rollout precision experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASETS = {
    "math": [
        "BytedTsinghua-SIA/DAPO-Math-17k",
        "train",
    ],
    "reasoning": [
        "open-thoughts/OpenThoughts-114k",
        "train",
    ],
    "agentic": [
        "Salesforce/xlam-function-calling-60k",
        "train",
    ],
}


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _first_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        for key in (
            "prompt",
            "query",
            "question",
            "problem",
            "instruction",
            "content",
            "value",
            "input",
        ):
            text = _first_text(value.get(key))
            if text:
                return text
        if "messages" in value:
            return _first_text(value["messages"])
        if "conversations" in value:
            return _first_text(value["conversations"])
    return None


def _completion(row: dict[str, Any]) -> str:
    for key in ("answer", "solution", "response", "completion", "output"):
        text = _first_text(row.get(key))
        if text:
            return text
    return "We need solve the task step by step."


def _load_hf_rows(repo: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install `datasets` or provide existing custom JSONL files."
        ) from exc

    dataset = load_dataset(repo, split=split)
    for row in dataset:
        yield dict(row)


def prepare_dataset(
    *,
    name: str,
    repo: str,
    split: str,
    output_dir: Path,
    num_prompts: int,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    rows = list(_load_hf_rows(repo, split))
    rng.shuffle(rows)

    out_path = output_dir / f"{name}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = _first_text(row)
            if not prompt:
                continue
            completion = _completion(row)
            record = {
                "category": name,
                "source": repo,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "assistant", "value": completion},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written >= num_prompts:
                break

    if written == 0:
        raise RuntimeError(f"No usable prompts found for {name} from {repo}:{split}")
    print(f"Wrote {written} prompts to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=HF_REPO:SPLIT",
        help="Override/add dataset spec. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = dict(DEFAULT_DATASETS)
    for spec in args.dataset:
        name, rest = spec.split("=", 1)
        repo, split = rest.rsplit(":", 1)
        specs[name] = [repo, split]

    output_dir = Path(args.output_dir)
    for name, (repo, split) in specs.items():
        prepare_dataset(
            name=name,
            repo=repo,
            split=split,
            output_dir=output_dir,
            num_prompts=args.num_prompts,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
