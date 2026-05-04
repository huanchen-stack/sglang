"""Simple per-layer expert assignment for HeterFusedMoE.

For each MoE layer, rank experts by routed-token count (DESC) and keep the
top-X as heter (BF16-promoted). The remaining ``num_experts - X`` per layer
are written to ``int4_only_experts.json``. Every layer ends up with EXACTLY
X heter experts — no cross-layer skew.

This is a simpler alternative to ``expert_precision_assignment/policy/
heter_assign/assign_experts.py``, which uses a GLOBAL top-K ranking and can
leave individual layers with very different heter counts.

Token-count source: ``legacy/sensitivity/per_expert/results/summary.json``
(the file's ``sensitivity`` column is unused; only ``token_count``).

Output (default path overwrites the file consumed by treatment_mc64.json):
    eval/configs/treatment_int4_only_experts.json

Usage:
    python assign_experts.py -n 64
    python assign_experts.py -n 80 --out /tmp/preview.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "expert_precision_assignment/legacy/sensitivity/per_expert/results/summary.json"
)
DEFAULT_OUT = THIS_DIR / "eval" / "configs" / "treatment_int4_only_experts.json"
NUM_LAYERS = 48
NUM_EXPERTS = 128


def _load_token_counts(summary_path: Path) -> dict[int, dict[int, int]]:
    """Per-layer dict[expert_id] = token_count. Validates full coverage."""
    with open(summary_path) as f:
        d = json.load(f)
    out: dict[int, dict[int, int]] = {}
    for L_str, layer_d in d["per_layer"].items():
        L = int(L_str)
        out[L] = {
            int(E): int(e_d["token_count"])
            for E, e_d in layer_d["experts"].items()
        }
    if set(out) != set(range(NUM_LAYERS)):
        missing = set(range(NUM_LAYERS)) - set(out)
        raise ValueError(f"{summary_path}: missing layers {sorted(missing)}")
    for L in range(NUM_LAYERS):
        if len(out[L]) != NUM_EXPERTS:
            raise ValueError(
                f"{summary_path}: layer {L} has {len(out[L])} experts, "
                f"expected {NUM_EXPERTS}"
            )
    return out


def assign_per_layer(
    token_counts: dict[int, dict[int, int]], x: int
) -> dict[str, list[int]]:
    """For each layer, top-x by token_count → heter; rest → int4_only.

    Tie-break: token_count DESC, then expert_id ASC (deterministic).
    Returns ``{str(layer_id): [int4_only_expert_ids sorted ASC]}``.
    """
    if not 0 <= x <= NUM_EXPERTS:
        raise ValueError(f"-n must be in [0, {NUM_EXPERTS}], got {x}")
    int4_by_layer: dict[str, list[int]] = {}
    for L in range(NUM_LAYERS):
        ranked = sorted(
            token_counts[L].items(), key=lambda kv: (-kv[1], kv[0])
        )
        int4_only = sorted(e for e, _ in ranked[x:])
        int4_by_layer[str(L)] = int4_only
    return int4_by_layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-n", type=int, required=True, dest="n",
        help=f"Number of heter (BF16-promoted) experts per layer "
             f"(out of {NUM_EXPERTS}).",
    )
    ap.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY,
        help="per-expert summary.json with token_count "
             "(default: %(default)s).",
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output int4_only_experts.json path "
             "(default: %(default)s).",
    )
    args = ap.parse_args()

    tc = _load_token_counts(args.summary)
    int4_by_layer = assign_per_layer(tc, args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(int4_by_layer, f, indent=2)

    n_int4 = sum(len(v) for v in int4_by_layer.values())
    n_heter = NUM_LAYERS * NUM_EXPERTS - n_int4
    print(f"wrote {args.out}")
    print(
        f"  per-layer n={args.n} -> heter={args.n}/layer, "
        f"int4_only={NUM_EXPERTS - args.n}/layer"
    )
    print(f"  totals over {NUM_LAYERS} layers: heter={n_heter}, "
          f"int4_only={n_int4}")


if __name__ == "__main__":
    main()
