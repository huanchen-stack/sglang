"""Self-contained generator for the two int4_only_experts.json files used
by run_e2e_eval.sh.

Outputs (next to this script):
  baseline_int4_only_experts.json   -- every expert in every layer is INT4-only
                                       (no BF16 group → pure-INT4 baseline)
  treatment_int4_only_experts.json  -- top-K experts by routed-token count are
                                       BF16-promoted (heter); rest are INT4-only

Token-count source: legacy/sensitivity/per_expert/results/summary.json
Ranking: pure --activation_frequency (no Hessian input). Ordered by
token_count DESC, then (layer ASC, expert ASC) for determinism.

Default K = 3775 matches the activation_frequency assignment of the existing
heter-final-tuning/mc64 budget (see expert_precision_assignment/.../mc64/
assignment_report.json). Override with --k.

Usage:
    python gen_int4_only_experts.py            # default K
    python gen_int4_only_experts.py --k 4096   # custom K
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "expert_precision_assignment/legacy/sensitivity/per_expert/results/summary.json"
)
DEFAULT_K = 3775  # heter expert count under the mc64 activation_frequency budget
NUM_LAYERS = 48
NUM_EXPERTS = 128


def _load_token_counts(summary_path: Path) -> dict[tuple[int, int], int]:
    with open(summary_path) as f:
        d = json.load(f)
    per_layer = d["per_layer"]
    tc: dict[tuple[int, int], int] = {}
    for L_str, layer_d in per_layer.items():
        L = int(L_str)
        for E_str, e_d in layer_d["experts"].items():
            tc[(L, int(E_str))] = int(e_d["token_count"])
    expected = NUM_LAYERS * NUM_EXPERTS
    if len(tc) != expected:
        raise ValueError(
            f"summary {summary_path}: got {len(tc)} (layer,expert) pairs, "
            f"expected {expected}"
        )
    return tc


def _by_layer(int4_pairs: list[tuple[int, int]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {str(L): [] for L in range(NUM_LAYERS)}
    for L, E in int4_pairs:
        out[str(L)].append(E)
    for k in out:
        out[k].sort()
    return out


def write_baseline(path: Path) -> None:
    """All experts INT4-only — forces pure-INT4 execution."""
    every_pair = [(L, E) for L in range(NUM_LAYERS) for E in range(NUM_EXPERTS)]
    with open(path, "w") as f:
        json.dump(_by_layer(every_pair), f, indent=2)
    total = NUM_LAYERS * NUM_EXPERTS
    print(f"wrote {path}  ({total} pairs, all INT4)")


def write_treatment(path: Path, k: int, summary_path: Path) -> None:
    """Top-k by token_count globally → heter (BF16 promoted); rest → INT4-only."""
    tc = _load_token_counts(summary_path)
    # DESC by count; tie-break (layer, expert) ASC for determinism.
    ranked = sorted(tc.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    heter = {pair for pair, _ in ranked[:k]}
    int4_only_pairs = [pair for pair, _ in ranked[k:]]
    with open(path, "w") as f:
        json.dump(_by_layer(int4_only_pairs), f, indent=2)
    print(
        f"wrote {path}  "
        f"(K_heter={len(heter)}, num_int4_only={len(int4_only_pairs)}, "
        f"source={summary_path.name})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"Heter expert count (top-k by token_count). "
                         f"Default {DEFAULT_K} matches mc64 activation_frequency.")
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                    help="per-expert summary.json with token_count.")
    ap.add_argument("--out_dir", type=Path, default=THIS_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_baseline(args.out_dir / "baseline_int4_only_experts.json")
    write_treatment(
        args.out_dir / "treatment_int4_only_experts.json",
        k=args.k, summary_path=args.summary,
    )


if __name__ == "__main__":
    main()
