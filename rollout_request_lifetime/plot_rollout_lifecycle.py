"""Parse a rollout-lifecycle run's trace and produce:
  - lifecycle.json  (decode start, finish times, live curve, bin durations)
  - lifecycle_<label>_<cat>_bs<N>.png  (left: lifetime, right: duration bins)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trace(run_dir: Path, expected_mnt: int | None):
    rows = []
    for p in sorted((run_dir / "traces").glob("*.jsonl")):
        rows.extend(read_jsonl(p))

    # auto-detect the benchmark cap = most common max_new_tokens among non-HEALTH
    # request_complete decode events, unless explicitly provided.
    if expected_mnt is None:
        from collections import Counter
        c = Counter(
            r.get("max_new_tokens") for r in rows
            if r.get("event") == "request_complete" and r.get("phase") == "decode"
            and not str(r.get("rid", "")).startswith("HEALTH")
            and r.get("max_new_tokens") is not None
        )
        expected_mnt = c.most_common(1)[0][0] if c else None

    def is_bench(r):
        rid = str(r.get("rid", ""))
        return r.get("max_new_tokens") == expected_mnt and not rid.startswith("HEALTH")

    completes = [r for r in rows
                 if r.get("event") == "request_complete"
                 and r.get("phase") == "decode" and is_bench(r)]
    decode_starts = [r for r in rows
                     if r.get("event") == "phase_start"
                     and r.get("phase") == "decode" and is_bench(r)]
    return rows, completes, decode_starts


def halving_bins(n):
    edges, hi = [], n
    while hi > 8:
        lo = hi // 2
        edges.append((lo, hi))
        hi = lo
    edges.append((0, hi))
    return edges


def analyze(run_dir: Path, expected_mnt: int):
    _, completes, decode_starts = load_trace(run_dir, expected_mnt)
    if not completes or not decode_starts:
        raise RuntimeError(f"no benchmark decode events in {run_dir}")

    t0 = min(r["ts"] for r in decode_starts)
    finish_ts = sorted(r["ts"] for r in completes)
    n = len(finish_ts)
    rel_finish = [t - t0 for t in finish_ts]

    # finish reasons
    reasons = {}
    for r in completes:
        fr = r.get("finish_reason")
        kind = fr.get("type") if isinstance(fr, dict) else fr
        reasons[kind] = reasons.get(kind, 0) + 1
    out_lens = [int(r.get("output_len") or 0) for r in completes]

    # bin durations: between consecutive completions, live = n - k
    bins = halving_bins(n)
    durations = [0.0] * len(bins)
    prev, cur_live = 0.0, n
    for t in rel_finish:
        dur = t - prev
        for i, (lo, hi) in enumerate(bins):
            if lo < cur_live <= hi:
                durations[i] += dur
                break
        prev, cur_live = t, cur_live - 1

    return {
        "n": n,
        "t0_perf": t0,
        "rel_finish": rel_finish,
        "total_drain_s": rel_finish[-1] if rel_finish else 0.0,
        "bin_labels": [f"{hi}-{lo}" for (lo, hi) in bins],
        "bin_durations_s": durations,
        "finish_reasons": reasons,
        "output_len_mean": float(np.mean(out_lens)) if out_lens else 0.0,
        "output_len_p50": float(np.median(out_lens)) if out_lens else 0.0,
        "output_len_p99": float(np.percentile(out_lens, 99)) if out_lens else 0.0,
        "output_len_max": int(np.max(out_lens)) if out_lens else 0,
    }


def plot(a: dict, title: str, out_png: Path):
    rel = np.array(a["rel_finish"])
    n = a["n"]
    # live curve: step from n down to 0 at each finish
    t = np.concatenate([[0.0], rel])
    live = np.concatenate([[n], n - np.arange(1, len(rel) + 1)])

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
    axl.step(t, live, where="post", lw=2, color="#C44E52")
    axl.axvline(0, color="gray", ls="--", lw=1)
    axl.text(a["total_drain_s"] * 0.01, n * 0.97, "decode start",
             fontsize=8, color="gray", va="top")
    axl.set_xlabel("time since decode start (s)")
    axl.set_ylabel("# live (decoding) requests")
    axl.set_title(f"Lifetime — {title}")
    axl.grid(alpha=0.3)
    axl.set_ylim(bottom=0)

    axr.bar(a["bin_labels"], a["bin_durations_s"], color="#4C72B0")
    axr.set_xlabel("live-request bin")
    axr.set_ylabel("duration (s)")
    axr.set_title(f"Tail-drain time per bin — {title}")
    axr.tick_params(axis="x", rotation=45)
    axr.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Rollout decode lifecycle — {title}   "
        f"(drain={a['total_drain_s']:.0f}s, out_len p50={a['output_len_p50']:.0f}/"
        f"p99={a['output_len_p99']:.0f}/max={a['output_len_max']})",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--expected-mnt", type=int, default=None,
                   help="benchmark max_new_tokens; auto-detected if omitted")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    a = analyze(run_dir, args.expected_mnt)
    (run_dir / "lifecycle.json").write_text(json.dumps(a, indent=2) + "\n")
    title = args.title or run_dir.name
    plot(a, title, run_dir / "lifecycle.png")
    print(f"wrote {run_dir}/lifecycle.json and lifecycle.png")
    print(f"  n={a['n']} drain={a['total_drain_s']:.1f}s reasons={a['finish_reasons']} "
          f"out_len p50={a['output_len_p50']:.0f} p99={a['output_len_p99']:.0f} "
          f"max={a['output_len_max']}")


if __name__ == "__main__":
    main()
