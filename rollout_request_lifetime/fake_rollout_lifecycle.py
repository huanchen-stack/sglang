"""
Fake-data mockup of the rollout lifecycle experiment (layout v2).

Per batch size in {128, 256, 512, 1024} we produce ONE figure with TWO subplots:
  left  : lifetime  -> x = time (s) since decode start, y = # live (decoding) reqs
  right : duration  -> x = halving live-count bins {N..N/2, ..., 8-0}
                       y = wall-clock seconds spent while live-count was in that bin

Workload mimic (real run will use qwen2.5-14b / qwen2.5-32b, bf16):
  - launch a batch of N reqs together, max_new_tokens = 16384
  - each request decodes until it emits its OWN eos (no early stop / shared cap)
  - t = 0 == decode start (whole-batch prefill done)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "fake_lifecycle_plots")
os.makedirs(OUTDIR, exist_ok=True)

RNG = np.random.default_rng(0)

MAX_NEW_TOKENS = 16384
TOKENS_PER_SEC = 55.0  # fake per-request decode rate; real value comes from server
BATCH_SIZES = [128, 256, 512, 1024]


def fake_completion_lengths(n):
    """Reasoning/agent outputs: long, heavy-tailed. Lognormal clipped to 16k."""
    lengths = RNG.lognormal(mean=np.log(3500), sigma=0.7, size=n)
    return np.clip(lengths, 64, MAX_NEW_TOKENS)


def lifecycle(n):
    lengths = fake_completion_lengths(n)
    finish = lengths / TOKENS_PER_SEC * RNG.uniform(0.9, 1.1, size=n)
    finish = np.sort(finish)
    t_grid = np.linspace(0, finish.max(), 2000)
    live = n - np.searchsorted(finish, t_grid, side="right")
    return finish, t_grid, live


def halving_bins(n):
    edges, hi = [], n
    while hi > 8:
        lo = hi // 2
        edges.append((lo, hi))
        hi = lo
    edges.append((0, hi))
    return edges  # list of (lo, hi); bin covers live in (lo, hi]


def time_in_bins(finish, n):
    bins = halving_bins(n)
    durations = np.zeros(len(bins))
    cur_live, last_t = n, 0.0
    for t in finish:  # live drops by 1 just after each finish
        dur = t - last_t
        for i, (lo, hi) in enumerate(bins):
            if lo < cur_live <= hi:
                durations[i] += dur
                break
        last_t, cur_live = t, cur_live - 1
    labels = [f"{hi}-{lo}" for (lo, hi) in bins]
    return labels, durations


for n in BATCH_SIZES:
    finish, t_grid, live = lifecycle(n)
    labels, durations = time_in_bins(finish, n)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))

    # left: lifetime
    axl.plot(t_grid, live, lw=2, color="#C44E52")
    axl.axvline(0, color="gray", ls="--", lw=1)
    axl.text(t_grid.max() * 0.01, n * 0.97, "decode start\n(prefill done)",
             fontsize=8, color="gray", va="top")
    axl.set_xlabel("time since decode start (s)")
    axl.set_ylabel("# live (decoding) requests")
    axl.set_title(f"Lifetime — batch={n}")
    axl.grid(alpha=0.3)

    # right: duration per live-count bin
    axr.bar(labels, durations, color="#4C72B0")
    axr.set_xlabel("live-request bin")
    axr.set_ylabel("duration (s)")
    axr.set_title(f"Tail-drain time per live-count bin — batch={n}")
    axr.tick_params(axis="x", rotation=45)
    axr.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Rollout decode lifecycle  [FAKE DATA]  —  batch size {n}",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"lifecycle_bs{n}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)
