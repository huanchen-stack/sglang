"""Build combined summary figures + an index from the 6 reasoning lifecycle runs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/huanchen/sglang/rollout_precision_data/lifecycle")
FINAL = ROOT / "FINAL_reasoning"
FINAL.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen2.5-14b", "qwen2.5-32b"]
BATCHES = [128, 256, 512]
COLORS = {128: "#4C72B0", 256: "#DD8452", 512: "#C44E52"}


def run_dir(model, bs):
    # bs512 14b comes from phase1; everything else phase2
    if model == "qwen2.5-14b" and bs == 512:
        return ROOT / "phase1_14b_reasoning_bs512"
    return ROOT / f"phase2_{model}_reasoning_bs{bs}"


data = {}
for m in MODELS:
    for b in BATCHES:
        d = run_dir(m, b)
        a = json.loads((d / "lifecycle.json").read_text())
        data[(m, b)] = a
        # copy the per-run two-panel figure into FINAL with a tidy name
        shutil.copy(d / "lifecycle.png", FINAL / f"lifecycle_{m}_bs{b}.png")


# ---- Summary 1: lifetime curves, one panel per model ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for ax, m in zip(axes, MODELS):
    for b in BATCHES:
        a = data[(m, b)]
        rel = np.array(a["rel_finish"])
        t = np.concatenate([[0.0], rel])
        live = np.concatenate([[a["n"]], a["n"] - np.arange(1, len(rel) + 1)])
        ax.step(t, live, where="post", lw=2, color=COLORS[b],
                label=f"bs{b} (drain {a['total_drain_s']:.0f}s)")
    ax.set_title(m)
    ax.set_xlabel("time since decode start (s)")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("# live (decoding) requests")
fig.suptitle("Rollout decode lifetime — reasoning (Qwen2.5, bf16, max_new=16384, temp=1.0)",
             fontsize=13)
fig.tight_layout()
fig.savefig(FINAL / "summary_lifetime.png", dpi=140)
plt.close(fig)

# ---- Summary 2: total drain time grouped bar ----
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(BATCHES))
w = 0.38
for i, m in enumerate(MODELS):
    vals = [data[(m, b)]["total_drain_s"] for b in BATCHES]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=m)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}s",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"bs{b}" for b in BATCHES])
ax.set_ylabel("total drain time (s)")
ax.set_title("Total decode-drain time — reasoning rollout")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FINAL / "summary_drain_time.png", dpi=140)
plt.close(fig)

# ---- Index table ----
lines = ["# Reasoning rollout decode-lifecycle — final results\n",
         "model | bs | drain_s | out_len p50 | p99 | max | finish_reasons",
         "----- | -- | ------- | ----------- | --- | --- | --------------"]
for m in MODELS:
    for b in BATCHES:
        a = data[(m, b)]
        lines.append(f"{m} | {b} | {a['total_drain_s']:.1f} | "
                     f"{a['output_len_p50']:.0f} | {a['output_len_p99']:.0f} | "
                     f"{a['output_len_max']} | {a['finish_reasons']}")
(FINAL / "INDEX.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote final figures + INDEX.md to {FINAL}")
