"""Final summary for the DeepSeek-R1-7B reasoning lifecycle sweep (32k cap)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/huanchen/sglang/rollout_precision_data/lifecycle")
FINAL = ROOT / "FINAL_r1_7b_reasoning_32k"
FINAL.mkdir(parents=True, exist_ok=True)

BATCHES = [128, 256, 512]
COLORS = {128: "#4C72B0", 256: "#DD8452", 512: "#C44E52"}


def run_dir(bs):
    return ROOT / f"r1_7b_reasoning_bs{bs}_cap32k"


data = {}
for b in BATCHES:
    d = run_dir(b)
    data[b] = json.loads((d / "lifecycle.json").read_text())
    shutil.copy(d / "lifecycle.png", FINAL / f"lifecycle_bs{b}.png")

# ---- combined lifetime overlay ----
fig, ax = plt.subplots(figsize=(10, 6))
for b in BATCHES:
    a = data[b]
    rel = np.array(a["rel_finish"])
    t = np.concatenate([[0.0], rel])
    live = np.concatenate([[a["n"]], a["n"] - np.arange(1, len(rel) + 1)])
    cap = a["finish_reasons"].get("length", 0)
    ax.step(t, live, where="post", lw=2, color=COLORS[b],
            label=f"bs{b} (drain {a['total_drain_s']:.0f}s, {cap} hit 32k cap)")
ax.set_xlabel("time since decode start (s)")
ax.set_ylabel("# live (decoding) requests")
ax.set_title("DeepSeek-R1-Distill-Qwen-7B — reasoning rollout decode lifetime\n"
             "(bf16, max_new=32768, temp=0.6/top_p=0.95/top_k=20)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FINAL / "summary_lifetime.png", dpi=140)
plt.close(fig)

# ---- drain time + cap-hit bar ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(BATCHES))
drains = [data[b]["total_drain_s"] for b in BATCHES]
ax1.bar(x, drains, color=[COLORS[b] for b in BATCHES])
for i, v in enumerate(drains):
    ax1.text(i, v, f"{v:.0f}s", ha="center", va="bottom")
ax1.set_xticks(x); ax1.set_xticklabels([f"bs{b}" for b in BATCHES])
ax1.set_ylabel("total drain time (s)"); ax1.set_title("Total decode-drain time")
ax1.grid(axis="y", alpha=0.3)

caps = [100.0 * data[b]["finish_reasons"].get("length", 0) / data[b]["n"] for b in BATCHES]
ax2.bar(x, caps, color=[COLORS[b] for b in BATCHES])
for i, v in enumerate(caps):
    ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
ax2.set_xticks(x); ax2.set_xticklabels([f"bs{b}" for b in BATCHES])
ax2.set_ylabel("% requests hitting 32k cap"); ax2.set_title("Cap-hit rate (finish_reason=length)")
ax2.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FINAL / "summary_drain_and_cap.png", dpi=140)
plt.close(fig)

# ---- index ----
lines = ["# DeepSeek-R1-Distill-Qwen-7B reasoning lifecycle (32k cap) — final\n",
         "bs | drain_s | out_len p50 | p99 | max | %cap | eos | length",
         "-- | ------- | ----------- | --- | --- | ---- | --- | ------"]
for b in BATCHES:
    a = data[b]
    n = a["n"]; ln = a["finish_reasons"].get("length", 0); st = a["finish_reasons"].get("stop", 0)
    lines.append(f"{b} | {a['total_drain_s']:.0f} | {a['output_len_p50']:.0f} | "
                 f"{a['output_len_p99']:.0f} | {a['output_len_max']} | "
                 f"{100*ln/n:.1f}% | {st} | {ln}")
(FINAL / "INDEX.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote final figures + INDEX.md to {FINAL}")
