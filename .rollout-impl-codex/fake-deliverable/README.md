# Fake Dynamic Precision Deliverable

These files are synthetic. They define the intended output shape for the
rollout precision-switching implementation before real measurements exist.

## Understanding

The system should serve RL rollout with two available weight paths:

- BF16 merged weights: LoRA is merged into BF16 weights, so rollout has no
  adapter compute.
- INT4 base weights plus BF16 LoRA patch: the LoRA path uses the torch GEMM
  two-stream implementation instead of SGLang's default csgmv backend.

During decoding, the live batch shrinks because requests finish at different
times. The runtime should choose the faster precision per live-batch window,
using an offline or cached kernel-profile JSON. The decision can differ by
projection group: qkv, o, up, and down.

The main reported speedup is window-based:

1. Find a live-request interval, such as 64 to 32 live requests, in the BF16
   baseline lifetime trace.
2. Measure the BF16 baseline duration for that exact interval.
3. Measure or estimate the dynamic-policy duration over the same interval.
4. Report `baseline_duration / policy_duration`.
5. Repeat against the default INT4+csgmv QLoRA baseline.

## Fake Outputs

- `fake_precision_lifetime.png`: main expected plot. It overlays precision
  decisions on a rollout lifetime curve and annotates each window with
  `Q(...)` for the INT4+LoRA projections and speedups versus both baselines.
  The diamond marker denotes the end of the rollout.
- `fake_speedup_summary.png`: supplementary TP1/TP4 window speedup bars.
- `fake_vram_timeline.png`: supplementary VRAM sketch showing the cost of
  keeping both BF16 and INT4 weights resident, plus a future offload idea.
- `fake_precision_policy.json`: synthetic switch policy data that mirrors the
  config/report structure the real system should eventually produce.
