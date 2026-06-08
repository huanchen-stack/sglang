# Real-Data Rollout Precision Deliverables

These figures combine the earlier estimator with actual live dynamic-precision
serving traces from this branch.

Inputs:

- Real request lifecycle traces: Qwen2.5-14B-Instruct on Eurus-2-RL, TP1/TP4, batch sizes 128/256/512.
- Real decode frontier: `.rollout-profile/qlora-decoding-throughput/frontier_decoding_throughput.json`.
- The estimator rescales each live-request bin independently by the frontier speedup for the nearest active batch size.
- Actual live dynamic traces: `.rollout-impl-codex/real-deliverable/live-runs/dynamic-qwen2.5-14b-eurus-tp4`.

Important caveat: the live dynamic traces are raw serving runs with stochastic
sampling. The bs512 dynamic run had one request hit the 32K length cap, so the
raw drain-time comparison is dominated by a different sampled long tail. The
estimator plots remain useful for controlled bin-by-bin projection of the BF16
lifecycle, while `real_live_dynamic_*` shows what the implemented system did.

In lifetime plots, the dotted vertical line plus diamond marker denotes the end
of the corresponding rollout trace.

Generated plots:

- `real_baseline_lifetime.png`: measured BF16 TP4 bs512 lifetime.
- `real_precision_lifetime.png`: estimated dynamic-precision TP4 bs512 lifetime on the recomputed clock.
- `real_lifetime_comparison.png`: BF16 measured, INT4 CSGMV estimated, and our dynamic estimate.
- `real_live_dynamic_lifetime.png`: actual BF16 baseline vs actual live dynamic TP4 bs512 trace.
- `real_live_dynamic_summary.png`: actual live dynamic drain time for TP4 bs128/256/512.
- `real_tp1_debug_breakdown.png`: actual one-GPU TP1 bs128 bin-by-bin debug comparison.
- `real_window_breakdown_tp4_bs512.png`: measured vs estimated time per live-request bin.
- `real_e2e_speedup_bars.png`: estimated full-drain speedup for TP1/TP4 and bs128/256/512.
- `real_frontier_throughput.png`: measured BF16/QLoRA serving frontier used by the estimator.

Best estimated row in this batch of data: TP1 bs256, x2.07 decode-drain speedup.
Best actual live dynamic row: TP4 bs128, x0.91 versus the prior BF16 trace.
