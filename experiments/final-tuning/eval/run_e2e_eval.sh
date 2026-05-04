#!/bin/bash
# E2E ShareGPT bench_serving comparison: baseline (pure INT4) vs treatment
# (heter-MoE with autotuned tiles + EfficiencyPromotionPolicy).
#
# Usage: ./run_e2e_eval.sh [config.yaml]   (default: configs/run_e2e.yaml)
#
# Two servers in parallel; bench_serving runs sequentially per mc against each.
set -o pipefail
# Note: set -u causes conda activate to fail on unset PS1; intentionally omitted

cd "$(dirname "$0")/../../.."

EVAL_DIR=experiments/final-tuning/eval
CFG_FILE="${1:-$EVAL_DIR/configs/run_e2e.yaml}"
LOG_DIR=$EVAL_DIR/logs
RESULT_DIR=$EVAL_DIR/results
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Per-run timestamp suffix so concurrent / repeated runs don't clobber logs.
# Format: YYYYMMDD_HHMMSS (sortable, grep-friendly).
TS=$(date +%Y%m%d_%H%M%S)
DRIVER_LOG="$LOG_DIR/driver_${TS}.log"

if [ ! -f "$CFG_FILE" ]; then
    echo "ERROR: config file not found: $CFG_FILE" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate sglang

# ---------------- YAML loader ----------------
yaml_get() {
    # Usage: yaml_get <file> <dotted.key>  (lists print space-separated)
    python - "$CFG_FILE" "$1" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
v = cfg
for k in sys.argv[2].split('.'):
    v = v[k]
if isinstance(v, list):
    print(' '.join(str(x) for x in v))
else:
    print(v)
PYEOF
}

resolve() { case "$1" in /*) echo "$1" ;; *) echo "$EVAL_DIR/$1" ;; esac; }

NUM_PROMPTS=$(yaml_get num_prompts)
# Sentinel: num_prompts=-1 means "run every available sample". The ShareGPT
# sampler stops at min(num_requests, len(filtered_dataset)), so a value far
# beyond the dataset size effectively unlocks the full corpus.
NUM_PROMPTS_LABEL="$NUM_PROMPTS"
if [ "$NUM_PROMPTS" -lt 0 ]; then
    NUM_PROMPTS=1000000000
    NUM_PROMPTS_LABEL="all"
fi
MC_LIST=($(yaml_get max_concurrency))

BF16_MODEL=$(yaml_get model.path)
TP_SIZE=$(yaml_get model.tp_size)
MEM_FRACTION=$(yaml_get model.mem_fraction_static)
KV_CACHE_NUM_BITS=$(yaml_get model.kv_cache_num_bits)

# Map kv_cache_num_bits -> sglang launch flags. 16 = no override (default).
KV_CACHE_FLAGS=()
case "$KV_CACHE_NUM_BITS" in
    16) KV_CACHE_LABEL="default" ;;
    8)
        KV_CACHE_FLAGS=(--kv-cache-dtype fp8_e4m3 --attention-backend triton)
        KV_CACHE_LABEL="fp8_e4m3+triton"
        ;;
    4)
        KV_CACHE_FLAGS=(--kv-cache-dtype fp4_e2m1 --attention-backend triton)
        KV_CACHE_LABEL="fp4_e2m1+triton"
        ;;
    *)
        echo "ERROR: model.kv_cache_num_bits must be 16, 8, or 4 (got '$KV_CACHE_NUM_BITS')" >&2
        exit 1
        ;;
esac

DATASET_NAME=$(yaml_get dataset.name)
DATASET_PATH=$(yaml_get dataset.path)

BASELINE_CFG=$(resolve "$(yaml_get servers.baseline.heter_config)")
BASELINE_GPUS=$(yaml_get servers.baseline.gpu)
BASELINE_PORT=$(yaml_get servers.baseline.port)

TREATMENT_CFG=$(resolve "$(yaml_get servers.treatment.heter_config)")
TREATMENT_GPUS=$(yaml_get servers.treatment.gpu)
TREATMENT_PORT=$(yaml_get servers.treatment.port)

READY_TIMEOUT=$(yaml_get server_ready_timeout_s)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG" >&2; }

log "config: $CFG_FILE"
log "num_prompts=$NUM_PROMPTS_LABEL  mc=[${MC_LIST[*]}]  kv_cache=${KV_CACHE_NUM_BITS}-bit (${KV_CACHE_LABEL})"
log "baseline: GPU $BASELINE_GPUS port $BASELINE_PORT  cfg=$BASELINE_CFG"
log "treatment: GPU $TREATMENT_GPUS port $TREATMENT_PORT  cfg=$TREATMENT_CFG"

# ---------------- launch + health check ----------------
launch_server() {
    local label=$1 cfg=$2 port=$3 gpus=$4 logfile=$5
    log "launching $label on GPUs $gpus port $port"
    CUDA_VISIBLE_DEVICES=$gpus python -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --tp "$TP_SIZE" \
        --host 127.0.0.1 --port "$port" \
        --trust-remote-code \
        --heter-precision-config "$cfg" \
        --mem-fraction-static "$MEM_FRACTION" \
        "${KV_CACHE_FLAGS[@]}" \
        > "$logfile" 2>&1 &
    echo $!
}

wait_ready() {
    local port=$1 pid=$2 elapsed=0
    while ! curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "  ERROR: server pid $pid died (port $port)"
            return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge "$READY_TIMEOUT" ]; then
            log "  ERROR: server pid $pid did not become ready in ${READY_TIMEOUT}s"
            return 1
        fi
    done
    log "  server ready (port $port) after ${elapsed}s"
}

bench_one() {
    local label=$1 port=$2 mc=$3
    local out=$RESULT_DIR/${label}_mc${mc}.jsonl
    local benchlog=$LOG_DIR/${label}_mc${mc}_${TS}.log
    log "bench_serving $label mc=$mc → $out"
    curl -s -X POST "http://127.0.0.1:${port}/flush_cache" > /dev/null 2>&1 || true
    sleep 1
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${port}" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$mc" \
        --output-file "$out" \
        > "$benchlog" 2>&1 || log "  bench failed (see $benchlog)"
}

# ---------------- main ----------------
# IMPORTANT: launch servers SEQUENTIALLY (not in parallel) so that each server
# transitions directly from cudagraph-capture (GPU at boost clock) into its
# own bench. Running in parallel let the fast-to-ready server sit idle for
# ~14 min while the slow one finished capture, biasing its TTFT upward
# because the GPU had clocked down.
START=$(date +%s)

CURRENT_PID=""
trap 'kill -9 $CURRENT_PID 2>/dev/null; exit' INT TERM

run_server_and_bench() {
    local label=$1 cfg=$2 port=$3 gpus=$4 logfile=$5

    log "=== $label: launching ==="
    local pid
    pid=$(launch_server "$label" "$cfg" "$port" "$gpus" "$logfile")
    CURRENT_PID=$pid
    log "$label pid=$pid  (sequential — no peer server running)"

    if ! wait_ready "$port" "$pid"; then
        log "$label failed to become ready"
        kill -9 "$pid" 2>/dev/null
        return 1
    fi

    for mc in "${MC_LIST[@]}"; do
        bench_one "$label" "$port" "$mc"
    done

    log "$label: tearing down (pid=$pid)"
    kill -INT "$pid" 2>/dev/null
    sleep 5
    kill -9 "$pid" 2>/dev/null
    CURRENT_PID=""
}

run_server_and_bench "baseline" "$BASELINE_CFG" \
    "$BASELINE_PORT" "$BASELINE_GPUS" "$LOG_DIR/server_baseline_${TS}.log" \
    || { log "baseline run failed"; exit 1; }

run_server_and_bench "treatment" "$TREATMENT_CFG" \
    "$TREATMENT_PORT" "$TREATMENT_GPUS" "$LOG_DIR/server_treatment_${TS}.log" \
    || { log "treatment run failed"; exit 1; }

END=$(date +%s)
log "DONE in $((END - START))s"
log "  results: $RESULT_DIR/"

# Aggregate to side-by-side CSV
python - << PYEOF 2>&1 | tee -a "$DRIVER_LOG"
import csv, glob, json, os
out_path = "$RESULT_DIR/sharegpt_e2e.csv"
rows = []
for f in sorted(glob.glob("$RESULT_DIR/*.jsonl")):
    name = os.path.basename(f).replace(".jsonl", "")
    label, mc = name.rsplit("_mc", 1)
    with open(f) as fp:
        for line in fp:
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows.append({
                "label": label, "mc": int(mc),
                "completed": d.get("completed"),
                "request_throughput": d.get("request_throughput"),
                "total_input_tokens": d.get("total_input_tokens"),
                "total_output_tokens": d.get("total_output_tokens"),
                "input_throughput": d.get("input_throughput"),
                "output_throughput": d.get("output_throughput"),
                "mean_ttft_ms": d.get("mean_ttft_ms"),
                "median_ttft_ms": d.get("median_ttft_ms"),
                "p99_ttft_ms": d.get("p99_ttft_ms"),
                "mean_itl_ms": d.get("mean_itl_ms"),
                "median_itl_ms": d.get("median_itl_ms"),
                "p99_itl_ms": d.get("p99_itl_ms"),
                "mean_e2el_ms": d.get("mean_e2el_ms"),
                "median_e2el_ms": d.get("median_e2el_ms"),
                "p99_e2el_ms": d.get("p99_e2el_ms"),
            })
if rows:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")
else:
    print("no results to aggregate")
PYEOF
