"""Run full-server LoRA-forward rollout throughput experiments.

Modes:
  bf16_full:            bf16 base checkpoint
  marlin_int4_full:     INT4/Marlin-compatible checkpoint
  qlora_marlin_bf16:    INT4/Marlin-compatible checkpoint plus bf16 rank-16 LoRA
  qlora_parallel_streams:
                       INT4 checkpoint plus bf16 rank-16 LoRA, tagged for the
                       separate-stream LoRA patch experiment

This runner deliberately forces torch.compile and CUDA Graph settings. If the
server auto-disables them or the trace indicates decode did not use CUDA Graph,
the experiment is recorded as a failure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import torch

from lora_forward_eval.create_qwen25_lora_adapter import create_adapter


DEFAULT_MODELS = {
    "qwen2.5-14b": {
        "bf16": "Qwen/Qwen2.5-14B-Instruct",
        "int4": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
    },
    "qwen2.5-32b": {
        "bf16": "Qwen/Qwen2.5-32B-Instruct",
        "int4": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    },
}

# NOTE: The separate-stream LoRA patch deliberately excludes the fused qkv_proj.
# qkv is a merged projection that sits on the critical attention path, and
# patching it on a side stream does not help (and breaks piecewise-CUDA-graph
# capture). Only the o/gate/up/down projections are patched.
TARGET_MODULES = [
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

QLORA_MODES = {"qlora_marlin_bf16", "qlora_parallel_streams"}
MODES = ["bf16_full", "marlin_int4_full", "qlora_marlin_bf16", "qlora_parallel_streams"]


@dataclass(frozen=True)
class Experiment:
    name: str
    model_label: str
    mode: str
    model_path: str
    adapter_path: str | None
    dataset_category: str
    dataset_path: Path
    batch_size: int
    gpu_id: int
    port: int


def available_gpus() -> list[int]:
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(x) for x in visible.split(",") if x.strip()]
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    )
    return [int(x.strip()) for x in out.splitlines() if x.strip()]


def wait_ready(base_url: str, timeout_s: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited before becoming ready: {base_url}; "
                f"returncode={proc.returncode}"
            )
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready: {base_url}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def server_cmd(exp: Experiment, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        exp.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(exp.port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16" if exp.mode == "bf16_full" else args.int4_dtype,
        "--max-running-requests",
        str(exp.batch_size),
        "--cuda-graph-max-bs",
        str(exp.batch_size),
        "--enable-torch-compile",
        "--torch-compile-max-bs",
        str(exp.batch_size),
        "--enforce-piecewise-cuda-graph",
        "--piecewise-cuda-graph-compiler",
        "inductor",
        "--piecewise-cuda-graph-max-tokens",
        str(args.piecewise_cuda_graph_max_tokens),
    ]
    if args.skip_server_warmup:
        cmd += ["--skip-server-warmup"]
    if exp.mode != "bf16_full" and args.int4_quantization != "auto":
        cmd += ["--quantization", args.int4_quantization]
    if exp.mode in QLORA_MODES:
        if exp.adapter_path is None:
            raise ValueError("QLoRA mode requires adapter_path")
        cmd += [
            "--lora-paths",
            f"qlora={exp.adapter_path}",
            "--max-lora-rank",
            "32",
            "--lora-target-modules",
            *TARGET_MODULES,
            "--max-loras-per-batch",
            "1",
            "--max-loaded-loras",
            "1",
        ]
    for item in args.server_arg:
        cmd += item.split()
    return cmd


def bench_cmd(exp: Experiment, args: argparse.Namespace, output_file: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{exp.port}",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(exp.dataset_path),
        "--num-prompts",
        str(exp.batch_size),
        "--max-concurrency",
        str(exp.batch_size),
        "--request-rate",
        "inf",
        "--sharegpt-output-len",
        str(args.output_len),
        "--output-file",
        str(output_file),
        "--disable-tqdm",
        "--disable-stream",
        "--warmup-requests",
        str(args.bench_warmup_requests),
    ]
    if not args.force_ignore_eos:
        cmd += ["--disable-ignore-eos"]
    if exp.mode in QLORA_MODES:
        cmd += ["--lora-name", "qlora"]
    for item in args.bench_arg:
        cmd += item.split()
    return cmd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_forced_graphs(exp_dir: Path) -> dict[str, Any]:
    server_log = exp_dir / "server.log"
    text = server_log.read_text(errors="replace") if server_log.exists() else ""
    bad_markers = [
        "Disabling torch.compile",
        "Disable piecewise CUDA graph",
        "disable piecewise cuda graph",
        "disable torch compile",
    ]
    marker_hits = [marker for marker in bad_markers if marker in text]

    trace_rows = []
    for path in sorted((exp_dir / "traces").glob("*.jsonl")):
        trace_rows.extend(read_jsonl(path))
    decode_values = [
        row.get("can_run_cuda_graph")
        for row in trace_rows
        if row.get("phase") == "decode"
        and row.get("event") == "phase_end"
        and row.get("can_run_cuda_graph") is not None
    ]
    decode_cuda_graph_fraction = (
        sum(1 for value in decode_values if value) / len(decode_values)
        if decode_values
        else None
    )
    ok = not marker_hits and (
        decode_cuda_graph_fraction is None or decode_cuda_graph_fraction == 1.0
    )
    return {
        "ok": ok,
        "marker_hits": marker_hits,
        "trace_decode_cuda_graph_fraction": decode_cuda_graph_fraction,
        "num_trace_events": len(trace_rows),
    }


def summarize_decode_trace(exp_dir: Path, expected_output_len: int) -> dict[str, Any]:
    """Compute decode-only throughput from scheduler lifecycle traces."""
    trace_rows = []
    for path in sorted((exp_dir / "traces").glob("*.jsonl")):
        trace_rows.extend(read_jsonl(path))

    completions = [
        row
        for row in trace_rows
        if row.get("event") == "request_complete"
        and row.get("phase") == "decode"
        and row.get("max_new_tokens") == expected_output_len
        and not str(row.get("rid", "")).startswith("HEALTH_CHECK")
    ]
    benchmark_rids = {row["rid"] for row in completions if row.get("rid")}
    decode_rows = [
        row
        for row in trace_rows
        if row.get("phase") == "decode"
        and row.get("rid") in benchmark_rids
        and row.get("batch_id") is not None
    ]
    start_rows = [row for row in decode_rows if row.get("event") == "phase_start"]
    end_rows = [row for row in decode_rows if row.get("event") == "phase_end"]

    total_completion_tokens = sum(int(row.get("output_len") or 0) for row in completions)
    first_decode_ts = min((row["ts"] for row in start_rows), default=None)
    last_complete_ts = max((row["ts"] for row in completions), default=None)
    decode_wall_s = (
        last_complete_ts - first_decode_ts
        if first_decode_ts is not None and last_complete_ts is not None
        else None
    )

    batch_starts: dict[int, float] = {}
    batch_ends: dict[int, float] = {}
    batch_tokens: dict[int, int] = {}
    batch_cuda_graph: dict[int, bool] = {}
    for row in start_rows:
        batch_id = int(row["batch_id"])
        batch_starts[batch_id] = min(batch_starts.get(batch_id, row["ts"]), row["ts"])
    for row in end_rows:
        batch_id = int(row["batch_id"])
        batch_ends[batch_id] = max(batch_ends.get(batch_id, row["ts"]), row["ts"])
        batch_tokens[batch_id] = batch_tokens.get(batch_id, 0) + 1
        if row.get("can_run_cuda_graph") is not None:
            batch_cuda_graph[batch_id] = bool(row["can_run_cuda_graph"])

    forward_s = 0.0
    forward_tokens = 0
    for batch_id in sorted(batch_tokens):
        if batch_id not in batch_starts or batch_id not in batch_ends:
            continue
        duration = batch_ends[batch_id] - batch_starts[batch_id]
        if duration <= 0:
            continue
        forward_s += duration
        forward_tokens += batch_tokens[batch_id]

    cuda_graph_values = list(batch_cuda_graph.values())
    cuda_graph_fraction = (
        sum(1 for value in cuda_graph_values if value) / len(cuda_graph_values)
        if cuda_graph_values
        else None
    )

    return {
        "num_completed_requests": len(completions),
        "expected_output_len": expected_output_len,
        "decode_completion_tokens": total_completion_tokens,
        "decode_wall_s": decode_wall_s,
        "decode_wall_tokens_per_s": (
            total_completion_tokens / decode_wall_s if decode_wall_s else None
        ),
        "decode_forward_tokens": forward_tokens,
        "decode_forward_s": forward_s,
        "decode_forward_tokens_per_s": (
            forward_tokens / forward_s if forward_s else None
        ),
        "decode_batch_steps": len(batch_tokens),
        "decode_cuda_graph_fraction": cuda_graph_fraction,
    }


def run_experiment(exp: Experiment, args: argparse.Namespace) -> None:
    exp_dir = Path(args.output_dir) / exp.name
    trace_dir = exp_dir / "traces"
    exp_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    failure_path = exp_dir / "failure.txt"

    if args.skip_existing and (exp_dir / "bench_serving.jsonl").exists():
        if not failure_path.exists():
            print(f"SKIP {exp.name}")
            return
    if failure_path.exists():
        failure_path.unlink()

    metadata = asdict(exp)
    metadata["dataset_path"] = str(exp.dataset_path)
    (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(exp.gpu_id)
    env["SGLANG_ROLLOUT_TRACE_DIR"] = str(trace_dir)
    env["SGLANG_ENABLE_TORCH_COMPILE"] = "1"
    if args.skip_server_warmup:
        env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
    if exp.mode == "qlora_parallel_streams":
        env["SGLANG_LORA_PARALLEL_STREAMS"] = "1"

    s_cmd = server_cmd(exp, args)
    b_cmd = bench_cmd(exp, args, exp_dir / "bench_serving.jsonl")
    (exp_dir / "commands.json").write_text(
        json.dumps({"server": s_cmd, "bench": b_cmd}, indent=2) + "\n"
    )
    print("SERVER:", " ".join(s_cmd))
    print("BENCH:", " ".join(b_cmd))
    if args.dry_run:
        return

    server_log = (exp_dir / "server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        s_cmd,
        cwd=args.repo_root,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_ready(f"http://127.0.0.1:{exp.port}", args.ready_timeout, proc)
        subprocess.run(b_cmd, cwd=args.repo_root, env=env, check=True)
    finally:
        terminate(proc)
        server_log.close()

    validation = validate_forced_graphs(exp_dir)
    (exp_dir / "forced_graph_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    decode_summary = summarize_decode_trace(exp_dir, args.output_len)
    (exp_dir / "decode_summary.json").write_text(
        json.dumps(decode_summary, indent=2) + "\n"
    )
    if not validation["ok"]:
        raise RuntimeError(f"Forced graph validation failed: {validation}")


def run_experiment_recording_failure(exp: Experiment, args: argparse.Namespace) -> None:
    try:
        run_experiment(exp, args)
    except Exception:
        exp_dir = Path(args.output_dir) / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAILED {exp.name}; wrote {exp_dir / 'failure.txt'}")


def maybe_create_adapters(args: argparse.Namespace) -> dict[str, str]:
    adapter_paths = {}
    explicit = {}
    for spec in args.adapter_path:
        label, path = spec.split("=", 1)
        explicit[label] = path
    for model_label in args.model_labels:
        if model_label in explicit:
            adapter_paths[model_label] = explicit[model_label]
            continue
        out_dir = Path(args.output_dir) / "adapters" / f"{model_label}_rank16_zero_bf16"
        if not (out_dir / "adapter_model.safetensors").exists():
            create_adapter(model_label, out_dir, rank=16, dtype=torch.bfloat16)
        adapter_paths[model_label] = str(out_dir)
    return adapter_paths


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    dataset_dir = Path(args.dataset_dir)
    datasets = {
        path.stem: path
        for path in sorted(dataset_dir.glob("*.jsonl"))
        if not args.categories or path.stem in args.categories
    }
    if not datasets:
        raise ValueError(f"No JSONL datasets found in {dataset_dir}")

    model_paths = {label: dict(paths) for label, paths in DEFAULT_MODELS.items()}
    for spec in args.model:
        label, rest = spec.split("=", 1)
        precision, path = rest.split(":", 1)
        model_paths.setdefault(label, {})[precision] = path

    model_labels = args.model_labels or list(DEFAULT_MODELS)
    args.model_labels = model_labels
    adapter_paths = maybe_create_adapters(args)
    gpus = available_gpus()
    if args.gpus:
        gpus = args.gpus
    if not gpus:
        raise RuntimeError("No GPUs available")

    experiments = []
    port = args.base_port
    gpu_cursor = 0
    for model_label, mode, (category, dataset_path), batch_size in itertools.product(
        model_labels, args.modes, datasets.items(), args.batch_sizes
    ):
        paths = model_paths[model_label]
        model_path = paths["bf16"] if mode == "bf16_full" else paths["int4"]
        adapter_path = adapter_paths[model_label] if mode in QLORA_MODES else None
        gpu_id = gpus[gpu_cursor % len(gpus)]
        gpu_cursor += 1
        name = f"{model_label}_{mode}_{category}_bs{batch_size}_gpu{gpu_id}"
        experiments.append(
            Experiment(
                name=name,
                model_label=model_label,
                mode=mode,
                model_path=model_path,
                adapter_path=adapter_path,
                dataset_category=category,
                dataset_path=dataset_path,
                batch_size=batch_size,
                gpu_id=gpu_id,
                port=port,
            )
        )
        port += 1
    return experiments


def run_experiments(experiments: list[Experiment], args: argparse.Namespace) -> None:
    if args.dry_run or args.parallelism == 1:
        for exp in experiments:
            run_experiment_recording_failure(exp, args)
        return

    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [
            executor.submit(run_experiment_recording_failure, exp, args)
            for exp in experiments
        ]
        for future in futures:
            future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-port", type=int, default=32000)
    parser.add_argument("--ready-timeout", type=int, default=2400)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-labels", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--categories", nargs="+", default=["math", "agentic", "reasoning"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--output-len", type=int, default=5000)
    parser.add_argument("--gpus", type=int, nargs="+", default=[])
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--int4-dtype", default="half")
    parser.add_argument("--int4-quantization", default="auto")
    parser.add_argument(
        "--piecewise-cuda-graph-max-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--skip-server-warmup",
        action="store_true",
        help="Skip SGLang's built-in warmup request; graph capture remains enabled.",
    )
    parser.add_argument(
        "--bench-warmup-requests",
        type=int,
        default=0,
        help=(
            "Number of bench_serving warmup requests. Default 0 avoids a duplicated "
            "prompt entering prefix-cache reuse before decode throughput measurement."
        ),
    )
    parser.add_argument(
        "--force-ignore-eos",
        action="store_true",
        help="Do not pass --disable-ignore-eos. By default EOS is allowed to stop generation.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=bf16|int4:PATH",
        help="Override a model path, e.g. qwen2.5-14b=int4:/path/to/gptq.",
    )
    parser.add_argument(
        "--adapter-path",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Use an existing LoRA adapter instead of generating a synthetic one.",
    )
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--bench-arg", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = build_experiments(args)
    print(f"Prepared {len(experiments)} experiments")
    run_experiments(experiments, args)


if __name__ == "__main__":
    main()
