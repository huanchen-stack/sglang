"""Run bf16/int4 rollout precision experiment matrix on SGLang servers."""

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

import requests


DEFAULT_MODELS = {
    "llama_dense_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_dense_30b": "meta-llama/Llama-2-34b-chat-hf",
    "qwen_moe_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "glm_moe_30b": "zai-org/GLM-4.5-Air",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    model_label: str
    model_path: str
    precision: str
    dtype: str
    dataset_category: str
    dataset_path: Path
    batch_size: int
    tp_size: int
    gpu_ids: tuple[int, ...]
    port: int


def available_gpus() -> list[int]:
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(x) for x in visible.split(",") if x.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return [0]
    return [int(x.strip()) for x in out.splitlines() if x.strip()]


def wait_ready(
    base_url: str, timeout_s: int, proc: subprocess.Popen | None = None
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Server exited before becoming ready: {base_url} "
                f"(returncode={proc.returncode})"
            )
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready: {base_url}")


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
        str(exp.tp_size),
        "--dtype",
        exp.dtype,
        "--max-running-requests",
        str(exp.batch_size),
        "--cuda-graph-max-bs",
        str(exp.batch_size),
        "--piecewise-cuda-graph-max-tokens",
        str(args.piecewise_cuda_graph_max_tokens),
    ]
    if exp.precision == "int4" and args.int4_quantization != "auto":
        cmd += ["--quantization", args.int4_quantization]
    if args.server_arg:
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
        "--output-file",
        str(output_file),
        "--disable-tqdm",
        "--disable-stream",
    ]
    if args.output_len is not None:
        cmd += ["--sharegpt-output-len", str(args.output_len)]
    if args.disable_ignore_eos:
        cmd += ["--disable-ignore-eos"]
    if args.bench_arg:
        for item in args.bench_arg:
            cmd += item.split()
    return cmd


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def run_experiment(exp: Experiment, args: argparse.Namespace) -> None:
    exp_dir = Path(args.output_dir) / exp.name
    trace_dir = exp_dir / "traces"
    exp_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    failure_path = exp_dir / "failure.txt"

    if args.skip_existing and (exp_dir / "bench_serving.jsonl").exists():
        trace_files = list(trace_dir.glob("*.jsonl"))
        if trace_files and not failure_path.exists():
            print(f"SKIP {exp.name}; existing benchmark and traces found")
            return
    if failure_path.exists():
        failure_path.unlink()

    metadata = asdict(exp)
    metadata["dataset_path"] = str(exp.dataset_path)
    metadata["gpu_ids"] = list(exp.gpu_ids)
    (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    bench_output = exp_dir / "bench_serving.jsonl"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in exp.gpu_ids)
    env["SGLANG_ROLLOUT_TRACE_DIR"] = str(trace_dir)

    s_cmd = server_cmd(exp, args)
    b_cmd = bench_cmd(exp, args, bench_output)
    print("SERVER:", " ".join(s_cmd))
    print("BENCH:", " ".join(b_cmd))
    if args.dry_run:
        return

    server_log = (exp_dir / "server.log").open("w")
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


def run_experiment_recording_failure(
    exp: Experiment, args: argparse.Namespace
) -> None:
    try:
        run_experiment(exp, args)
    except Exception:
        exp_dir = Path(args.output_dir) / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAILED {exp.name}; wrote {exp_dir / 'failure.txt'}")


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    dataset_dir = Path(args.dataset_dir)
    datasets = {
        path.stem: path
        for path in sorted(dataset_dir.glob("*.jsonl"))
        if not args.categories or path.stem in args.categories
    }
    if not datasets:
        raise ValueError(f"No JSONL datasets found in {dataset_dir}")

    models = dict(DEFAULT_MODELS)
    for spec in args.model:
        label, path = spec.split("=", 1)
        models[label] = path
    int4_models = {}
    for spec in args.int4_model:
        label, path = spec.split("=", 1)
        int4_models[label] = path
    int4_dtypes = {}
    for spec in args.int4_dtype:
        label, dtype = spec.split("=", 1)
        int4_dtypes[label] = dtype
    if args.model_labels:
        models = {k: v for k, v in models.items() if k in args.model_labels}

    gpus = available_gpus()
    experiments = []
    port = args.base_port
    group_cursor = {tp_size: 0 for tp_size in args.tp_sizes}
    for tp_size in args.tp_sizes:
        gpu_groups = [
            tuple(gpus[start : start + tp_size])
            for start in range(0, len(gpus), tp_size)
            if len(gpus[start : start + tp_size]) == tp_size
        ]
        if not gpu_groups:
            continue

        for model_label, model_path in models.items():
            for precision, (category, dataset_path), batch_size in itertools.product(
                args.precisions,
                datasets.items(),
                args.batch_sizes,
            ):
                model_path_for_precision = (
                    int4_models.get(model_label, model_path)
                    if precision == "int4"
                    else model_path
                )
                dtype = (
                    int4_dtypes.get(model_label, "half")
                    if precision == "int4"
                    else "bfloat16"
                )
                cursor = group_cursor[tp_size]
                gpu_ids = gpu_groups[cursor % len(gpu_groups)]
                group_cursor[tp_size] = cursor + 1
                name = (
                    f"{model_label}_{precision}_{category}"
                    f"_bs{batch_size}_tp{tp_size}_gpus{'-'.join(map(str, gpu_ids))}"
                )
                experiments.append(
                    Experiment(
                        name=name,
                        model_label=model_label,
                        model_path=model_path_for_precision,
                        precision=precision,
                        dtype=dtype,
                        dataset_category=category,
                        dataset_path=dataset_path,
                        batch_size=batch_size,
                        tp_size=tp_size,
                        gpu_ids=gpu_ids,
                        port=port,
                    )
                )
                port += 1
    return experiments


def run_experiments(experiments: list[Experiment], args: argparse.Namespace) -> None:
    if args.dry_run:
        for exp in experiments:
            run_experiment_recording_failure(exp, args)
        return

    for tp_size in args.tp_sizes:
        tp_experiments = [exp for exp in experiments if exp.tp_size == tp_size]
        if not tp_experiments:
            continue

        by_gpu_group: dict[tuple[int, ...], list[Experiment]] = {}
        for exp in tp_experiments:
            by_gpu_group.setdefault(exp.gpu_ids, []).append(exp)

        print(
            f"Running {len(tp_experiments)} tp{tp_size} experiments "
            f"on {len(by_gpu_group)} GPU groups"
        )
        with ThreadPoolExecutor(max_workers=len(by_gpu_group)) as executor:
            futures = [
                executor.submit(
                    lambda group_exps: [
                        run_experiment_recording_failure(group_exp, args)
                        for group_exp in group_exps
                    ],
                    group_experiments,
                )
                for group_experiments in by_gpu_group.values()
            ]
            for future in futures:
                future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-port", type=int, default=31000)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip experiment directories that already contain benchmark output and traces.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--piecewise-cuda-graph-max-tokens",
        type=int,
        default=8192,
        help="Maximum prefill token count to cover during piecewise CUDA graph capture.",
    )
    parser.add_argument("--precisions", nargs="+", default=["bf16", "int4"])
    parser.add_argument("--categories", nargs="+", default=[])
    parser.add_argument("--model-labels", nargs="+", default=[])
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--int4-model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Precision-specific int4 checkpoint path for a model label.",
    )
    parser.add_argument(
        "--int4-dtype",
        action="append",
        default=[],
        metavar="LABEL=DTYPE",
        help="Precision-specific dtype override for an int4 model label.",
    )
    parser.add_argument(
        "--output-len",
        type=str,
        default=1024,
        help=(
            "Fixed benchmark output length. Use --output-len none to keep "
            "per-row dataset output lengths."
        ),
    )
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Pass --disable-ignore-eos to bench_serving so EOS can stop generation.",
    )
    parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
        help="Extra bench_serving args, repeated as needed.",
    )
    parser.add_argument(
        "--int4-quantization",
        default="auto",
        help="Quantization backend for int4 runs. Use 'auto' to infer from checkpoint config.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra launch_server args, repeated as needed.",
    )
    args = parser.parse_args()
    if isinstance(args.output_len, str) and args.output_len.lower() == "none":
        args.output_len = None
    return args


def main() -> None:
    args = parse_args()
    experiments = build_experiments(args)
    print(f"Prepared {len(experiments)} experiments")
    run_experiments(experiments, args)


if __name__ == "__main__":
    main()
