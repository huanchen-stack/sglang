#!/usr/bin/env python3
"""Launch SGLang and run CoT request-lifetime profiles."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


OUT_DIR = Path(__file__).resolve().parent
CLIENT = OUT_DIR / "cot_stream_client.py"
ANALYZER = OUT_DIR / "analyze_lifetime_cot.py"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_ready(base_url: str, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with rc={proc.returncode}")
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become ready at {base_url}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def build_server_cmd(config: dict, port: int, batch_size: int, python: str) -> list[str]:
    cmd = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config["model_path"],
        "--served-model-name",
        config.get("served_model_name", config["name"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        str(config.get("tp_size", 1)),
        "--dtype",
        config.get("dtype", "bfloat16"),
        "--max-running-requests",
        str(batch_size),
        "--cuda-graph-max-bs",
        str(batch_size),
        "--context-length",
        str(config.get("context_length", 32768)),
        *config.get("common_server_args", []),
    ]
    cuda_graph_bs = config.get("cuda_graph_bs_by_batch", {}).get(
        str(batch_size), config.get("cuda_graph_bs")
    )
    if cuda_graph_bs:
        cmd.extend(["--cuda-graph-bs", *[str(bs) for bs in cuda_graph_bs]])
    mem_fraction_static = config.get("mem_fraction_static")
    if mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(mem_fraction_static)])
    if config.get("lora_path"):
        lora_name = config.get("lora_name", "default")
        cmd.extend(
            [
                "--enable-lora",
                "--lora-paths",
                f"{lora_name}={config['lora_path']}",
                "--max-lora-rank",
                str(config.get("max_lora_rank", 16)),
                "--max-loras-per-batch",
                str(config.get("max_loras_per_batch", 1)),
            ]
        )
    if config.get("lora_backend"):
        cmd.extend(["--lora-backend", str(config["lora_backend"])])
    if config.get("rollout_precision_policy"):
        cmd.extend(
            ["--rollout-precision-policy", str(config["rollout_precision_policy"])]
        )
    if config.get("rollout_precision_int4_model_path"):
        cmd.extend(
            [
                "--rollout-precision-int4-model-path",
                str(config["rollout_precision_int4_model_path"]),
            ]
        )
    if config.get("rollout_precision_int4_load_format"):
        cmd.extend(
            [
                "--rollout-precision-int4-load-format",
                str(config["rollout_precision_int4_load_format"]),
            ]
        )
    if config.get("rollout_precision_assume_merged_bf16"):
        cmd.append("--rollout-precision-assume-merged-bf16")
    cmd.extend(config.get("extra_server_args", []))
    return cmd


def build_client_cmd(config: dict, port: int, batch_size: int, run_dir: Path, python: str) -> list[str]:
    cmd = [
        python,
        str(CLIENT),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--dataset-path",
        config["dataset_path"],
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(config.get("max_new_tokens", 32768)),
        "--temperature",
        str(config.get("temperature", 0.9)),
        "--top-p",
        str(config.get("top_p", 0.95)),
        "--top-k",
        str(config.get("top_k", 20)),
        "--stream-interval",
        str(config.get("stream_interval", 64)),
        "--stream-idle-timeout-s",
        str(config.get("stream_idle_timeout_s", 300)),
        "--prompt-suffix",
        config.get("prompt_suffix", ""),
        "--ready-timeout-s",
        str(config.get("ready_timeout_s", 300)),
        "--request-timeout-s",
        str(config.get("request_timeout_s", 21600)),
        "--output-dir",
        str(run_dir),
    ]
    if config.get("request_lora_path") is not None:
        cmd.extend(["--lora-path", str(config["request_lora_path"])])
    return cmd


def run_one(args: argparse.Namespace, config: dict, batch_size: int) -> None:
    config = dict(config)
    if args.tp_size is not None:
        config["tp_size"] = args.tp_size
    if args.mem_fraction_static is not None:
        config["mem_fraction_static"] = args.mem_fraction_static
    if args.context_length is not None:
        config["context_length"] = args.context_length
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.top_p is not None:
        config["top_p"] = args.top_p
    if args.top_k is not None:
        config["top_k"] = args.top_k
    if args.stream_interval is not None:
        config["stream_interval"] = args.stream_interval

    result_group = args.result_group or config.get("result_group")
    output_dir = args.output_dir / result_group if result_group else args.output_dir
    run_dir = output_dir / f"bs{batch_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = args.port
    base_url = f"http://127.0.0.1:{port}"

    server_cmd = build_server_cmd(config, port, batch_size, args.python)
    client_cmd = build_client_cmd(config, port, batch_size, run_dir, args.python)
    metadata = {
        "config": config,
        "batch_size": batch_size,
        "server_cmd": server_cmd,
        "client_cmd": client_cmd,
    }
    (run_dir / "commands.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[bs{batch_size}] SERVER: {' '.join(server_cmd)}", flush=True)
    print(f"[bs{batch_size}] CLIENT: {' '.join(client_cmd)}", flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpus or config.get("gpus", "0"))
    env.setdefault("HF_HOME", "/data/huggingface")
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})
    server_log = (run_dir / "server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        server_cmd,
        cwd=args.repo_root,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_ready(base_url, config.get("ready_timeout_s", 300), proc)
        subprocess.run(client_cmd, cwd=args.repo_root, env=env, check=True)
    finally:
        terminate(proc)
        server_log.close()

    subprocess.run(
        [args.python, str(ANALYZER), "--run-dir", str(run_dir)],
        cwd=args.repo_root,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR / "results")
    parser.add_argument(
        "--result-group",
        default=None,
        help="Optional subdirectory under output-dir, e.g. model-dataset-tp4.",
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream-interval", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    batch_sizes = args.batch_size or config.get("batch_sizes", [128, 256, 512])
    for batch_size in batch_sizes:
        run_one(args, config, int(batch_size))


if __name__ == "__main__":
    main()
