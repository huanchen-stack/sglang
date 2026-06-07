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
    mem_fraction_static = config.get("mem_fraction_static")
    if mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(mem_fraction_static)])
    return cmd


def build_client_cmd(config: dict, port: int, batch_size: int, run_dir: Path, python: str) -> list[str]:
    return [
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


def run_one(args: argparse.Namespace, config: dict, batch_size: int) -> None:
    config = dict(config)
    if args.tp_size is not None:
        config["tp_size"] = args.tp_size
    if args.mem_fraction_static is not None:
        config["mem_fraction_static"] = args.mem_fraction_static
    if args.context_length is not None:
        config["context_length"] = args.context_length

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
