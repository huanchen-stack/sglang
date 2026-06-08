#!/usr/bin/env python3
"""Verify fixed-length decode throughput through the real dynamic-policy server."""

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


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PROFILE_CLIENT = REPO_ROOT / ".rollout-profile/qlora-decoding-throughput/decoding_client.py"


def load_json(path: Path) -> dict:
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


def cuda_graph_bs(config: dict, batch_size: int) -> list[int]:
    values = config.get("cuda_graph_bs_by_batch", {}).get(
        str(batch_size), config.get("cuda_graph_bs")
    )
    if values:
        return [int(v) for v in values]
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def build_server_cmd(config: dict, *, port: int, batch_size: int, python: str) -> list[str]:
    cmd = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config["model_path"],
        "--served-model-name",
        config.get("served_model_name", config.get("name", "dynamic-decode-verify")),
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
        str(config.get("context_length", 4096)),
        "--cuda-graph-bs",
        *[str(v) for v in cuda_graph_bs(config, batch_size) if v <= batch_size],
        *config.get("common_server_args", []),
    ]

    if config.get("lora_path"):
        cmd.extend(
            [
                "--enable-lora",
                "--lora-paths",
                f"{config.get('lora_name', 'default')}={config['lora_path']}",
                "--max-lora-rank",
                str(config.get("max_lora_rank", 16)),
                "--max-loras-per-batch",
                str(config.get("max_loras_per_batch", 1)),
            ]
        )
    if config.get("lora_backend"):
        cmd.extend(["--lora-backend", str(config["lora_backend"])])
    if config.get("rollout_precision_policy"):
        cmd.extend(["--rollout-precision-policy", str(config["rollout_precision_policy"])])
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
    if config.get("rollout_precision_force_torch_lora"):
        cmd.append("--rollout-precision-force-torch-lora")
    cmd.extend(config.get("extra_server_args", []))
    return cmd


def build_client_cmd(
    config: dict,
    *,
    port: int,
    batch_size: int,
    decode_tokens: int,
    warmup_batches: int,
    output: Path,
    python: str,
    prompt: str,
) -> list[str]:
    cmd = [
        python,
        str(PROFILE_CLIENT),
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--scheme",
        "dynamic_policy",
        "--batch-size",
        str(batch_size),
        "--decode-tokens",
        str(decode_tokens),
        "--prompt",
        prompt,
        "--warmup-batches",
        str(warmup_batches),
        "--ready-timeout-s",
        str(config.get("ready_timeout_s", 300)),
        "--output",
        str(output),
    ]
    if config.get("request_lora_path") is not None:
        cmd.extend(["--lora-path", str(config["request_lora_path"])])
    return cmd


def run_one(args: argparse.Namespace, config: dict, batch_size: int, job_index: int) -> None:
    run_dir = args.out_dir / f"bs{batch_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = args.port + job_index
    base_url = f"http://127.0.0.1:{port}"
    output = run_dir / "dynamic_policy.json"
    server_cmd = build_server_cmd(config, port=port, batch_size=batch_size, python=args.python)
    client_cmd = build_client_cmd(
        config,
        port=port,
        batch_size=batch_size,
        decode_tokens=args.decode_tokens,
        warmup_batches=args.warmup_batches,
        output=output,
        python=args.python,
        prompt=args.prompt,
    )
    metadata = {
        "purpose": "fixed-length decode verification through real dynamic-policy server",
        "config": config,
        "batch_size": batch_size,
        "decode_tokens": args.decode_tokens,
        "server_cmd": server_cmd,
        "client_cmd": client_cmd,
    }
    (run_dir / "commands.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[bs{batch_size}] SERVER: {' '.join(server_cmd)}", flush=True)
    print(f"[bs{batch_size}] CLIENT: {' '.join(client_cmd)}", flush=True)
    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/data/huggingface")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpus or config.get("gpus", "0"))
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})

    with (run_dir / "server.log").open("w", encoding="utf-8") as server_log:
        proc = subprocess.Popen(
            server_cmd,
            cwd=args.repo_root,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(base_url, float(config.get("ready_timeout_s", 300)), proc)
            subprocess.run(client_cmd, cwd=args.repo_root, env=env, check=True)
        finally:
            terminate(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / ".rollout-impl-codex/real-deliverable/configs/qwen2.5-14b-eurus-dynamic-tp4.json",
    )
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--prompt", default="Briefly count upward:")
    parser.add_argument("--out-dir", type=Path, default=THIS_DIR / "results")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--port", type=int, default=31600)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    batch_sizes = args.batch_size or [128, 256, 512]
    for idx, batch_size in enumerate(batch_sizes):
        run_one(args, config, int(batch_size), idx)


if __name__ == "__main__":
    main()
