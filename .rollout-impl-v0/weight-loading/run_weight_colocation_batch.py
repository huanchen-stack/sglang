#!/usr/bin/env python3
"""Validate rollout weight colocation with simple batch serving.

The script launches an SGLang server with:

* primary BF16 model path as --model-path
* INT4 shadow path as --rollout-weight-colocation-int4-model-path
* one startup LoRA adapter

It then runs batch=8 and batch=16 requests through /generate, records VRAM
snapshots, and checks the server log for path markers:

* path=bf16_prefill
* path=int4_decode
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "results"


def now() -> float:
    return time.perf_counter()


def read_url(url: str, *, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def post_json(url: str, payload: dict[str, Any], *, timeout: float = 3600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = now() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while True:
        try:
            read_url(health_url, timeout=10.0)
            return
        except Exception:
            if now() >= deadline:
                raise TimeoutError(f"server did not become ready at {health_url}")
            time.sleep(2.0)


def sample_vram() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        return [{"error": str(exc)}]

    rows = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "gpu": int(parts[0]),
                "memory_used_mib": int(parts[1]),
                "memory_free_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
            }
        )
    return rows


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=60)


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    lora_arg = (
        args.lora_startup_arg
        if args.lora_startup_arg
        else f"{args.lora_name}={args.lora_path}"
    )
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.bf16_model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--lora-paths",
        lora_arg,
        "--rollout-weight-colocation-int4-model-path",
        args.int4_model_path,
        "--cuda-graph-bs",
        "8",
        "16",
        "--cuda-graph-max-bs",
        "16",
    ]
    if args.int4_load_format:
        cmd.extend(
            ["--rollout-weight-colocation-int4-load-format", args.int4_load_format]
        )
    if args.int4_quantization:
        cmd.extend(
            ["--rollout-weight-colocation-int4-quantization", args.int4_quantization]
        )
    cmd.extend(args.extra_server_arg)
    return cmd


def run_one_request(
    *,
    base_url: str,
    prompt: str,
    request_id: int,
    decode_tokens: int,
    lora_path: str,
) -> dict[str, Any]:
    payload = {
        "text": f"{prompt} [{request_id}]",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": decode_tokens,
            "ignore_eos": True,
        },
        "stream": False,
        "return_logprob": False,
        "lora_path": lora_path,
    }
    start = now()
    try:
        response = post_json(f"{base_url.rstrip('/')}/generate", payload)
        latency_s = now() - start
        text = response.get("text", "")
        return {
            "request_id": request_id,
            "success": True,
            "latency_s": latency_s,
            "text_len": len(text),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "request_id": request_id,
            "success": False,
            "error": f"HTTP {exc.code}: {body}",
        }
    except Exception as exc:
        return {"request_id": request_id, "success": False, "error": repr(exc)}


def run_batch(
    *,
    base_url: str,
    batch_size: int,
    prompt: str,
    decode_tokens: int,
    lora_path: str,
) -> dict[str, Any]:
    start = now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                run_one_request,
                base_url=base_url,
                prompt=prompt,
                request_id=i,
                decode_tokens=decode_tokens,
                lora_path=lora_path,
            )
            for i in range(batch_size)
        ]
        requests = [future.result() for future in futures]
    elapsed_s = now() - start
    ok = [item for item in requests if item.get("success")]
    return {
        "batch_size": batch_size,
        "success": len(ok) == batch_size,
        "completed": len(ok),
        "elapsed_s": elapsed_s,
        "requests": requests,
    }


def parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "loaded_int4_shadow": "Loaded rollout weight colocation INT4 shadow model"
        in text,
        "attached_shadow_layers": "Rollout weight colocation attached INT4 shadow layers"
        in text,
        "saw_bf16_prefill_path": "path=bf16_prefill" in text,
        "saw_int4_decode_path": "path=int4_decode" in text,
        "captured_bs8": "bs=8" in text or "bs=8 " in text,
        "captured_bs16": "bs=16" in text or "bs=16 " in text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-model-path", required=True)
    parser.add_argument("--int4-model-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--lora-name", default="default")
    parser.add_argument(
        "--lora-startup-arg",
        default=None,
        help="Optional raw --lora-paths value. Defaults to '<lora-name>=<lora-path>'.",
    )
    parser.add_argument("--int4-load-format", default=None)
    parser.add_argument("--int4-quantization", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--prompt", default="Count upward briefly.")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "weight_colocation_server.log"
    summary_path = args.out_dir / "weight_colocation_batch8_16.json"
    base_url = f"http://{args.host}:{args.port}"
    cmd = build_server_cmd(args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["SGLANG_ROLLOUT_WEIGHT_COLOCATION_TRACE"] = "1"
    env["SGLANG_LORA_TORCH_TWOSTREAM"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    summary: dict[str, Any] = {
        "server_cmd": cmd,
        "base_url": base_url,
        "vram": {"before_launch": sample_vram()},
        "batches": [],
        "log_path": str(log_path),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_ready(base_url, args.ready_timeout_s)
            summary["vram"]["after_ready"] = sample_vram()
            for batch_size in (8, 16):
                batch_summary = run_batch(
                    base_url=base_url,
                    batch_size=batch_size,
                    prompt=args.prompt,
                    decode_tokens=args.decode_tokens,
                    lora_path=args.lora_path,
                )
                batch_summary["vram_after_batch"] = sample_vram()
                summary["batches"].append(batch_summary)
        finally:
            terminate_process(proc)
            summary["server_returncode"] = proc.returncode
            summary["vram"]["after_shutdown"] = sample_vram()

    summary["log_checks"] = parse_log(log_path)
    summary["success"] = (
        all(batch.get("success") for batch in summary["batches"])
        and summary["log_checks"]["loaded_int4_shadow"]
        and summary["log_checks"]["attached_shadow_layers"]
        and summary["log_checks"]["saw_bf16_prefill_path"]
        and summary["log_checks"]["saw_int4_decode_path"]
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
