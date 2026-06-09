#!/usr/bin/env python3
"""Validate rollout precision mixing with simple batch serving."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
ROLLOUT_DIR = THIS_DIR.parent
ROOT = ROLLOUT_DIR.parent
WEIGHT_LOADING = ROLLOUT_DIR / "weight-loading" / "run_weight_colocation_batch.py"
DEFAULT_POLICY = THIS_DIR / "policy_0_8_qkv_o_bf16_up_down_int4_torch2s.json"
OUT_DIR = THIS_DIR / "results"


def load_weight_harness():
    spec = importlib.util.spec_from_file_location("weight_harness", WEIGHT_LOADING)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {WEIGHT_LOADING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "--rollout-weight-colocation-precision-policy-path",
        str(args.precision_policy_path),
        "--cuda-graph-bs",
        *[str(bs) for bs in args.cuda_graph_bs],
        "--cuda-graph-max-bs",
        str(max(args.cuda_graph_bs)),
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


def has_path_for_projection(log_text: str, path: str, projection: str) -> bool:
    pattern = re.compile(
        rf"Rollout weight colocation path={re.escape(path)} .*projection={re.escape(projection)}"
    )
    return bool(pattern.search(log_text))


def parse_log(log_path: Path, cuda_graph_bs: list[int]) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    captured = f"Capture cuda graph bs {cuda_graph_bs}" in text
    checks: dict[str, Any] = {
        "loaded_int4_shadow": "Loaded rollout weight colocation INT4 shadow model"
        in text,
        "attached_shadow_layers": "Rollout weight colocation attached INT4 shadow layers"
        in text,
        "saw_bf16_prefill_path": "path=bf16_prefill" in text,
        "saw_bf16_decode_qkv": has_path_for_projection(text, "bf16_decode", "qkv"),
        "saw_bf16_decode_o": has_path_for_projection(text, "bf16_decode", "o"),
        "saw_int4_decode_up": has_path_for_projection(text, "int4_decode", "up"),
        "saw_int4_decode_down": has_path_for_projection(text, "int4_decode", "down"),
        "captured_cuda_graph_bs": captured,
    }
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-model-path", required=True)
    parser.add_argument("--int4-model-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--lora-name", default="default")
    parser.add_argument("--lora-startup-arg", default=None)
    parser.add_argument(
        "--precision-policy-path", type=Path, default=DEFAULT_POLICY
    )
    parser.add_argument("--cuda-graph-bs", nargs="+", type=int, default=[8])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--int4-load-format", default=None)
    parser.add_argument("--int4-quantization", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--prompt", default="Count upward briefly.")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    harness = load_weight_harness()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    server_log = args.out_dir / "precision_mixing_server.log"
    result_path = args.out_dir / "precision_mixing_batch.json"
    base_url = f"http://{args.host}:{args.port}"
    cmd = build_server_cmd(args)

    if args.dry_run:
        print(json.dumps({"server_cmd": cmd}, indent=2))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    env["SGLANG_ROLLOUT_WEIGHT_COLOCATION_TRACE"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    result: dict[str, Any] = {
        "server_cmd": cmd,
        "base_url": base_url,
        "precision_policy_path": str(args.precision_policy_path),
        "cuda_graph_bs": args.cuda_graph_bs,
        "vram": {"before_launch": harness.sample_vram()},
        "batches": [],
    }

    with server_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            harness.wait_ready(base_url, args.ready_timeout_s)
            result["vram"]["after_ready"] = harness.sample_vram()
            for batch_size in args.batch_sizes:
                batch = harness.run_batch(
                    base_url=base_url,
                    batch_size=batch_size,
                    prompt=args.prompt,
                    decode_tokens=args.decode_tokens,
                    lora_path=args.lora_path,
                )
                batch["vram_after_batch"] = harness.sample_vram()
                result["batches"].append(batch)
        finally:
            harness.terminate_process(proc)
            result["vram"]["after_shutdown"] = harness.sample_vram()
            result["server_returncode"] = proc.returncode

    result["log_path"] = str(server_log)
    result["log_checks"] = parse_log(server_log, args.cuda_graph_bs)
    result["success"] = all(batch["success"] for batch in result["batches"]) and all(
        bool(value) for value in result["log_checks"].values()
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
