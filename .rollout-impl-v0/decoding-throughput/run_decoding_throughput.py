#!/usr/bin/env python3
"""Measure rollout-colocated decode throughput against the old frontier policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
OLD_THROUGHPUT_DIR = ROOT / ".rollout-profile" / "qlora-decoding-throughput"
OLD_FRONTIER = OLD_THROUGHPUT_DIR / "frontier_decoding_throughput.json"
OLD_CLIENT = OLD_THROUGHPUT_DIR / "decoding_client.py"
DEFAULT_POLICY = THIS_DIR / "frontier_precision_policy.json"
DEFAULT_OUT_DIR = THIS_DIR / "results" / "qwen2.5-14b-tp1-colocated"


def now() -> float:
    return time.perf_counter()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def precision_from_frontier_choice(choice: str) -> str:
    if choice == "bf16":
        return "bf16"
    if choice == "qlora":
        return "int4+torch2s"
    raise ValueError(f"Unsupported frontier precision choice: {choice}")


def load_frontier_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows = list(data.get("rows", []))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return sorted(rows, key=lambda row: int(row["batch_size"]))


def build_policy_from_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch_sizes = {}
    for row in rows:
        batch_sizes[str(int(row["batch_size"]))] = {
            "qkv": precision_from_frontier_choice(row["select_qkv"]),
            "o": precision_from_frontier_choice(row["select_o"]),
            "up": precision_from_frontier_choice(row["select_gate_up"]),
            "down": precision_from_frontier_choice(row["select_down"]),
        }
    return {
        "metadata": {
            "source": str(OLD_FRONTIER.relative_to(ROOT)),
            "kernel_source": ".rollout-profile/qlora-kernel",
            "projection_mapping": {
                "gate_up": "up",
                "qlora": "int4+torch2s",
                "bf16": "bf16",
            },
            "note": (
                "Generated from the old kernel-guided frontier, then measured "
                "with BF16 and INT4 weights colocated by rollout/impl_v0."
            ),
        },
        "batch_sizes": batch_sizes,
    }


def sample_vram() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        return [{"error": repr(exc)}]

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


def wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = now() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=10.0) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        if now() >= deadline:
            raise TimeoutError(f"server did not become ready at {health_url}")
        time.sleep(2.0)


def build_server_cmd(args: argparse.Namespace, batch_size: int) -> list[str]:
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
        str(batch_size),
        "--cuda-graph-max-bs",
        str(batch_size),
        "--tp-size",
        str(args.tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(batch_size),
        "--attention-backend",
        "triton",
        "--disable-radix-cache",
        "--enable-torch-compile",
        "--torch-compile-max-bs",
        str(batch_size),
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--max-loras-per-batch",
        "1",
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


def run_client(args: argparse.Namespace, batch_size: int, output_path: Path) -> None:
    cmd = [
        args.python,
        str(OLD_CLIENT),
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--scheme",
        "rollout_colocated_frontier",
        "--batch-size",
        str(batch_size),
        "--decode-tokens",
        str(args.decode_tokens),
        "--prompt",
        args.prompt,
        "--warmup-batches",
        str(args.warmup_batches),
        "--ready-timeout-s",
        "60",
        "--lora-path",
        args.lora_name,
        "--extra-request-body",
        json.dumps(args.extra_request_body),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def expected_path_for_precision(precision: str) -> str:
    return "bf16_decode" if precision == "bf16" else "int4_decode"


def log_has_path(log_text: str, path: str, projection: str) -> bool:
    needle_a = f"Rollout weight colocation path={path}"
    needle_b = f"projection={projection}"
    return any(
        needle_a in line and needle_b in line for line in log_text.splitlines()
    )


def parse_log_checks(log_path: Path, batch_size: int, policy: dict[str, str]) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    projection_paths = {}
    for projection, precision in policy.items():
        expected_path = expected_path_for_precision(precision)
        projection_paths[projection] = {
            "precision": precision,
            "expected_path": expected_path,
            "saw_expected_path": log_has_path(text, expected_path, projection),
        }

    return {
        "loaded_int4_shadow": "Loaded rollout weight colocation INT4 shadow model"
        in text,
        "attached_shadow_layers": "Rollout weight colocation attached INT4 shadow layers"
        in text,
        "saw_bf16_prefill_path": "path=bf16_prefill" in text,
        "captured_requested_cuda_graph_bs": (
            f"Capture cuda graph bs [{batch_size}]" in text
            or f"Capture cuda graph bs {batch_size}" in text
        ),
        "used_cuda_graph_decode": "cuda graph: True" in text,
        "projection_paths": projection_paths,
    }


def summarize_batch_result(
    *,
    batch_size: int,
    frontier_row: dict[str, Any],
    policy: dict[str, str],
    client_path: Path,
    server_log: Path,
    server_cmd: list[str],
    vram: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    client_payload = load_json(client_path)
    client_summary = dict(client_payload.get("summary", {}))
    measured_tok_s = client_summary.get("full_decode_tok_s")
    if measured_tok_s is None:
        measured_tok_s = client_summary.get("decode_tok_s")
    frontier_tok_s = frontier_row.get("kernel_guided_frontier_tok_s")
    ratio = (
        measured_tok_s / frontier_tok_s
        if measured_tok_s is not None and frontier_tok_s
        else None
    )
    log_checks = parse_log_checks(server_log, batch_size, policy)
    projection_ok = all(
        item["saw_expected_path"] for item in log_checks["projection_paths"].values()
    )
    core_log_ok = all(
        bool(log_checks[key])
        for key in (
            "loaded_int4_shadow",
            "attached_shadow_layers",
            "saw_bf16_prefill_path",
            "captured_requested_cuda_graph_bs",
            "used_cuda_graph_decode",
        )
    )
    success = bool(client_summary.get("success")) and projection_ok and core_log_ok
    return {
        "batch_size": batch_size,
        "success": success,
        "policy": policy,
        "server_cmd": server_cmd,
        "client_json": str(client_path),
        "server_log": str(server_log),
        "elapsed_wall_s": elapsed_s,
        "vram": vram,
        "client_summary": client_summary,
        "measured_full_decode_tok_s": measured_tok_s,
        "measured_all_started_decode_tok_s": client_summary.get(
            "all_started_decode_tok_s"
        ),
        "old_frontier_tok_s": frontier_tok_s,
        "old_bf16_merged_tok_s": frontier_row.get("bf16_merged_tok_s"),
        "old_qlora_torch_twostream_tok_s": frontier_row.get(
            "qlora_torch_twostream_tok_s"
        ),
        "measured_vs_old_frontier_ratio": ratio,
        "log_checks": log_checks,
    }


def run_one_batch(
    args: argparse.Namespace,
    *,
    batch_size: int,
    policy: dict[str, str],
    frontier_row: dict[str, Any],
) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    server_log = args.out_dir / f"rollout_colocated_frontier.bs{batch_size}.server.log"
    client_json = args.out_dir / f"rollout_colocated_frontier.bs{batch_size}.json"
    base_url = f"http://{args.host}:{args.port}"
    server_cmd = build_server_cmd(args, batch_size)
    batch_start = now()
    vram: dict[str, Any] = {"before_launch": sample_vram()}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["SGLANG_ROLLOUT_WEIGHT_COLOCATION_TRACE"] = "1"
    if args.flip_twostream:
        env["SGLANG_ROLLOUT_FLIP_TWOSTREAM"] = "1"
    env.setdefault("SGLANG_LORA_TORCH_TWOSTREAM", "1")
    env.setdefault("SGLANG_LORA_TWOSTREAM_RESERVE_SMS", str(args.reserve_sms))
    env.setdefault("SGLANG_MARLIN_RESERVE_SMS", str(args.reserve_sms))
    cache_base = Path(
        os.environ.get("SGLANG_ROLLOUT_DECODE_CACHE_ROOT", "/data/huanchen/tmp/rollout-dt")
    )
    cache_root = cache_base / f"{args.out_dir.name}-p{os.getpid()}"
    tmp_dir = cache_root / "t"
    for path in (
        cache_root / "torchinductor",
        cache_root / "triton",
        cache_root / "cuda",
        cache_root / "xdg",
        tmp_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    env["CUDA_CACHE_PATH"] = str(cache_root / "cuda")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env["TMPDIR"] = str(tmp_dir)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    result_prefix = {
        "batch_size": batch_size,
        "policy": policy,
        "server_cmd": server_cmd,
        "client_json": str(client_json),
        "server_log": str(server_log),
    }

    if args.dry_run:
        print(" ".join(server_cmd), flush=True)
        return {
            **result_prefix,
            "success": True,
            "dry_run": True,
            "vram": vram,
        }

    with server_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            server_cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_ready(base_url, proc, args.ready_timeout_s)
            vram["after_ready"] = sample_vram()
            run_client(args, batch_size, client_json)
            vram["after_client"] = sample_vram()
        except Exception as exc:
            vram["after_failure"] = sample_vram()
            return {
                **result_prefix,
                "success": False,
                "error": repr(exc),
                "elapsed_wall_s": now() - batch_start,
                "vram": vram,
                "server_returncode": proc.poll(),
            }
        finally:
            terminate_process(proc)
            vram["after_shutdown"] = sample_vram()

    return summarize_batch_result(
        batch_size=batch_size,
        frontier_row=frontier_row,
        policy=policy,
        client_path=client_json,
        server_log=server_log,
        server_cmd=server_cmd,
        vram=vram,
        elapsed_s=now() - batch_start,
    )


def write_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    write_json(out_dir / "summary.json", payload)
    rows = payload.get("rows", [])
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "success",
                "measured_full_decode_tok_s",
                "measured_all_started_decode_tok_s",
                "old_frontier_tok_s",
                "measured_vs_old_frontier_ratio",
                "policy",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "batch_size": row.get("batch_size"),
                    "success": row.get("success"),
                    "measured_full_decode_tok_s": row.get(
                        "measured_full_decode_tok_s"
                    ),
                    "measured_all_started_decode_tok_s": row.get(
                        "measured_all_started_decode_tok_s"
                    ),
                    "old_frontier_tok_s": row.get("old_frontier_tok_s"),
                    "measured_vs_old_frontier_ratio": row.get(
                        "measured_vs_old_frontier_ratio"
                    ),
                    "policy": json.dumps(row.get("policy", {}), sort_keys=True),
                    "error": row.get("error"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-model-path", default="Qwen/Qwen2.5-14B-Instruct"
    )
    parser.add_argument(
        "--int4-model-path", default="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    )
    parser.add_argument(
        "--lora-path",
        default="/data/huanchen/._delete/adapters/qwen2.5-14b_rank16_zero_bf16",
    )
    parser.add_argument("--lora-name", default="default")
    parser.add_argument("--lora-startup-arg", default=None)
    parser.add_argument("--precision-policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--frontier-json", type=Path, default=OLD_FRONTIER)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--prompt", default="Briefly count upward:")
    parser.add_argument("--ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30120)
    parser.add_argument(
        "--python",
        default="/data/huanchen/miniforge3/envs/sglang/bin/python",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--reserve-sms", type=int, default=1)
    parser.add_argument(
        "--flip-twostream",
        action="store_true",
        help=(
            "Set SGLANG_ROLLOUT_FLIP_TWOSTREAM=1 so the INT4 base projection "
            "runs on the side stream and the LoRA patch runs on the current stream."
        ),
    )
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--int4-load-format", default=None)
    parser.add_argument("--int4-quantization", default=None)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--extra-request-body", type=json.loads, default={})
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--parallel-gpus",
        default=None,
        help="Comma-separated GPU ids. Runs one TP1 batch experiment per GPU.",
    )
    parser.add_argument("--write-policy-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_child_cmd(
    args: argparse.Namespace,
    *,
    batch_size: int,
    gpu: str,
    port: int,
    out_dir: Path,
) -> list[str]:
    cmd = [
        args.python,
        str(Path(__file__).resolve()),
        "--bf16-model-path",
        args.bf16_model_path,
        "--int4-model-path",
        args.int4_model_path,
        "--lora-path",
        args.lora_path,
        "--lora-name",
        args.lora_name,
        "--precision-policy-path",
        str(args.precision_policy_path),
        "--frontier-json",
        str(args.frontier_json),
        "--batch-sizes",
        str(batch_size),
        "--decode-tokens",
        str(args.decode_tokens),
        "--warmup-batches",
        str(args.warmup_batches),
        "--prompt",
        args.prompt,
        "--ready-timeout-s",
        str(args.ready_timeout_s),
        "--host",
        args.host,
        "--port",
        str(port),
        "--python",
        args.python,
        "--gpu",
        gpu,
        "--tp-size",
        str(args.tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--reserve-sms",
        str(args.reserve_sms),
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--extra-request-body",
        json.dumps(args.extra_request_body),
        "--out-dir",
        str(out_dir),
    ]
    if args.lora_startup_arg:
        cmd.extend(["--lora-startup-arg", args.lora_startup_arg])
    if args.int4_load_format:
        cmd.extend(["--int4-load-format", args.int4_load_format])
    if args.int4_quantization:
        cmd.extend(["--int4-quantization", args.int4_quantization])
    for extra_arg in args.extra_server_arg:
        cmd.append(f"--extra-server-arg={extra_arg}")
    if args.continue_after_failure:
        cmd.append("--continue-after-failure")
    if args.flip_twostream:
        cmd.append("--flip-twostream")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_parallel_batches(
    args: argparse.Namespace,
    *,
    batch_sizes: list[int],
    policy_payload: dict[str, Any],
    frontier_by_batch: dict[int, dict[str, Any]],
) -> int:
    gpus = [gpu.strip() for gpu in args.parallel_gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--parallel-gpus must contain at least one GPU id")
    if args.tp_size != 1:
        raise ValueError("Parallel decoding-throughput mode expects --tp-size 1")

    parent_out_dir = args.out_dir
    jobs_dir = parent_out_dir / "jobs"
    parent_out_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "metadata": {
            "description": (
                "Parallel TP1 decode throughput validation for rollout/impl_v0 "
                "with BF16 and INT4 weights colocated in VRAM."
            ),
            "frontier_json": str(args.frontier_json),
            "precision_policy_path": str(args.precision_policy_path),
            "old_client": str(OLD_CLIENT),
            "batch_sizes_requested": batch_sizes,
            "decode_tokens": args.decode_tokens,
            "warmup_batches": args.warmup_batches,
            "parallel_gpus": gpus,
            "tp_size": args.tp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "flip_twostream": args.flip_twostream,
        },
        "rows": [],
    }

    pending = deque(
        (idx, batch_size) for idx, batch_size in enumerate(batch_sizes)
    )
    active: list[dict[str, Any]] = []
    used_ports = set()

    def launch_next(gpu: str) -> None:
        idx, batch_size = pending.popleft()
        port = args.port + idx
        while port in used_ports:
            port += 1
        used_ports.add(port)
        out_dir = jobs_dir / f"bs{batch_size}"
        cmd = build_child_cmd(
            args,
            batch_size=batch_size,
            gpu=gpu,
            port=port,
            out_dir=out_dir,
        )
        log_path = parent_out_dir / f"worker.bs{batch_size}.gpu{gpu}.log"
        print(
            json.dumps(
                {
                    "event": "launch_worker",
                    "batch_size": batch_size,
                    "gpu": gpu,
                    "port": port,
                    "out_dir": str(out_dir),
                }
            ),
            flush=True,
        )
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            summary["rows"].append(
                {
                    "batch_size": batch_size,
                    "success": True,
                    "dry_run": True,
                    "gpu": gpu,
                    "port": port,
                    "policy": policy_payload["batch_sizes"][str(batch_size)],
                    "old_frontier_tok_s": frontier_by_batch[batch_size].get(
                        "kernel_guided_frontier_tok_s"
                    ),
                }
            )
            return
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append(
            {
                "proc": proc,
                "gpu": gpu,
                "batch_size": batch_size,
                "out_dir": out_dir,
                "log_handle": log_handle,
                "worker_log": log_path,
            }
        )

    if args.dry_run:
        while pending:
            gpu = gpus[len(summary["rows"]) % len(gpus)]
            launch_next(gpu)
        write_summary(parent_out_dir, summary)
        return 0

    for gpu in gpus:
        if pending:
            launch_next(gpu)

    while active:
        still_active = []
        for job in active:
            proc = job["proc"]
            rc = proc.poll()
            if rc is None:
                still_active.append(job)
                continue

            job["log_handle"].close()
            batch_size = job["batch_size"]
            child_summary_path = job["out_dir"] / "summary.json"
            if child_summary_path.exists():
                child_summary = load_json(child_summary_path)
                rows = child_summary.get("rows", [])
                if rows:
                    row = rows[0]
                else:
                    row = {"batch_size": batch_size, "success": False}
            else:
                row = {
                    "batch_size": batch_size,
                    "success": False,
                    "error": f"worker exited {rc} without summary.json",
                }
            row["worker_returncode"] = rc
            row["worker_gpu"] = job["gpu"]
            row["worker_log"] = str(job["worker_log"])
            summary["rows"].append(row)
            summary["rows"].sort(key=lambda item: int(item["batch_size"]))
            write_summary(parent_out_dir, summary)
            print(
                json.dumps(
                    {
                        "event": "finish_worker",
                        "batch_size": batch_size,
                        "gpu": job["gpu"],
                        "returncode": rc,
                        "success": row.get("success"),
                        "measured_full_decode_tok_s": row.get(
                            "measured_full_decode_tok_s"
                        ),
                        "old_frontier_tok_s": row.get("old_frontier_tok_s"),
                        "ratio": row.get("measured_vs_old_frontier_ratio"),
                        "error": row.get("error"),
                    }
                ),
                flush=True,
            )

            if pending:
                launch_next(job["gpu"])

        active = still_active
        if active:
            time.sleep(2.0)

    write_summary(parent_out_dir, summary)
    return 0 if all(row.get("success") for row in summary["rows"]) else 1


def main() -> int:
    args = parse_args()
    frontier_rows = load_frontier_rows(args.frontier_json)
    frontier_by_batch = {int(row["batch_size"]): row for row in frontier_rows}
    policy_payload = build_policy_from_frontier(frontier_rows)
    write_json(args.precision_policy_path, policy_payload)

    if args.write_policy_only:
        print(json.dumps({"precision_policy_path": str(args.precision_policy_path)}))
        return 0

    batch_sizes = (
        list(args.batch_sizes)
        if args.batch_sizes is not None
        else [int(row["batch_size"]) for row in frontier_rows]
    )
    if args.max_batch_size is not None:
        batch_sizes = [bs for bs in batch_sizes if bs <= args.max_batch_size]

    if args.parallel_gpus is not None:
        return run_parallel_batches(
            args,
            batch_sizes=batch_sizes,
            policy_payload=policy_payload,
            frontier_by_batch=frontier_by_batch,
        )

    summary: dict[str, Any] = {
        "metadata": {
            "description": (
                "Decode throughput validation for rollout/impl_v0 with BF16 and "
                "INT4 weights colocated in VRAM."
            ),
            "frontier_json": str(args.frontier_json),
            "precision_policy_path": str(args.precision_policy_path),
            "old_client": str(OLD_CLIENT),
            "batch_sizes_requested": batch_sizes,
            "decode_tokens": args.decode_tokens,
            "warmup_batches": args.warmup_batches,
            "gpu": args.gpu,
            "tp_size": args.tp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "flip_twostream": args.flip_twostream,
            "stop_policy": (
                "continue after failure"
                if args.continue_after_failure
                else "stop at first failed batch"
            ),
        },
        "rows": [],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for batch_size in batch_sizes:
        if batch_size not in frontier_by_batch:
            raise KeyError(f"No old frontier row for batch size {batch_size}")
        policy = policy_payload["batch_sizes"][str(batch_size)]
        print(
            json.dumps(
                {
                    "event": "start_batch",
                    "batch_size": batch_size,
                    "policy": policy,
                }
            ),
            flush=True,
        )
        row = run_one_batch(
            args,
            batch_size=batch_size,
            policy=policy,
            frontier_row=frontier_by_batch[batch_size],
        )
        summary["rows"].append(row)
        write_summary(args.out_dir, summary)
        print(
            json.dumps(
                {
                    "event": "finish_batch",
                    "batch_size": batch_size,
                    "success": row.get("success"),
                    "measured_full_decode_tok_s": row.get(
                        "measured_full_decode_tok_s"
                    ),
                    "old_frontier_tok_s": row.get("old_frontier_tok_s"),
                    "ratio": row.get("measured_vs_old_frontier_ratio"),
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )
        if not row.get("success") and not args.continue_after_failure:
            break

    write_summary(args.out_dir, summary)
    return 0 if all(row.get("success") for row in summary["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
