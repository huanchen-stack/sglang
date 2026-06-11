#!/usr/bin/env python3
"""Search for a long-running Eurus batch and measure TP4 e2e latency."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
CLIENT = THIS_DIR / "stream_batch_client.py"
ANALYZER = THIS_DIR / "analyze_e2e_latency.py"
DEFAULT_DATASET = (
    ROOT
    / ".rollout-profile"
    / "request-lifetime-cot"
    / "datasets"
    / "eurus-2-rl-qwen2.5-14b-instruct.jsonl"
)
DEFAULT_POLICY = (
    ROOT / ".rollout-impl-v0" / "decoding-throughput" / "frontier_precision_policy.json"
)
DEFAULT_OUTPUT = THIS_DIR / "results" / "qwen2.5-14b-eurus-tp4-cap16k"


def now() -> float:
    return time.perf_counter()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def slugify_label(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def default_served_model_name(model_path: str, suffix: str) -> str:
    base = model_path.rstrip("/").split("/")[-1]
    return f"{slugify_label(base)}-{suffix}"


def first_prompt_text(row: dict[str, Any]) -> str | None:
    for key in ("prompt", "query", "question", "problem", "instruction", "content", "value"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "").strip().lower()
            if role not in ("human", "user"):
                continue
            value = turn.get("value") or turn.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def extract_finish_reason(data: dict[str, Any]) -> Any:
    meta = data.get("meta_info") or {}
    for key in ("finish_reason", "finished_reason", "finish_reason_type"):
        if key in meta:
            return meta[key]
        if key in data:
            return data[key]
    return None


def finish_reason_is_length(value: Any, target_length: int) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "length":
            length = value.get("length")
            return length is None or int(length) >= target_length
    text = str(value)
    return "length" in text.lower()


def load_dataset_rows(path: Path, prompt_suffix: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = first_prompt_text(row)
            if not prompt:
                continue
            full_prompt = prompt + prompt_suffix
            extra_info = row.get("extra_info")
            source_dataset_index = None
            if isinstance(extra_info, dict) and extra_info.get("index") is not None:
                source_dataset_index = int(extra_info["index"])
            rows.append(
                {
                    "dataset_index": idx,
                    "source_dataset_index": source_dataset_index,
                    "prompt": full_prompt,
                    "row": row,
                    "prompt_words": len(full_prompt.split()),
                    "prompt_chars": len(full_prompt),
                }
            )
    if not rows:
        raise RuntimeError(f"no prompts found in {path}")
    return rows


def order_rows(
    rows: list[dict[str, Any]],
    *,
    ranking_mode: str,
    ranking_seed: int,
) -> list[dict[str, Any]]:
    if ranking_mode == "prompt_words_desc":
        return sorted(
            rows,
            key=lambda row: (
                int(row["prompt_words"]),
                int(row["prompt_chars"]),
                -int(row["dataset_index"]),
            ),
            reverse=True,
        )
    if ranking_mode == "random":
        ordered = list(rows)
        rng = random.Random(ranking_seed)
        rng.shuffle(ordered)
        return ordered
    raise ValueError(f"unsupported ranking_mode={ranking_mode}")


@dataclass
class ServerHandle:
    name: str
    base_url: str
    proc: subprocess.Popen
    log_path: Path
    log_handle: Any


async def wait_ready(base_url: str, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server at {base_url} exited early with rc={proc.returncode}")
            try:
                async with session.get(f"{normalize_base_url(base_url)}/health") as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2.0)
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


def build_bf16_server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.bf16_model_path,
        "--served-model-name",
        args.bf16_served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.bf16_port),
        "--tp-size",
        str(args.tp_size),
        "--dtype",
        "bfloat16",
        "--max-running-requests",
        str(args.batch_size),
        "--cuda-graph-bs",
        str(args.batch_size),
        "--cuda-graph-max-bs",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--attention-backend",
        "triton",
    ]
    if args.enable_torch_compile:
        cmd.extend(
            [
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                str(args.batch_size),
            ]
        )
    if args.bf16_mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(args.bf16_mem_fraction_static)])
    for extra in args.bf16_extra_server_arg:
        cmd.append(extra)
    return cmd


def build_dynamic_server_cmd(args: argparse.Namespace) -> list[str]:
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
        "--served-model-name",
        args.dynamic_served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.dynamic_port),
        "--tp-size",
        str(args.tp_size),
        "--dtype",
        "bfloat16",
        "--lora-paths",
        lora_arg,
        "--rollout-weight-colocation-int4-model-path",
        args.int4_model_path,
        "--rollout-weight-colocation-precision-policy-path",
        str(args.precision_policy_path),
        "--max-running-requests",
        str(args.batch_size),
        "--cuda-graph-bs",
        str(args.batch_size),
        "--cuda-graph-max-bs",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--attention-backend",
        "triton",
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--max-loras-per-batch",
        "1",
    ]
    if args.enable_torch_compile:
        cmd.extend(
            [
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                str(args.batch_size),
            ]
        )
    if args.dynamic_mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(args.dynamic_mem_fraction_static)])
    if args.int4_load_format:
        cmd.extend(["--rollout-weight-colocation-int4-load-format", args.int4_load_format])
    if args.int4_quantization:
        cmd.extend(
            ["--rollout-weight-colocation-int4-quantization", args.int4_quantization]
        )
    for extra in args.dynamic_extra_server_arg:
        cmd.append(extra)
    return cmd


async def launch_server(
    *,
    name: str,
    cmd: list[str],
    gpu_list: str,
    base_url: str,
    log_path: Path,
    ready_timeout_s: float,
    extra_env: dict[str, str] | None = None,
) -> ServerHandle:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_list
    env.setdefault("HF_HOME", "/data/huggingface")
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        await wait_ready(base_url, ready_timeout_s, proc)
    except Exception:
        terminate(proc)
        log_handle.close()
        raise
    return ServerHandle(
        name=name,
        base_url=base_url,
        proc=proc,
        log_path=log_path,
        log_handle=log_handle,
    )


async def close_server(handle: ServerHandle) -> None:
    terminate(handle.proc)
    handle.log_handle.close()


async def screen_one_server(
    *,
    base_url: str,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stream_interval: int,
    request_timeout_s: float,
    lora_path: str | None,
    seed_base: int,
) -> list[dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def run_one(item: dict[str, Any]) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "text": item["prompt"],
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_new_tokens": max_new_tokens,
                    "stream_interval": stream_interval,
                    "ignore_eos": False,
                    "sampling_seed": seed_base + int(item["dataset_index"]),
                },
                "stream": False,
                "return_logprob": False,
            }
            if lora_path:
                payload["lora_path"] = lora_path
            url = f"{normalize_base_url(base_url)}/generate"
            start = now()
            try:
                async with session.post(url, json=payload) as response:
                    body = await response.text()
                    if response.status != 200:
                        return {
                            "dataset_index": item["dataset_index"],
                            "success": False,
                            "error": f"HTTP {response.status}: {body[:400]}",
                            "latency_s": now() - start,
                        }
                    data = json.loads(body)
            except Exception as exc:
                return {
                    "dataset_index": item["dataset_index"],
                    "success": False,
                    "error": repr(exc),
                    "latency_s": now() - start,
                }
            meta = data.get("meta_info") or {}
            finish_reason = extract_finish_reason(data)
            completion_tokens = int(meta.get("completion_tokens", 0))
            return {
                "dataset_index": item["dataset_index"],
                "success": True,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "latency_s": now() - start,
            }

        return await asyncio.gather(*(run_one(item) for item in rows))


def chunk_with_padding(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    anchor_pool = list(rows[:batch_size]) if rows else []
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        if len(chunk) < batch_size:
            needed = batch_size - len(chunk)
            if not anchor_pool:
                raise RuntimeError("cannot pad screening chunk without anchor rows")
            chunk.extend(anchor_pool[:needed])
        chunks.append(chunk)
    return chunks


async def screen_stage(
    *,
    bf16: ServerHandle,
    dynamic: ServerHandle,
    batches: list[list[dict[str, Any]]],
    cap: int,
    args: argparse.Namespace,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    kept_batches: list[list[dict[str, Any]]] = []
    batch_summaries = []
    for batch_id, batch in enumerate(batches):
        bf16_task = screen_one_server(
            base_url=bf16.base_url,
            rows=batch,
            max_new_tokens=cap,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream_interval=args.stream_interval,
            request_timeout_s=args.request_timeout_s,
            lora_path=None,
            seed_base=args.seed_base,
        )
        dynamic_task = screen_one_server(
            base_url=dynamic.base_url,
            rows=batch,
            max_new_tokens=cap,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream_interval=args.stream_interval,
            request_timeout_s=args.request_timeout_s,
            lora_path=args.lora_name,
            seed_base=args.seed_base,
        )
        bf16_results, dynamic_results = await asyncio.gather(bf16_task, dynamic_task)
        bf16_cap_hit_indices = [
            int(item["dataset_index"])
            for item, result in zip(batch, bf16_results, strict=True)
            if result.get("success")
            and finish_reason_is_length(result.get("finish_reason"), cap)
            and int(result.get("completion_tokens", 0)) >= cap
        ]
        dynamic_cap_hit_indices = [
            int(item["dataset_index"])
            for item, result in zip(batch, dynamic_results, strict=True)
            if result.get("success")
            and finish_reason_is_length(result.get("finish_reason"), cap)
            and int(result.get("completion_tokens", 0)) >= cap
        ]
        keep_batch = (
            len(bf16_cap_hit_indices) >= args.min_cap_hit_requests_per_mode
            and len(dynamic_cap_hit_indices) >= args.min_cap_hit_requests_per_mode
        )
        if keep_batch:
            kept_batches.append(batch)
        batch_summaries.append(
            {
                "batch_id": batch_id,
                "cap": cap,
                "batch_size": len(batch),
                "bf16_cap_hit_count": len(bf16_cap_hit_indices),
                "dynamic_cap_hit_count": len(dynamic_cap_hit_indices),
                "bf16_cap_hit_indices": bf16_cap_hit_indices,
                "dynamic_cap_hit_indices": dynamic_cap_hit_indices,
                "kept": keep_batch,
            }
        )
    return kept_batches, {
        "cap": cap,
        "input_batch_count": len(batches),
        "kept_batch_count": len(kept_batches),
        "batches": batch_summaries,
    }


def write_sample(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row["row"])
            payload["dataset_index"] = int(row["dataset_index"])
            payload["prompt"] = row["prompt"]
            if row.get("sample_copy_index") is not None:
                payload["sample_copy_index"] = int(row["sample_copy_index"])
            if row.get("sample_position") is not None:
                payload["sample_position"] = int(row["sample_position"])
            if row.get("sampling_seed") is not None:
                payload["sampling_seed"] = int(row["sampling_seed"])
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def expand_rows_to_batch(
    rows: list[dict[str, Any]], batch_size: int, seed_base: int
) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("cannot expand an empty verified survivor set")
    expanded: list[dict[str, Any]] = []
    for sample_position in range(batch_size):
        base = rows[sample_position % len(rows)]
        item = dict(base)
        item["sample_copy_index"] = sample_position // len(rows)
        item["sample_position"] = sample_position
        item["sampling_seed"] = seed_base + sample_position
        expanded.append(item)
    return expanded


def persist_search_summary(path: Path, search_summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(search_summary, indent=2) + "\n", encoding="utf-8")


def run_client(
    *,
    args: argparse.Namespace,
    base_url: str,
    dataset_path: Path,
    output_dir: Path,
    lora_path: str | None,
) -> None:
    cmd = [
        args.python,
        str(CLIENT),
        "--base-url",
        base_url,
        "--dataset-path",
        str(dataset_path),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--stream-interval",
        str(args.stream_interval),
        "--stream-idle-timeout-s",
        str(args.stream_idle_timeout_s),
        "--ready-timeout-s",
        str(args.ready_timeout_s),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--seed-base",
        str(args.seed_base),
        "--output-dir",
        str(output_dir),
    ]
    if getattr(args, "ignore_eos", False):
        cmd.append("--ignore-eos")
    if lora_path:
        cmd.extend(["--lora-path", lora_path])
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_analyzer(run_dir: Path, mode_label: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ANALYZER),
        "--run-dir",
        str(run_dir),
        "--mode-label",
        mode_label,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads((run_dir / "lifecycle.json").read_text(encoding="utf-8"))


def count_requests_hit_cap(run_dir: Path, cap: int) -> int:
    payload = json.loads((run_dir / "requests.json").read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    return sum(
        1
        for req in requests
        if (
            req.get("success")
            and int(req.get("output_len", 0)) >= cap
            and finish_reason_is_length(req.get("finish_reason"), cap)
        )
    )


def mode_has_enough_cap_hits(run_dir: Path, cap: int, min_count: int) -> bool:
    return count_requests_hit_cap(run_dir, cap) >= min_count


def collect_cap_hit_request_ids(run_dir: Path, cap: int) -> list[int]:
    payload = json.loads((run_dir / "requests.json").read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    return [
        int(req["request_id"])
        for req in requests
        if (
            req.get("success")
            and int(req.get("output_len", 0)) >= cap
            and finish_reason_is_length(req.get("finish_reason"), cap)
        )
    ]


def unique_rows_from_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in batch:
        idx = int(row["dataset_index"])
        if idx in seen:
            continue
        seen.add(idx)
        unique_rows.append(row)
    return unique_rows


def summarize_selection(batch: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in batch:
        idx = int(row["dataset_index"])
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def batch_all_request_ids(batch: list[dict[str, Any]]) -> list[int]:
    return [int(row["dataset_index"]) for row in batch]


def choose_sparse_tail_batch(
    kept_batches: list[list[dict[str, Any]]],
    final_stage_summary: dict[str, Any],
    max_cap_hits_per_mode: int | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    kept_batch_summaries = [item for item in final_stage_summary.get("batches", []) if item.get("kept")]
    if len(kept_batch_summaries) != len(kept_batches):
        raise RuntimeError("kept batch summaries are inconsistent with kept batches")

    candidates: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for batch, summary in zip(kept_batches, kept_batch_summaries, strict=True):
        bf16_hits = int(summary["bf16_cap_hit_count"])
        dynamic_hits = int(summary["dynamic_cap_hit_count"])
        if max_cap_hits_per_mode is not None and (
            bf16_hits > max_cap_hits_per_mode or dynamic_hits > max_cap_hits_per_mode
        ):
            continue
        candidates.append((batch, summary))

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: (
            max(int(item[1]["bf16_cap_hit_count"]), int(item[1]["dynamic_cap_hit_count"])),
            int(item[1]["bf16_cap_hit_count"]) + int(item[1]["dynamic_cap_hit_count"]),
            int(item[1]["batch_id"]),
        )
    )
    return candidates[0]


def request_hit_count_summary(run_dir: Path, cap: int) -> dict[str, Any]:
    payload = json.loads((run_dir / "requests.json").read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    cap_hit_request_ids = []
    for req in requests:
        if (
            req.get("success")
            and int(req.get("output_len", 0)) >= cap
            and finish_reason_is_length(req.get("finish_reason"), cap)
        ):
            cap_hit_request_ids.append(int(req["request_id"]))
    return {
        "cap_hit_count": len(cap_hit_request_ids),
        "cap_hit_request_ids": cap_hit_request_ids,
        "total_requests": len(requests),
    }


def summarize(
    *,
    out_dir: Path,
    selected_indices: list[int],
    selection_metadata: dict[str, Any],
    bf16_lifecycle: dict,
    dynamic_lifecycle: dict,
    search_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    def ratio(num: float, den: float) -> float | None:
        if den == 0:
            return None
        return num / den

    bf16_cap_hits = request_hit_count_summary(out_dir / "bf16", args.max_new_tokens)
    dynamic_cap_hits = request_hit_count_summary(out_dir / "dynamic", args.max_new_tokens)
    summary = {
        "experiment_label": args.experiment_label,
        "dataset_path": str(args.dataset_path),
        "bf16_model_path": args.bf16_model_path,
        "int4_model_path": args.int4_model_path,
        "selected_indices": selected_indices,
        "selection": selection_metadata,
        "cap_hit_requirement_per_mode": args.min_cap_hit_requests_per_mode,
        "bf16": {
            "drain_s": bf16_lifecycle["total_drain_s"],
            "request_latency_p50_s": bf16_lifecycle["request_latency_p50_s"],
            "request_latency_p99_s": bf16_lifecycle["request_latency_p99_s"],
            "ttft_p50_s": bf16_lifecycle["ttft_p50_s"],
            "decode_output_tok_s": bf16_lifecycle["decode_output_tok_s"],
            "finish_reasons": bf16_lifecycle["finish_reasons"],
            "cap_hit_summary": bf16_cap_hits,
        },
        "dynamic": {
            "drain_s": dynamic_lifecycle["total_drain_s"],
            "request_latency_p50_s": dynamic_lifecycle["request_latency_p50_s"],
            "request_latency_p99_s": dynamic_lifecycle["request_latency_p99_s"],
            "ttft_p50_s": dynamic_lifecycle["ttft_p50_s"],
            "decode_output_tok_s": dynamic_lifecycle["decode_output_tok_s"],
            "finish_reasons": dynamic_lifecycle["finish_reasons"],
            "cap_hit_summary": dynamic_cap_hits,
        },
        "speedups": {
            "drain": ratio(bf16_lifecycle["total_drain_s"], dynamic_lifecycle["total_drain_s"]),
            "request_latency_p50": ratio(
                bf16_lifecycle["request_latency_p50_s"],
                dynamic_lifecycle["request_latency_p50_s"],
            ),
            "request_latency_p99": ratio(
                bf16_lifecycle["request_latency_p99_s"],
                dynamic_lifecycle["request_latency_p99_s"],
            ),
            "ttft_p50": ratio(
                bf16_lifecycle["ttft_p50_s"],
                dynamic_lifecycle["ttft_p50_s"],
            ),
            "decode_output_tok_s": ratio(
                dynamic_lifecycle["decode_output_tok_s"],
                bf16_lifecycle["decode_output_tok_s"],
            ),
        },
        "search_summary_path": str(out_dir / "search_summary.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "search_summary.json").write_text(
        json.dumps(search_summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def make_comparison_plot(
    out_path: Path,
    bf16_dir: Path,
    dynamic_dir: Path,
    bf16_lifecycle: dict,
    dynamic_lifecycle: dict,
    experiment_label: str,
) -> None:
    import matplotlib.pyplot as plt

    def load_sampled(run_dir: Path) -> tuple[list[float], list[int], list[float]]:
        ts, live, mean = [], [], []
        with (run_dir / "timeline_sampled.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts.append(float(row["time_since_decode_start_s"]))
                live.append(int(float(row["live_requests"])))
                mean.append(float(row["mean_decoded_len"]))
        return ts, live, mean

    bf16_t, bf16_live, bf16_mean = load_sampled(bf16_dir)
    dyn_t, dyn_live, dyn_mean = load_sampled(dynamic_dir)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    axes[0].plot(bf16_t, bf16_live, label="BF16", color="#4C72B0", linewidth=1.8)
    axes[0].plot(dyn_t, dyn_live, label="Dynamic", color="#C44E52", linewidth=1.8)
    axes[0].set_xlabel("Decode time (s)")
    axes[0].set_ylabel("Live requests")
    axes[0].grid(True, alpha=0.25, linewidth=0.6)
    axes[0].legend()

    axes[1].plot(bf16_t, bf16_mean, label="BF16", color="#4C72B0", linewidth=1.8)
    axes[1].plot(dyn_t, dyn_mean, label="Dynamic", color="#C44E52", linewidth=1.8)
    axes[1].set_xlabel("Decode time (s)")
    axes[1].set_ylabel("Mean decoded length")
    axes[1].grid(True, alpha=0.25, linewidth=0.6)

    labels = ["Drain", "Req p50", "Req p99", "TTFT p50"]
    bf16_vals = [
        bf16_lifecycle["total_drain_s"],
        bf16_lifecycle["request_latency_p50_s"],
        bf16_lifecycle["request_latency_p99_s"],
        bf16_lifecycle["ttft_p50_s"],
    ]
    dyn_vals = [
        dynamic_lifecycle["total_drain_s"],
        dynamic_lifecycle["request_latency_p50_s"],
        dynamic_lifecycle["request_latency_p99_s"],
        dynamic_lifecycle["ttft_p50_s"],
    ]
    x = np.arange(len(labels))
    width = 0.38
    axes[2].bar(x - width / 2, bf16_vals, width, label="BF16", color="#4C72B0")
    axes[2].bar(x + width / 2, dyn_vals, width, label="Dynamic", color="#C44E52")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25)
    axes[2].set_ylabel("Seconds")
    axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)

    drain_speedup = (
        bf16_lifecycle["total_drain_s"] / dynamic_lifecycle["total_drain_s"]
        if dynamic_lifecycle["total_drain_s"] > 0
        else float("nan")
    )
    fig.suptitle(
        f"{experiment_label}: dynamic vs BF16 (drain speedup {drain_speedup:.3f}x)",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--ranking-mode",
        choices=["prompt_words_desc", "random"],
        default="prompt_words_desc",
    )
    parser.add_argument("--ranking-seed", type=int, default=20260609)
    parser.add_argument("--bf16-model-path", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--int4-model-path", default="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")
    parser.add_argument("--experiment-label", default=None)
    parser.add_argument("--bf16-served-model-name", default=None)
    parser.add_argument("--dynamic-served-model-name", default=None)
    parser.add_argument(
        "--lora-path",
        default="/data/huanchen/._delete/adapters/qwen2.5-14b_rank16_zero_bf16",
    )
    parser.add_argument("--lora-name", default="default")
    parser.add_argument("--lora-startup-arg", default=None)
    parser.add_argument("--precision-policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--bf16-gpus", default="0,1,2,3")
    parser.add_argument("--dynamic-gpus", default="4,5,6,7")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--bf16-port", type=int, default=31600)
    parser.add_argument("--dynamic-port", type=int, default=31610)
    parser.add_argument("--context-length", type=int, default=20000)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stream-interval", type=int, default=64)
    parser.add_argument("--stream-idle-timeout-s", type=float, default=300.0)
    parser.add_argument("--request-timeout-s", type=float, default=21600.0)
    parser.add_argument("--ready-timeout-s", type=float, default=2400.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--screen-caps", type=int, nargs="+", default=[2048, 4096, 8192, 16000])
    parser.add_argument("--min-cap-hit-requests-per-mode", type=int, default=1)
    parser.add_argument("--max-final-cap-hit-requests-per-mode", type=int, default=None)
    parser.add_argument("--min-verified-unique", type=int, default=32)
    parser.add_argument("--initial-pool", type=int, default=512)
    parser.add_argument("--pool-step", type=int, default=256)
    parser.add_argument("--max-pool", type=int, default=2048)
    parser.add_argument("--bf16-mem-fraction-static", type=float, default=None)
    parser.add_argument("--dynamic-mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--enable-torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--int4-load-format", default=None)
    parser.add_argument("--int4-quantization", default=None)
    parser.add_argument("--python", default="/data/huanchen/miniforge3/envs/sglang/bin/python")
    parser.add_argument("--bf16-extra-server-arg", action="append", default=[])
    parser.add_argument("--dynamic-extra-server-arg", action="append", default=[])
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    if args.clean_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = order_rows(
        load_dataset_rows(args.dataset_path, args.prompt_suffix),
        ranking_mode=args.ranking_mode,
        ranking_seed=args.ranking_seed,
    )
    bf16_dir = args.output_dir / "bf16"
    dynamic_dir = args.output_dir / "dynamic"
    bf16_dir.mkdir(parents=True, exist_ok=True)
    dynamic_dir.mkdir(parents=True, exist_ok=True)

    bf16_cmd = build_bf16_server_cmd(args)
    dynamic_cmd = build_dynamic_server_cmd(args)
    (bf16_dir / "commands.json").write_text(
        json.dumps({"server_cmd": bf16_cmd}, indent=2) + "\n", encoding="utf-8"
    )
    (dynamic_dir / "commands.json").write_text(
        json.dumps({"server_cmd": dynamic_cmd}, indent=2) + "\n", encoding="utf-8"
    )

    bf16_server = await launch_server(
        name="bf16",
        cmd=bf16_cmd,
        gpu_list=args.bf16_gpus,
        base_url=f"http://{args.host}:{args.bf16_port}",
        log_path=bf16_dir / "server.log",
        ready_timeout_s=args.ready_timeout_s,
    )
    dynamic_server = None
    try:
        dynamic_server = await launch_server(
            name="dynamic",
            cmd=dynamic_cmd,
            gpu_list=args.dynamic_gpus,
            base_url=f"http://{args.host}:{args.dynamic_port}",
            log_path=dynamic_dir / "server.log",
            ready_timeout_s=args.ready_timeout_s,
            extra_env={"SGLANG_ROLLOUT_WEIGHT_COLOCATION_TRACE": "1"},
        )

        pool = min(args.initial_pool, len(rows), args.max_pool)
        search_summary: dict[str, Any] = {
            "dataset_path": str(args.dataset_path),
            "ranking": args.ranking_mode,
            "ranking_seed": args.ranking_seed,
            "screen_caps": args.screen_caps,
            "min_cap_hit_requests_per_mode": args.min_cap_hit_requests_per_mode,
            "attempts": [],
        }
        search_summary_path = args.output_dir / "search_summary.json"
        persist_search_summary(search_summary_path, search_summary)
        selected_rows: list[dict[str, Any]] | None = None
        verified_unique_rows: list[dict[str, Any]] | None = None
        selection_metadata: dict[str, Any] | None = None
        best_kept_batch_count = 0

        while pool <= min(len(rows), args.max_pool):
            candidate_rows = rows[:pool]
            attempt: dict[str, Any] = {"pool": pool, "stages": []}
            search_summary["attempts"].append(attempt)
            persist_search_summary(search_summary_path, search_summary)
            print(
                f"[search] pool={pool} start, min_cap_hits_per_mode={args.min_cap_hit_requests_per_mode}",
                flush=True,
            )
            current_batches = chunk_with_padding(candidate_rows, args.batch_size)
            attempt["input_batch_count"] = len(current_batches)
            final_stage_summary: dict[str, Any] | None = None
            for cap in args.screen_caps:
                current_batches, stage_summary = await screen_stage(
                    bf16=bf16_server,
                    dynamic=dynamic_server,
                    batches=current_batches,
                    cap=cap,
                    args=args,
                )
                attempt["stages"].append(stage_summary)
                final_stage_summary = stage_summary
                attempt["kept_batch_count_after_cap"] = len(current_batches)
                if current_batches:
                    attempt["example_kept_batch_indices_after_cap"] = batch_all_request_ids(
                        current_batches[0]
                    )
                if len(current_batches) > best_kept_batch_count:
                    best_kept_batch_count = len(current_batches)
                    search_summary["best_kept_batch_count"] = best_kept_batch_count
                print(
                    f"[search] pool={pool} cap={cap} kept_batches={len(current_batches)}",
                    flush=True,
                )
                persist_search_summary(search_summary_path, search_summary)
                if not current_batches:
                    break
            attempt["final_kept_batch_count"] = len(current_batches)
            if current_batches:
                attempt["final_selected_batch_indices"] = batch_all_request_ids(current_batches[0])
            search_summary["best_kept_batch_count"] = max(
                best_kept_batch_count, len(current_batches)
            )
            persist_search_summary(search_summary_path, search_summary)
            sparse_batch, sparse_summary = choose_sparse_tail_batch(
                current_batches,
                final_stage_summary or {"batches": []},
                args.max_final_cap_hit_requests_per_mode,
            )
            if sparse_batch is not None and sparse_summary is not None:
                selected_rows = list(sparse_batch)
                verified_unique_rows = unique_rows_from_batch(selected_rows)
                selection_metadata = {
                    "mode": "screened_batch",
                    "verified_batch_count": len(current_batches),
                    "selected_unique_count": len(verified_unique_rows),
                    "sampled_batch_size": args.batch_size,
                    "selection_counts_by_dataset_index": summarize_selection(selected_rows),
                    "final_cap_hit_counts": {
                        "bf16": int(sparse_summary["bf16_cap_hit_count"]),
                        "dynamic": int(sparse_summary["dynamic_cap_hit_count"]),
                    },
                }
                break
            pool += args.pool_step

        if selected_rows is None or verified_unique_rows is None or selection_metadata is None:
            raise RuntimeError(
                "unable to find a valid batch meeting the requested cap-hit criterion "
                f"(need >= {args.min_cap_hit_requests_per_mode} cap-hit requests per mode at caps "
                f"{args.screen_caps}; best_kept_batches={search_summary.get('best_kept_batch_count', 0)})"
            )

        write_sample(bf16_dir / "verified_unique_requests.jsonl", verified_unique_rows)
        write_sample(dynamic_dir / "verified_unique_requests.jsonl", verified_unique_rows)
        write_sample(bf16_dir / "selected_requests.jsonl", selected_rows)
        write_sample(dynamic_dir / "selected_requests.jsonl", selected_rows)

        await asyncio.gather(
            asyncio.to_thread(
                run_client,
                args=args,
                base_url=bf16_server.base_url,
                dataset_path=bf16_dir / "selected_requests.jsonl",
                output_dir=bf16_dir,
                lora_path=None,
            ),
            asyncio.to_thread(
                run_client,
                args=args,
                base_url=dynamic_server.base_url,
                dataset_path=dynamic_dir / "selected_requests.jsonl",
                output_dir=dynamic_dir,
                lora_path=args.lora_name,
            ),
        )

        if not mode_has_enough_cap_hits(
            bf16_dir, args.max_new_tokens, args.min_cap_hit_requests_per_mode
        ):
            raise RuntimeError(
                "final BF16 run did not meet the batch-level cap-hit requirement "
                f"(need >= {args.min_cap_hit_requests_per_mode})"
            )
        if not mode_has_enough_cap_hits(
            dynamic_dir, args.max_new_tokens, args.min_cap_hit_requests_per_mode
        ):
            raise RuntimeError(
                "final dynamic run did not meet the batch-level cap-hit requirement "
                f"(need >= {args.min_cap_hit_requests_per_mode})"
            )

        bf16_lifecycle = run_analyzer(bf16_dir, "BF16")
        dynamic_lifecycle = run_analyzer(dynamic_dir, "Dynamic")
        make_comparison_plot(
            args.output_dir / "comparison.png",
            bf16_dir,
            dynamic_dir,
            bf16_lifecycle,
            dynamic_lifecycle,
            args.experiment_label,
        )
        summary = summarize(
            out_dir=args.output_dir,
            selected_indices=batch_all_request_ids(selected_rows),
            selection_metadata=selection_metadata,
            bf16_lifecycle=bf16_lifecycle,
            dynamic_lifecycle=dynamic_lifecycle,
            search_summary=search_summary,
            args=args,
        )
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        await close_server(bf16_server)
        if dynamic_server is not None:
            await close_server(dynamic_server)


def main() -> int:
    args = parse_args()
    if args.experiment_label is None:
        args.experiment_label = args.output_dir.name
    if args.bf16_served_model_name is None:
        args.bf16_served_model_name = default_served_model_name(
            args.bf16_model_path, "bf16-e2e"
        )
    if args.dynamic_served_model_name is None:
        args.dynamic_served_model_name = default_served_model_name(
            args.bf16_model_path, "dynamic-e2e"
        )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
