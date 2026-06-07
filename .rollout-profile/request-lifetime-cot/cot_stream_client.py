#!/usr/bin/env python3
"""Stream a CoT batch and record per-request decoded length over time."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class RequestSummary:
    request_id: int
    success: bool
    prompt: str
    start_s: float
    end_s: float | None = None
    first_token_s: float | None = None
    last_token_s: float | None = None
    output_len: int = 0
    finish_reason: Any = None
    error: str | None = None


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def first_prompt_text(row: dict[str, Any]) -> str | None:
    for key in ("prompt", "query", "question", "problem", "instruction", "content", "value"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("conversations", "conversation", "messages"):
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", item.get("from", ""))).lower()
                if role in {"assistant", "gpt"}:
                    continue
                text = item.get("content", item.get("value"))
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return None


def load_prompts(dataset_path: Path, count: int, suffix: str) -> list[str]:
    prompts: list[str] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = first_prompt_text(json.loads(line))
            if text:
                prompts.append(text + suffix)
    if not prompts:
        raise RuntimeError(f"no prompts found in {dataset_path}")
    return [prompts[i % len(prompts)] for i in range(count)]


async def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    url = f"{normalize_base_url(base_url)}/health"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        while time.perf_counter() < deadline:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"server did not become ready at {url}")


def parse_generate_chunk(raw: bytes) -> dict[str, Any] | None:
    line = raw.strip()
    if not line:
        return None
    text = line.decode("utf-8")
    if text.startswith("data: "):
        text = text[len("data: ") :]
    if text == "[DONE]":
        return None
    return json.loads(text)


def extract_finish_reason(data: dict[str, Any]) -> Any:
    meta = data.get("meta_info") or {}
    for key in ("finish_reason", "finished_reason", "finish_reason_type"):
        if key in meta:
            return meta[key]
        if key in data:
            return data[key]
    return None


async def event_writer(path: Path, queue: asyncio.Queue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "timestamp_s", "decoded_tokens"])
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            writer.writerow(item)
            queue.task_done()


async def run_one_request(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    request_id: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stream_interval: int,
    stream_idle_timeout_s: float,
    queue: asyncio.Queue,
) -> RequestSummary:
    url = f"{normalize_base_url(base_url)}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "stream_interval": stream_interval,
            "ignore_eos": False,
        },
        "stream": True,
        "return_logprob": False,
    }
    summary = RequestSummary(
        request_id=request_id,
        success=False,
        prompt=prompt,
        start_s=time.perf_counter(),
    )
    last_seen = 0
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                summary.error = f"HTTP {response.status}: {await response.text()}"
                return summary
            while True:
                try:
                    raw = await asyncio.wait_for(
                        response.content.readline(), timeout=stream_idle_timeout_s
                    )
                except TimeoutError:
                    summary.error = (
                        f"stream idle for {stream_idle_timeout_s:.1f}s "
                        f"after {last_seen} completion tokens"
                    )
                    break
                if not raw:
                    break
                data = parse_generate_chunk(raw)
                if data is None:
                    continue
                if "error" in data:
                    summary.error = str(data["error"])
                    continue
                meta = data.get("meta_info") or {}
                completion_tokens = int(meta.get("completion_tokens", last_seen))
                finish_reason = extract_finish_reason(data)
                if finish_reason is not None:
                    summary.finish_reason = finish_reason
                if completion_tokens <= last_seen:
                    continue
                now = time.perf_counter()
                if summary.first_token_s is None:
                    summary.first_token_s = now
                summary.last_token_s = now
                summary.output_len = completion_tokens
                await queue.put((request_id, f"{now:.9f}", completion_tokens))
                last_seen = completion_tokens
        summary.end_s = time.perf_counter()
        summary.success = summary.first_token_s is not None and summary.output_len > 0
        if summary.finish_reason is None:
            summary.finish_reason = "stream_end" if summary.error is None else "stream_idle_timeout"
    except Exception:
        summary.end_s = time.perf_counter()
        summary.error = traceback.format_exc()
    return summary


async def run_batch(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.dataset_path, args.batch_size, args.prompt_suffix)
    await wait_ready(args.base_url, args.ready_timeout_s)
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.event_queue_size)
    writer_task = asyncio.create_task(event_writer(args.output_dir / "token_events.csv", queue))
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    start_s = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, read_bufsize=10 * 1024**2
    ) as session:
        tasks = [
            run_one_request(
                session=session,
                base_url=args.base_url,
                request_id=i,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stream_interval=args.stream_interval,
                stream_idle_timeout_s=args.stream_idle_timeout_s,
                queue=queue,
            )
            for i, prompt in enumerate(prompts)
        ]
        summaries = await asyncio.gather(*tasks)
    end_s = time.perf_counter()
    await queue.put(None)
    await queue.join()
    await writer_task

    payload = {
        "metadata": {
            "base_url": args.base_url,
            "dataset_path": str(args.dataset_path),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stream_interval": args.stream_interval,
            "stream_idle_timeout_s": args.stream_idle_timeout_s,
            "ignore_eos": False,
            "wall_start_s": start_s,
            "wall_end_s": end_s,
            "wall_time_s": end_s - start_s,
        },
        "requests": [asdict(summary) for summary in summaries],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "requests.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    ok = sum(1 for summary in summaries if summary.success)
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "successful_requests": ok,
                "total_requests": len(summaries),
                "wall_time_s": end_s - start_s,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stream-interval", type=int, default=64)
    parser.add_argument("--stream-idle-timeout-s", type=float, default=300.0)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--request-timeout-s", type=float, default=21600.0)
    parser.add_argument("--event-queue-size", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_batch(parse_args()))


if __name__ == "__main__":
    main()
