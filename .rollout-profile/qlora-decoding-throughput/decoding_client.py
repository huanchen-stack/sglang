#!/usr/bin/env python3
"""Measure pure decode throughput from SGLang's streaming /generate endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class RequestTrace:
    request_id: int
    success: bool
    output_len: int = 0
    latency_s: float = 0.0
    first_token_s: float | None = None
    last_token_s: float | None = None
    token_timestamps_s: list[tuple[int, float]] = field(default_factory=list)
    error: str | None = None


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


async def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    health_url = f"{normalize_base_url(base_url)}/health"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        while True:
            try:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"server did not become ready at {health_url}")
            await asyncio.sleep(1.0)


async def run_one_request(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    request_id: int,
    prompt: str,
    decode_tokens: int,
    lora_path: str | None,
    extra_request_body: dict[str, Any],
) -> RequestTrace:
    url = f"{normalize_base_url(base_url)}/generate"
    payload: dict[str, Any] = {
        "text": f"{prompt} [{request_id}]",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": decode_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }
    if lora_path:
        payload["lora_path"] = lora_path
    payload.update(extra_request_body)

    trace = RequestTrace(request_id=request_id, success=False)
    start = time.perf_counter()
    last_seen_completion_tokens = 0
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                trace.error = f"HTTP {response.status}: {await response.text()}"
                return trace

            async for raw in response.content:
                raw = raw.strip()
                if not raw:
                    continue
                chunk = raw.decode("utf-8")
                if chunk.startswith("data: "):
                    chunk = chunk[len("data: ") :]
                if chunk == "[DONE]":
                    continue
                data = json.loads(chunk)
                if not data.get("text"):
                    continue
                meta = data.get("meta_info") or {}
                completion_tokens = int(
                    meta.get("completion_tokens", last_seen_completion_tokens)
                )
                if completion_tokens <= last_seen_completion_tokens:
                    continue
                timestamp = time.perf_counter()
                if trace.first_token_s is None:
                    trace.first_token_s = timestamp
                trace.last_token_s = timestamp
                trace.token_timestamps_s.append((completion_tokens, timestamp))
                last_seen_completion_tokens = completion_tokens

        trace.success = last_seen_completion_tokens >= decode_tokens
        trace.output_len = last_seen_completion_tokens
        trace.latency_s = time.perf_counter() - start
        if not trace.success:
            trace.error = (
                f"request produced {last_seen_completion_tokens} tokens, "
                f"expected at least {decode_tokens}"
            )
    except Exception:
        trace.error = traceback.format_exc()
        trace.latency_s = time.perf_counter() - start
    return trace


async def run_batch(
    *,
    base_url: str,
    batch_size: int,
    prompt: str,
    decode_tokens: int,
    lora_path: str | None,
    warmup_batches: int,
    connection_limit: int,
    extra_request_body: dict[str, Any],
) -> tuple[list[RequestTrace], dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(
        limit=connection_limit, limit_per_host=connection_limit
    )
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, read_bufsize=10 * 1024**2
    ) as session:
        for warmup_idx in range(warmup_batches):
            warmup = [
                run_one_request(
                    session=session,
                    base_url=base_url,
                    request_id=-(warmup_idx * batch_size + i + 1),
                    prompt=prompt,
                    decode_tokens=decode_tokens,
                    lora_path=lora_path,
                    extra_request_body=extra_request_body,
                )
                for i in range(batch_size)
            ]
            await asyncio.gather(*warmup)

        traces = await asyncio.gather(
            *[
                run_one_request(
                    session=session,
                    base_url=base_url,
                    request_id=i,
                    prompt=prompt,
                    decode_tokens=decode_tokens,
                    lora_path=lora_path,
                    extra_request_body=extra_request_body,
                )
                for i in range(batch_size)
            ]
        )

    successful = [t for t in traces if t.success and t.first_token_s and t.last_token_s]
    if len(successful) != batch_size:
        return traces, {
            "success": False,
            "error": f"{len(successful)}/{batch_size} requests completed successfully",
        }

    first_decode_start = min(t.first_token_s for t in successful if t.first_token_s)
    all_started_decode_start = max(
        t.first_token_s for t in successful if t.first_token_s
    )
    decode_end = max(t.last_token_s for t in successful if t.last_token_s)
    full_decode_time_s = max(decode_end - first_decode_start, 0.0)
    all_started_decode_time_s = max(decode_end - all_started_decode_start, 0.0)
    observed_tokens = sum(t.output_len for t in successful)
    tokens_after_all_started = sum(
        1
        for trace in successful
        for _, timestamp in trace.token_timestamps_s
        if timestamp >= all_started_decode_start
    )
    latencies = [t.latency_s for t in successful]
    return traces, {
        "success": True,
        "batch_size": batch_size,
        "decode_tokens_per_request": decode_tokens,
        "completed_requests": len(successful),
        "first_decode_start_s": first_decode_start,
        "all_started_decode_start_s": all_started_decode_start,
        "decode_start_s": first_decode_start,
        "decode_end_s": decode_end,
        "decode_time_s": full_decode_time_s,
        "decode_tok_s": (
            observed_tokens / full_decode_time_s if full_decode_time_s > 0 else None
        ),
        "full_decode_time_s": full_decode_time_s,
        "full_decode_tok_s": (
            observed_tokens / full_decode_time_s if full_decode_time_s > 0 else None
        ),
        "all_started_decode_time_s": all_started_decode_time_s,
        "all_started_decode_tok_s": (
            tokens_after_all_started / all_started_decode_time_s
            if all_started_decode_time_s > 0
            else None
        ),
        "tokens_after_all_started": tokens_after_all_started,
        "observed_tokens": observed_tokens,
        "first_token_spread_s": all_started_decode_start - first_decode_start,
        "mean_request_latency_s": statistics.mean(latencies),
        "median_request_latency_s": statistics.median(latencies),
        "min_output_len": min(t.output_len for t in successful),
        "max_output_len": max(t.output_len for t in successful),
        "prefill_excluded": True,
        "global_decode_window": "min(first_token_time) to max(last_token_time)",
        "all_started_decode_window": "max(first_token_time) to max(last_token_time)",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="Briefly count upward:")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--connection-limit",
        type=int,
        default=0,
        help="aiohttp connector limit; 0 disables the cap so high batches are not split into 100-request waves.",
    )
    parser.add_argument("--extra-request-body", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    extra_request_body = json.loads(args.extra_request_body)
    await wait_ready(args.base_url, args.ready_timeout_s)
    traces, summary = await run_batch(
        base_url=args.base_url,
        batch_size=args.batch_size,
        prompt=args.prompt,
        decode_tokens=args.decode_tokens,
        lora_path=args.lora_path,
        warmup_batches=args.warmup_batches,
        connection_limit=args.connection_limit,
        extra_request_body=extra_request_body,
    )
    payload = {
        "scheme": args.scheme,
        "base_url": args.base_url,
        "summary": summary,
        "requests": [asdict(trace) for trace in traces],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scheme": args.scheme, **summary}, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
