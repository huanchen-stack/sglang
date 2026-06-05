"""Rollout decode-lifecycle experiment.

Mimic an RL-rollout workload: launch a batch of N prompts together, let each
request decode until it emits its OWN eos (or hits max_new_tokens), and record
the request lifecycle so we can plot:

  (1) lifetime  : # live (not-yet-finished) requests vs time since decode start
  (2) duration  : wall-clock time spent at each halving live-count bin

Ground truth comes from the server-side rollout trace (SGLANG_ROLLOUT_TRACE_DIR),
which logs decode phase_start (-> decode start t0) and per-request
request_complete (-> finish time + finish_reason).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


def wait_ready(base_url: str, timeout_s: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError("server did not become ready")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def load_prompts(dataset_path: Path, n: int) -> list[str]:
    prompts: list[str] = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            convs = row.get("conversations", row.get("conversation", []))
            if len(convs) >= 1:
                user = convs[0].get("value", convs[0].get("content", ""))
                if user:
                    prompts.append(user)
    if not prompts:
        raise RuntimeError(f"no prompts in {dataset_path}")
    # tile if we need more than available
    out = [prompts[i % len(prompts)] for i in range(n)]
    return out


def send_one(base_url: str, model: str, prompt: str, args) -> dict[str, Any]:
    t0 = time.time()
    try:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "stream": False,
            },
            timeout=args.request_timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        return {
            "ok": True,
            "client_start": t0,
            "client_end": time.time(),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "client_start": t0, "client_end": time.time(),
                "error": repr(exc)}


def run(args) -> None:
    out_dir = Path(args.output_dir)
    trace_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    served = args.model_label

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["SGLANG_ROLLOUT_TRACE_DIR"] = str(trace_dir)
    env.setdefault("HF_HOME", "/data/huggingface")

    s_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--served-model-name", served,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--tp-size", str(args.tp),
        "--dtype", "bfloat16",
        "--max-running-requests", str(args.batch_size),
        "--cuda-graph-max-bs", str(args.batch_size),
        "--context-length", str(args.context_length),
        "--mem-fraction-static", str(args.mem_fraction),
    ]
    (out_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    (out_dir / "server_cmd.txt").write_text(" ".join(s_cmd) + "\n")

    server_log = (out_dir / "server.log").open("w")
    proc = subprocess.Popen(s_cmd, cwd=args.repo_root, env=env,
                            stdout=server_log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    results: list[dict[str, Any]] = []
    try:
        wait_ready(base_url, args.ready_timeout, proc)
        prompts = load_prompts(Path(args.dataset_path), args.batch_size)
        print(f"[{args.model_label}/{args.category}/bs{args.batch_size}] "
              f"firing {len(prompts)} requests...", flush=True)
        wall_start = time.time()
        with ThreadPoolExecutor(max_workers=args.batch_size) as ex:
            futs = [ex.submit(send_one, base_url, served, p, args) for p in prompts]
            results = [f.result() for f in futs]
        wall_end = time.time()
        n_ok = sum(1 for r in results if r["ok"])
        print(f"  done: {n_ok}/{len(results)} ok in {wall_end-wall_start:.1f}s", flush=True)
    finally:
        terminate(proc)
        server_log.close()

    (out_dir / "client_results.json").write_text(
        json.dumps({"wall_s": (wall_end - wall_start) if results else None,
                    "results": results}, indent=2) + "\n")
    print(f"  wrote {out_dir}/client_results.json", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--mem-fraction", type=float, default=0.9)
    p.add_argument("--ready-timeout", type=int, default=2400)
    p.add_argument("--request-timeout", type=int, default=7200)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
