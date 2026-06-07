#!/usr/bin/env python3
"""Launch SGLang policies and measure decode-only throughput."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
CLIENT = OUT_DIR / "decoding_client.py"


@dataclass(frozen=True)
class SchemeConfig:
    name: str
    model_path: str
    request_lora_path: str | None
    server_args: list[str]
    env: dict[str, str]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scheme_from_config(raw: dict, name: str) -> SchemeConfig:
    schemes = raw.get("schemes", {})
    if name not in schemes:
        raise KeyError(f"scheme {name!r} is not defined in {raw.get('name', '<config>')}")
    item = schemes[name]
    return SchemeConfig(
        name=name,
        model_path=item["model_path"],
        request_lora_path=item.get("request_lora_path", item.get("lora_path")),
        server_args=list(item.get("server_args", [])),
        env={str(k): str(v) for k, v in item.get("env", {}).items()},
    )


def build_server_cmd(
    *,
    python: str,
    scheme: SchemeConfig,
    host: str,
    port: int,
    common_server_args: list[str],
) -> list[str]:
    return [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        scheme.model_path,
        "--host",
        host,
        "--port",
        str(port),
        *common_server_args,
        *scheme.server_args,
    ]


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def run_client(
    *,
    python: str,
    base_url: str,
    scheme: SchemeConfig,
    batch_size: int,
    decode_tokens: int,
    prompt: str,
    warmup_batches: int,
    ready_timeout_s: float,
    output: Path,
    extra_request_body: dict,
) -> None:
    cmd = [
        python,
        str(CLIENT),
        "--base-url",
        base_url,
        "--scheme",
        scheme.name,
        "--batch-size",
        str(batch_size),
        "--decode-tokens",
        str(decode_tokens),
        "--prompt",
        prompt,
        "--warmup-batches",
        str(warmup_batches),
        "--ready-timeout-s",
        str(ready_timeout_s),
        "--extra-request-body",
        json.dumps(extra_request_body),
        "--output",
        str(output),
    ]
    if scheme.request_lora_path:
        cmd.extend(["--lora-path", scheme.request_lora_path])
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=OUT_DIR / "configs" / "qwen2.5-14b.json",
    )
    parser.add_argument("--scheme", action="append", default=None)
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument("--decode-tokens", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--warmup-batches", type=int, default=None)
    parser.add_argument("--ready-timeout-s", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--base-url", default=None, help="Attach to an existing server.")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPUs. Runs schemes in parallel, one scheme per GPU.",
    )
    parser.add_argument(
        "--parallel-jobs",
        action="store_true",
        help="With --gpus, run each scheme/batch pair as an independent GPU job.",
    )
    parser.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        help="Extra launch_server argument appended after config args. Repeat for flag/value pairs.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "measurements_14b_current")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_parallel_scheme_workers(args: argparse.Namespace, raw: dict) -> None:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    scheme_names = args.scheme or list(raw.get("default_schemes", raw.get("schemes", {})))
    procs: list[subprocess.Popen] = []
    for idx, scheme_name in enumerate(scheme_names):
        gpu = gpus[idx % len(gpus)]
        cmd = [
            args.python,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--scheme",
            scheme_name,
            "--gpu",
            gpu,
            "--port",
            str(args.port + idx),
            "--out-dir",
            str(args.out_dir),
        ]
        for batch_size in args.batch_size or []:
            cmd.extend(["--batch-size", str(batch_size)])
        if args.decode_tokens is not None:
            cmd.extend(["--decode-tokens", str(args.decode_tokens)])
        if args.prompt is not None:
            cmd.extend(["--prompt", args.prompt])
        if args.warmup_batches is not None:
            cmd.extend(["--warmup-batches", str(args.warmup_batches)])
        if args.ready_timeout_s is not None:
            cmd.extend(["--ready-timeout-s", str(args.ready_timeout_s)])
        for extra_arg in args.extra_server_arg or []:
            cmd.append(f"--extra-server-arg={extra_arg}")
        if args.no_launch:
            cmd.append("--no-launch")
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[gpu {gpu}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            procs.append(subprocess.Popen(cmd))

    failures = []
    for proc in procs:
        returncode = proc.wait()
        if returncode != 0:
            failures.append(returncode)
    if failures:
        raise SystemExit(f"parallel workers failed with codes: {failures}")


def run_parallel_job_workers(args: argparse.Namespace, raw: dict) -> None:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    scheme_names = args.scheme or list(raw.get("default_schemes", raw.get("schemes", {})))
    batch_sizes = args.batch_size or list(raw.get("batch_sizes", [1, 4, 8, 16]))
    jobs = [
        (scheme_name, int(batch_size))
        for scheme_name in scheme_names
        for batch_size in batch_sizes
    ]

    def build_job_cmd(idx: int, scheme_name: str, batch_size: int, gpu: str) -> list[str]:
        cmd = [
            args.python,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--scheme",
            scheme_name,
            "--batch-size",
            str(batch_size),
            "--gpu",
            gpu,
            "--port",
            str(args.port + idx),
            "--out-dir",
            str(args.out_dir),
            "--extra-server-arg=--cuda-graph-max-bs",
            f"--extra-server-arg={batch_size}",
            "--extra-server-arg=--torch-compile-max-bs",
            f"--extra-server-arg={batch_size}",
        ]
        if args.decode_tokens is not None:
            cmd.extend(["--decode-tokens", str(args.decode_tokens)])
        if args.prompt is not None:
            cmd.extend(["--prompt", args.prompt])
        if args.warmup_batches is not None:
            cmd.extend(["--warmup-batches", str(args.warmup_batches)])
        if args.ready_timeout_s is not None:
            cmd.extend(["--ready-timeout-s", str(args.ready_timeout_s)])
        for extra_arg in args.extra_server_arg or []:
            cmd.append(f"--extra-server-arg={extra_arg}")
        if args.no_launch:
            cmd.append("--no-launch")
        if args.dry_run:
            cmd.append("--dry-run")
        return cmd

    if args.dry_run:
        for idx, (scheme_name, batch_size) in enumerate(jobs):
            gpu = gpus[idx % len(gpus)]
            cmd = build_job_cmd(idx, scheme_name, batch_size, gpu)
            print(f"[gpu {gpu}] {' '.join(cmd)}", flush=True)
        return

    next_job = 0
    active: list[tuple[subprocess.Popen, str]] = []
    failures = []
    while next_job < len(jobs) or active:
        while next_job < len(jobs) and len(active) < len(gpus):
            active_gpus = {gpu for _, gpu in active}
            gpu = next(gpu for gpu in gpus if gpu not in active_gpus)
            scheme_name, batch_size = jobs[next_job]
            cmd = build_job_cmd(next_job, scheme_name, batch_size, gpu)
            print(f"[gpu {gpu}] {' '.join(cmd)}", flush=True)
            active.append((subprocess.Popen(cmd), gpu))
            next_job += 1

        still_active: list[tuple[subprocess.Popen, str]] = []
        for proc, gpu in active:
            returncode = proc.poll()
            if returncode is None:
                still_active.append((proc, gpu))
            elif returncode != 0:
                failures.append(returncode)
        active = still_active
        if active and next_job < len(jobs):
            time.sleep(2.0)

    if failures:
        raise SystemExit(f"parallel jobs failed with codes: {failures}")


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    if args.gpus is not None:
        if args.parallel_jobs:
            run_parallel_job_workers(args, raw)
        else:
            run_parallel_scheme_workers(args, raw)
        return

    decode_tokens = int(
        args.decode_tokens if args.decode_tokens is not None else raw.get("decode_tokens", 256)
    )
    prompt = str(args.prompt if args.prompt is not None else raw.get("prompt", "Briefly count upward:"))
    warmup_batches = int(
        args.warmup_batches
        if args.warmup_batches is not None
        else raw.get("warmup_batches", 1)
    )
    ready_timeout_s = float(
        args.ready_timeout_s
        if args.ready_timeout_s is not None
        else raw.get("ready_timeout_s", 1800.0)
    )
    batch_sizes = args.batch_size or list(raw.get("batch_sizes", [1, 4, 8, 16]))
    scheme_names = args.scheme or list(raw.get("default_schemes", raw.get("schemes", {})))
    common_server_args = list(raw.get("common_server_args", []))
    common_server_args.extend(args.extra_server_arg or [])
    extra_request_body = dict(raw.get("extra_request_body", {}))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for scheme_name in scheme_names:
        scheme = scheme_from_config(raw, scheme_name)
        base_url = args.base_url or f"http://{args.host}:{args.port}"
        proc = None
        log_name = f"{scheme.name}.server.log"
        if len(batch_sizes) == 1:
            log_name = f"{scheme.name}.bs{batch_sizes[0]}.server.log"
        log_file = args.out_dir / log_name
        env = os.environ.copy()
        env.update(scheme.env)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu

        if not args.no_launch and args.base_url is None:
            cmd = build_server_cmd(
                python=args.python,
                scheme=scheme,
                host=args.host,
                port=args.port,
                common_server_args=common_server_args,
            )
            if args.dry_run:
                print(" ".join(cmd))
            else:
                log_handle = log_file.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                # Give import/JIT errors a chance to fail before the client waits.
                time.sleep(5)
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"server for {scheme.name} exited early; see {log_file}"
                    )

        try:
            for batch_size in batch_sizes:
                output = args.out_dir / f"{scheme.name}.bs{batch_size}.json"
                if args.dry_run:
                    print(
                        f"{args.python} {CLIENT} --base-url {base_url} "
                        f"--scheme {scheme.name} --batch-size {batch_size} "
                        f"--decode-tokens {decode_tokens} --output {output}"
                    )
                    continue
                run_client(
                    python=args.python,
                    base_url=base_url,
                    scheme=scheme,
                    batch_size=batch_size,
                    decode_tokens=decode_tokens,
                    prompt=prompt,
                    warmup_batches=warmup_batches,
                    ready_timeout_s=ready_timeout_s,
                    output=output,
                    extra_request_body=extra_request_body,
                )
        finally:
            if proc is not None:
                terminate_process(proc)


if __name__ == "__main__":
    main()
