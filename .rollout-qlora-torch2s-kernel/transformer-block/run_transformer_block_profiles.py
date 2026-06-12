#!/usr/bin/env python3
"""Run a scope/precision matrix for the transformer block benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_ENV_PYTHON = Path("/data/huanchen/miniforge3/envs/sglang/bin/python")
PYTHON = DEFAULT_ENV_PYTHON if DEFAULT_ENV_PYTHON.exists() else Path(sys.executable)
BENCHMARK = ROOT / "benchmark_transformer_block.py"
DEFAULT_SCOPES = ("block", "attn", "mlp", "qkv", "o", "up", "down")
DEFAULT_PRECISIONS = ("bf16", "marlin", "qlora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--scope", nargs="+", default=list(DEFAULT_SCOPES), choices=DEFAULT_SCOPES)
    parser.add_argument(
        "--precision",
        nargs="+",
        default=list(DEFAULT_PRECISIONS),
        choices=DEFAULT_PRECISIONS,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--l2-flush-mib", type=int, default=96)
    parser.add_argument("--two-stream-reserve-sms", type=int, default=1)
    parser.add_argument("--two-stream-layout", choices=["flipped", "standard"], default="flipped")
    parser.add_argument("--output-root", type=Path, default=ROOT / "nsys")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT / "python")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)

    model_name = args.model_config.stem
    run_root = args.output_root / model_name / f"bs{args.batch_size}_kv{args.kv_len}"

    for precision in args.precision:
        for scope in args.scope:
            output = run_root / precision / f"{scope}.json"
            rep = output.with_suffix(".nsys-rep")
            sqlite = output.with_suffix(".sqlite")
            if output.exists() and rep.exists() and sqlite.exists() and not args.force:
                print(f"skip existing {output}", flush=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(PYTHON),
                str(BENCHMARK),
                "--model-config",
                str(args.model_config),
                "--precision",
                precision,
                "--scope",
                scope,
                "--batch-size",
                str(args.batch_size),
                "--kv-len",
                str(args.kv_len),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--l2-flush-mib",
                str(args.l2_flush_mib),
                "--two-stream-reserve-sms",
                str(args.two_stream_reserve_sms),
                "--two-stream-layout",
                args.two_stream_layout,
                "--output",
                str(output),
                "--profile-nsys",
            ]
            if args.no_cuda_graph:
                cmd.append("--no-cuda-graph")
            if args.no_torch_compile:
                cmd.append("--no-torch-compile")
            run(cmd, env)


if __name__ == "__main__":
    main()
