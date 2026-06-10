#!/usr/bin/env python3
"""Run TP-shaped Marlin and flipped Torch QLoRA kernel measurements.

The full JSON measurements use the same benchmark mechanics as
``.rollout-profile/qlora-kernel``: torch.compile for base/Torch callables,
CUDA Graph capture/replay, and optional L2 flushing. Nsight profiles are kept
to the representative ``down``/token-row-1 projection so the timeline PNGs are
readable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PYTHON = Path("/data/huanchen/miniforge3/envs/sglang/bin/python")
TORCHRUN = Path("/data/huanchen/miniforge3/envs/sglang/bin/torchrun")
BENCHMARK = ROOT / "benchmark.py"
SGLANG_TP_BENCHMARK = ROOT / "sglang_tp_linear_benchmark.py"
PEEK = REPO_ROOT / ".rollout-profile" / "qlora-kernel" / "nsys" / "peek_nsys_graph.py"
TPS = (1, 2, 4)
SCHEMES = {
    "bf16": "bf16 dense base",
    "marlin": "int4 Marlin base",
    "qlora": "Torch QLoRA matmul two-stream",
}


def run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def run_first_success(commands: list[list[str]], *, env: dict[str, str]) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for cmd in commands:
        print("+ " + " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"command failed with exit code {exc.returncode}; trying next fallback", flush=True)
    assert last_error is not None
    raise last_error


def base_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(REPO_ROOT / "python"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return env


def config_path(tp: int) -> Path:
    return ROOT / "configs" / f"qwen2.5-32b-tp{tp}.json"


def result_path(kind: str, tp: int) -> Path:
    return ROOT / kind / f"tp{tp}" / f"qwen32b_tp{tp}_{kind}.json"


def nsys_stem(kind: str, tp: int) -> Path:
    return ROOT / kind / f"tp{tp}" / "nsys" / f"qwen32b_tp{tp}_{kind}_down_bs1"


def sglang_tp_stem(kind: str, tp: int) -> Path:
    return (
        ROOT
        / "sglang-tp"
        / kind
        / f"tp{tp}-up-down"
        / "nsys"
        / f"qwen32b_tp{tp}_{kind}_up_down_bs1"
    )


def benchmark_cmd(kind: str, tp: int, output: Path, *, profiler: bool) -> list[str]:
    cmd = [
        str(PYTHON),
        str(BENCHMARK),
        "--model-config",
        str(config_path(tp)),
        "--scheme",
        SCHEMES[kind],
        "--output",
        str(output),
        "--two-stream-reserve-sms",
        "1",
        "--two-stream-layout",
        "flipped",
    ]
    if profiler:
        cmd.extend(
            [
                "--projection",
                "down",
                "--tokens",
                "1",
                "--warmup",
                "5",
                "--iters",
                "10",
                "--cuda-profiler-range",
                "--tp-all-reduce",
            ]
        )
    return cmd


def profile_benchmark_cmd(kind: str, tp: int, output: Path) -> list[str]:
    return [
        str(TORCHRUN),
        "--standalone",
        "--nproc_per_node",
        str(tp),
        str(BENCHMARK),
    ] + benchmark_cmd(kind, tp, output, profiler=True)[2:]


def sglang_tp_profile_cmd(kind: str, tp: int, output: Path) -> list[str]:
    cmd = [
        str(TORCHRUN),
        "--standalone",
        "--nproc_per_node",
        str(tp),
        str(SGLANG_TP_BENCHMARK),
        "--kind",
        kind,
        "--tokens",
        "1",
        "--warmup",
        "5",
        "--iters",
        "10",
        "--cuda-profiler-range",
        "--output",
        str(output),
    ]
    if tp > 1:
        cmd.append("--force-nccl-all-reduce")
    return cmd


def run_measurements(args: argparse.Namespace, env: dict[str, str]) -> None:
    for kind in args.kind:
        for tp in args.tp:
            out = result_path(kind, tp)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not args.force:
                print(f"skip existing {out}", flush=True)
                continue
            run(benchmark_cmd(kind, tp, out, profiler=False), env=env)


def run_profiles(args: argparse.Namespace, env: dict[str, str]) -> None:
    for kind in args.kind:
        for tp in args.tp:
            stem = nsys_stem(kind, tp)
            stem.parent.mkdir(parents=True, exist_ok=True)
            rep = stem.with_suffix(".nsys-rep")
            sqlite = stem.with_suffix(".sqlite")
            png = stem.with_name(stem.name + "_peek.png")
            profile_json = stem.with_suffix(".json")
            if rep.exists() and sqlite.exists() and png.exists() and not args.force:
                print(f"skip existing profile {stem}", flush=True)
                continue

            visible_gpus = [gpu.strip() for gpu in args.gpu.split(",") if gpu.strip()]
            if len(visible_gpus) < tp:
                raise RuntimeError(
                    f"{kind}/tp{tp} profile needs {tp} visible GPUs, got {args.gpu!r}"
                )

            profile_cmd = [
                "nsys",
                "profile",
                "--output",
                str(stem),
                "--trace=cuda,nvtx,osrt",
                "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node",
                "--force-overwrite=true",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
            ] + profile_benchmark_cmd(kind, tp, profile_json)
            run(profile_cmd, env=env)

            run(
                [
                    "nsys",
                    "export",
                    "--type",
                    "sqlite",
                    "--force-overwrite=true",
                    "--output",
                    str(sqlite),
                    str(rep),
                ],
                env=env,
            )


def run_sglang_tp_profiles(args: argparse.Namespace, env: dict[str, str]) -> None:
    for kind in args.kind:
        for tp in args.tp:
            stem = sglang_tp_stem(kind, tp)
            stem.parent.mkdir(parents=True, exist_ok=True)
            rep = stem.with_suffix(".nsys-rep")
            sqlite = stem.with_suffix(".sqlite")
            png = stem.with_name(stem.name + "_peek.png")
            profile_json = stem.with_suffix(".json")
            if rep.exists() and sqlite.exists() and png.exists() and not args.force:
                print(f"skip existing SGLang TP profile {stem}", flush=True)
                continue

            visible_gpus = [gpu.strip() for gpu in args.gpu.split(",") if gpu.strip()]
            if len(visible_gpus) < tp:
                raise RuntimeError(
                    f"{kind}/tp{tp} SGLang TP profile needs {tp} visible GPUs, got {args.gpu!r}"
                )

            profile_cmd = [
                "nsys",
                "profile",
                "--output",
                str(stem),
                "--trace=cuda,nvtx,osrt",
                "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node",
                "--force-overwrite=true",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
            ] + sglang_tp_profile_cmd(kind, tp, profile_json)
            run(profile_cmd, env=env)
            run(
                [
                    "nsys",
                    "export",
                    "--type",
                    "sqlite",
                    "--force-overwrite=true",
                    "--output",
                    str(sqlite),
                    str(rep),
                ],
                env=env,
            )
            anchor_candidates = ["nccl", "all_reduce", "allreduce", "Marlin", "Kernel", "gemm", "matmul"]
            run_first_success(
                [
                    [
                        str(PYTHON),
                        str(PEEK),
                        str(sqlite),
                        "--output",
                        str(png),
                        "--anchor-substring",
                        anchor,
                        "--window-us",
                        "260",
                        "--title",
                        f"Qwen2.5-32B real SGLang TP{tp} {kind} up_down bs1",
                    ]
                    for anchor in anchor_candidates
                ],
                env=env,
            )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU list. When set, independent TP/scheme jobs are fanned out across these GPUs.",
    )
    parser.add_argument("--tp", type=int, nargs="+", default=list(TPS), choices=TPS)
    parser.add_argument(
        "--kind",
        nargs="+",
        default=list(SCHEMES),
        choices=sorted(SCHEMES),
    )
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--sglang-tp-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_parallel_children(args: argparse.Namespace) -> bool:
    if not args.gpus:
        return False
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpus) <= 1:
        args.gpu = gpus[0] if gpus else args.gpu
        return False

    jobs = [(kind, tp) for kind in args.kind for tp in args.tp]
    active: list[tuple[subprocess.Popen, str, list[str]]] = []
    available_gpus = list(gpus)
    next_job = 0
    failed = False

    def launch(kind: str, tp: int, gpu_group: list[str]) -> None:
        cmd = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "--gpu",
            ",".join(gpu_group),
            "--kind",
            kind,
            "--tp",
            str(tp),
        ]
        if args.measure_only:
            cmd.append("--measure-only")
        if args.profile_only:
            cmd.append("--profile-only")
        if args.sglang_tp_only:
            cmd.append("--sglang-tp-only")
        if args.force:
            cmd.append("--force")
        label = f"{kind}/tp{tp}/gpu{','.join(gpu_group)}"
        print("+ [" + label + "] " + " ".join(cmd), flush=True)
        active.append((subprocess.Popen(cmd, cwd=REPO_ROOT), label, gpu_group))

    while next_job < len(jobs) or active:
        launched = False
        while next_job < len(jobs):
            kind, tp = jobs[next_job]
            need = tp if not args.measure_only else 1
            if len(available_gpus) < need:
                break
            gpu_group = available_gpus[:need]
            available_gpus = available_gpus[need:]
            launch(kind, tp, gpu_group)
            next_job += 1
            launched = True
        for proc, label, gpu_group in list(active):
            rc = proc.poll()
            if rc is None:
                continue
            active.remove((proc, label, gpu_group))
            available_gpus.extend(gpu_group)
            if rc != 0:
                failed = True
                print(f"{label} failed with exit code {rc}", flush=True)
        if active or (next_job < len(jobs) and not launched):
            time.sleep(1.0)

    if failed:
        raise SystemExit(1)
    return True


def main() -> None:
    args = parse_args()
    if run_parallel_children(args):
        return
    env = base_env(args.gpu)
    if args.sglang_tp_only:
        run_sglang_tp_profiles(args, env)
        return
    if not args.profile_only:
        run_measurements(args, env)
    if not args.measure_only:
        run_profiles(args, env)


if __name__ == "__main__":
    main()
