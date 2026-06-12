#!/usr/bin/env python3
"""Run latency matrices and render LaTeX tables."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_ENV_PYTHON = Path("/data/huanchen/miniforge3/envs/sglang/bin/python")
DEFAULT_ENV_TORCHRUN = Path("/data/huanchen/miniforge3/envs/sglang/bin/torchrun")
PYTHON = DEFAULT_ENV_PYTHON if DEFAULT_ENV_PYTHON.exists() else Path("python")
TORCHRUN = DEFAULT_ENV_TORCHRUN if DEFAULT_ENV_TORCHRUN.exists() else Path("torchrun")
BENCHMARK = ROOT / "benchmark_transformer_block.py"
SCOPES = ("block", "attn", "mlp", "qkv", "o", "up", "down")
PRECISIONS = ("bf16", "marlin", "qlora")
BATCH_SIZES = (1, 4, 8, 16, 32, 64, 128, 256, 512)
KV_LEN = 1024
WARMUP = 10
ITERS = 30
L2_FLUSH_MIB = 96
GROUPS = [
    ("core", ("block", "attn", "mlp")),
    ("attn_kernels", ("attn", "qkv", "o")),
    ("mlp_kernels", ("mlp", "up", "down")),
]
DISPLAY_SCOPE = {
    "block": "Block",
    "attn": "Attn",
    "mlp": "MLP",
    "qkv": "QKV",
    "o": "O",
    "up": "Up",
    "down": "Down",
}
DISPLAY_PRECISION = {
    "bf16": "BF16",
    "marlin": "Marlin INT4",
    "qlora": "QLoRA",
}


@dataclass(frozen=True)
class Setup:
    name: str
    model_label: str
    model_config: Path
    tp_size: int


@dataclass(frozen=True)
class Job:
    setup: Setup
    batch_size: int
    precision: str
    scope: str

    @property
    def output_path(self) -> Path:
        return (
            ROOT
            / "latency-results"
            / self.setup.name
            / f"tp{self.setup.tp_size}"
            / f"kv{KV_LEN}"
            / f"bs{self.batch_size}"
            / self.precision
            / f"{self.scope}.json"
        )

    @property
    def log_path(self) -> Path:
        return self.output_path.with_suffix(".log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "latency-results",
    )
    parser.add_argument(
        "--tables-output",
        type=Path,
        default=ROOT / "latency-results" / "tables.tex",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def build_setups() -> list[Setup]:
    return [
        Setup(
            name="qwen2.5-32b",
            model_label="Qwen2.5 32B, TP1",
            model_config=ROOT / "configs" / "qwen2.5-32b.json",
            tp_size=1,
        ),
        Setup(
            name="qwen2.5-32b",
            model_label="Qwen2.5 32B, TP4",
            model_config=ROOT / "configs" / "qwen2.5-32b.json",
            tp_size=4,
        ),
        Setup(
            name="deepseek-r1-distill-qwen-7b",
            model_label="DeepSeek-R1-Distill-Qwen-7B, TP1",
            model_config=ROOT / "configs" / "deepseek-r1-distill-qwen-7b.json",
            tp_size=1,
        ),
    ]


def build_jobs(setups: list[Setup]) -> list[Job]:
    jobs: list[Job] = []
    for setup in setups:
        for batch_size in BATCH_SIZES:
            for scope in SCOPES:
                for precision in PRECISIONS:
                    jobs.append(
                        Job(
                            setup=setup,
                            batch_size=batch_size,
                            precision=precision,
                            scope=scope,
                        )
                    )
    return jobs


def run_job(job: Job, output_root: Path) -> None:
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT / "python")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)

    output_path = output_root / job.output_path.relative_to(ROOT / "latency-results")
    log_path = output_root / job.log_path.relative_to(ROOT / "latency-results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(BENCHMARK),
        "--model-config",
        str(job.setup.model_config),
        "--precision",
        job.precision,
        "--scope",
        job.scope,
        "--batch-size",
        str(job.batch_size),
        "--kv-len",
        str(KV_LEN),
        "--warmup",
        str(WARMUP),
        "--iters",
        str(ITERS),
        "--l2-flush-mib",
        str(L2_FLUSH_MIB),
        "--output",
        str(output_path),
    ]
    if job.setup.tp_size == 1:
        cmd.insert(0, str(PYTHON))
        gpu = threading.current_thread().name.rsplit("-", 1)[-1]
        env["CUDA_VISIBLE_DEVICES"] = gpu
    else:
        group = threading.current_thread().name.rsplit("-", 1)[-1]
        visible = "0,1,2,3" if group == "0" else "4,5,6,7"
        env["CUDA_VISIBLE_DEVICES"] = visible
        cmd = [
            str(TORCHRUN),
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            *cmd,
        ]

    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def drain_jobs(label: str, jobs: list[Job], worker_count: int, output_root: Path, force: bool) -> None:
    pending = queue.Queue()
    runnable: list[Job] = []
    for job in jobs:
        output_path = output_root / job.output_path.relative_to(ROOT / "latency-results")
        if output_path.exists() and not force:
            continue
        runnable.append(job)
        pending.put(job)

    total = len(runnable)
    if total == 0:
        print(f"{label}: nothing to run", flush=True)
        return

    completed = 0
    completed_lock = threading.Lock()
    failures: list[tuple[Job, Exception]] = []

    def worker() -> None:
        nonlocal completed
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(job, output_root)
                with completed_lock:
                    completed += 1
                    print(
                        f"{label}: {completed}/{total} finished "
                        f"{job.setup.model_label} bs={job.batch_size} {job.scope} {job.precision}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append((job, exc))
                with completed_lock:
                    print(
                        f"{label}: FAILED {job.setup.model_label} bs={job.batch_size} "
                        f"{job.scope} {job.precision} -> {job.log_path}",
                        flush=True,
                    )
                return
            finally:
                pending.task_done()

    threads = [
        threading.Thread(target=worker, name=f"{label}-worker-{idx}", daemon=True)
        for idx in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        job, exc = failures[0]
        raise RuntimeError(
            f"{label} failed for {job.setup.model_label} bs={job.batch_size} "
            f"{job.scope} {job.precision}: {exc}"
        )


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_us(value: float) -> str:
    return f"{value:.1f}"


def format_with_speedup(value: float, bf16_value: float) -> str:
    return f"{value:.1f}(x{bf16_value / value:.2f})"


def render_table(setup: Setup, group_name: str, scopes: tuple[str, ...], output_root: Path) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\begin{tabular}{r" + "r" * (len(scopes) * len(PRECISIONS)) + "}"
    )
    lines.append("\\toprule")
    header_a = ["Batch"]
    for scope in scopes:
        header_a.append(f"\\multicolumn{{3}}{{c}}{{{DISPLAY_SCOPE[scope]}}}")
    lines.append(" & ".join(header_a) + " \\\\")
    header_b = [""]
    for _scope in scopes:
        header_b.extend(DISPLAY_PRECISION[p] for p in PRECISIONS)
    lines.append(" & ".join(header_b) + " \\\\")
    lines.append("\\midrule")
    for batch_size in BATCH_SIZES:
        row = [str(batch_size)]
        for scope in scopes:
            bf16_result_path = (
                output_root
                / setup.name
                / f"tp{setup.tp_size}"
                / f"kv{KV_LEN}"
                / f"bs{batch_size}"
                / "bf16"
                / f"{scope}.json"
            )
            bf16_result = load_result(bf16_result_path)
            bf16_median = float(bf16_result["median_us"])
            for precision in PRECISIONS:
                if precision == "bf16":
                    row.append(format_us(bf16_median))
                    continue

                result_path = (
                    output_root
                    / setup.name
                    / f"tp{setup.tp_size}"
                    / f"kv{KV_LEN}"
                    / f"bs{batch_size}"
                    / precision
                    / f"{scope}.json"
                )
                result = load_result(result_path)
                row.append(format_with_speedup(float(result["median_us"]), bf16_median))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    caption = f"{setup.model_label} median decode latency (\\textmu s), kv cache length {KV_LEN}."
    lines.append(f"\\caption{{{caption} {group_name.replace('_', ' ').title()} scopes.}}")
    label = f"tab:{setup.name.replace('.', '').replace('-', '')}-tp{setup.tp_size}-{group_name}"
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def render_tables(setups: list[Setup], output_root: Path, tables_output: Path) -> None:
    sections: list[str] = [
        "% Auto-generated by run_latency_tables.py",
        "\\usepackage{booktabs}",
    ]
    for setup in setups:
        sections.append(f"% {setup.model_label}")
        for group_name, scopes in GROUPS:
            sections.append(render_table(setup, group_name, scopes, output_root))
            sections.append("")
    tables_output.parent.mkdir(parents=True, exist_ok=True)
    tables_output.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setups = build_setups()
    output_root = args.output_root

    if not args.skip_run:
        jobs = build_jobs(setups)
        tp4_jobs = [job for job in jobs if job.setup.tp_size == 4]
        tp1_jobs = [job for job in jobs if job.setup.tp_size == 1]
        start = time.time()
        drain_jobs("tp4", tp4_jobs, worker_count=2, output_root=output_root, force=args.force)
        drain_jobs("tp1", tp1_jobs, worker_count=8, output_root=output_root, force=args.force)
        print(f"all runs complete in {time.time() - start:.1f}s", flush=True)

    render_tables(setups, output_root, args.tables_output)
    print(f"wrote LaTeX tables to {args.tables_output}", flush=True)


if __name__ == "__main__":
    main()
