#!/usr/bin/env python3
"""Probe a no-custom-kernel TP-style split-K LoRA shrink path.

This script intentionally uses ordinary PyTorch matmul kernels.  It simulates:

    x @ A.T = sum_i x[:, k_i:k_{i+1}] @ A[:, k_i:k_{i+1}].T

The multi-stream variant launches each K shard on a side stream, then reduces
the tiny partial outputs before running the LoRA B expansion.  CUDA Graph replay
is used for all measured timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[1]


@dataclass(frozen=True)
class ProjectionConfig:
    name: str
    kind: str
    in_features: int
    out_features: int
    slice_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ProbeConfig:
    model: str
    projection: str
    token_rows: int
    rank: int
    alpha: float
    dtype: str
    warmup_iters: int
    measure_iters: int
    l2_flush_mib: int
    compile_mode: str
    split_align: int


@dataclass(frozen=True)
class ProbeRow:
    model: str
    projection: str
    token_rows: int
    in_features: int
    out_features: int
    rank: int
    variant: str
    splits: int
    latency_us: float
    p20_us: float
    p80_us: float
    min_us: float
    max_us: float
    note: str


def add_repo_python_to_path() -> None:
    python_dir = REPO_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def dtype_from_name(name: str):
    import torch

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def parse_csv_ints(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out or any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def load_projection(path: Path, name: str) -> tuple[str, ProjectionConfig, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for item in raw["projections"]:
        if item["name"] == name:
            slice_sizes = tuple(int(x) for x in item.get("slice_sizes", [item["out_features"]]))
            return (
                str(raw["model"]),
                ProjectionConfig(
                    name=str(item["name"]),
                    kind=str(item.get("kind", "simple")),
                    in_features=int(item["in_features"]),
                    out_features=int(item["out_features"]),
                    slice_sizes=slice_sizes,
                ),
                raw,
            )
    raise ValueError(f"{path} has no projection named {name!r}")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def split_ranges(k_dim: int, splits: int, align: int) -> list[tuple[int, int]]:
    if splits == 1:
        return [(0, k_dim)]
    if k_dim % align != 0:
        raise ValueError(f"K={k_dim} must be divisible by split_align={align}")
    chunks = k_dim // align
    if splits > chunks:
        raise ValueError(f"splits={splits} is too large for K={k_dim}, align={align}")

    ranges = []
    start_chunk = 0
    for i in range(splits):
        end_chunk = round((i + 1) * chunks / splits)
        if end_chunk <= start_chunk:
            end_chunk = start_chunk + 1
        ranges.append((start_chunk * align, end_chunk * align))
        start_chunk = end_chunk
    ranges[-1] = (ranges[-1][0], k_dim)
    return ranges


def compile_callable(fn: Callable, mode: str) -> Callable:
    import torch

    if mode == "none":
        return fn
    if mode == "eager":
        return torch.compile(fn, fullgraph=False, backend="eager")
    if mode == "default":
        return torch.compile(fn, fullgraph=False)
    return torch.compile(fn, fullgraph=False, mode=mode)


def prewarm_activation(x, fn: Callable[[], object]) -> None:
    import torch

    with torch.no_grad():
        _ = x.sum()
        fn()
        torch.cuda.synchronize()


def capture_cuda_graph(fn: Callable[[], object]):
    import torch

    graph = torch.cuda.CUDAGraph()
    holder = {}
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        holder["out"] = fn()
    return graph, holder


def replay_and_measure(graph, flush_buffer, warmup_iters: int, measure_iters: int) -> list[float]:
    import torch

    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(measure_iters):
        if flush_buffer is not None:
            flush_buffer.fill_(i)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


class SplitKLoRAProbe:
    def __init__(
        self,
        *,
        x,
        lora_a,
        lora_b,
        projection: ProjectionConfig,
        rank: int,
        scaling: float,
        splits: int,
        split_align: int,
        compile_mode: str,
        multistream: bool,
    ):
        import torch

        self.x = x
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.projection = projection
        self.rank = rank
        self.scaling = scaling
        self.ranges = split_ranges(projection.in_features, splits, split_align)
        self.multistream = multistream
        self.streams = [
            torch.cuda.Stream(device=x.device) for _ in self.ranges
        ] if multistream else []
        self.a_t_shards = [
            lora_a[:, k0:k1].t().contiguous() for k0, k1 in self.ranges
        ]
        self.shard_mm_fn = compile_callable(self._shard_mm, compile_mode)
        self.reduce_expand_fn = compile_callable(self._reduce_expand, compile_mode)
        self.full_patch_fn = compile_callable(self._full_patch, compile_mode)

    def _expand(self, shrink):
        import torch

        parts = []
        out_offset = 0
        for idx, slice_size in enumerate(self.projection.slice_sizes):
            r0 = idx * self.rank
            r1 = r0 + self.rank
            out_slice = self.lora_b[out_offset : out_offset + slice_size, :]
            parts.append(torch.mm(shrink[:, r0:r1], out_slice.t()))
            out_offset += slice_size
        if len(parts) == 1:
            return parts[0] * self.scaling
        return torch.cat(parts, dim=-1) * self.scaling

    def _full_patch(self):
        import torch

        shrink = torch.mm(self.x, self.lora_a.t())
        return self._expand(shrink)

    def _shard_mm(self, x_shard, a_t):
        import torch

        return torch.mm(x_shard, a_t)

    def _reduce_expand(self, *partials):
        shrink = partials[0]
        for partial in partials[1:]:
            shrink = shrink + partial
        return self._expand(shrink)

    def torch_full(self):
        return self.full_patch_fn()

    def tp_sequential(self):
        partials = [
            self.shard_mm_fn(self.x[:, k0:k1], a_t)
            for (k0, k1), a_t in zip(self.ranges, self.a_t_shards)
        ]
        return self.reduce_expand_fn(*partials)

    def tp_multistream(self):
        import torch

        current = torch.cuda.current_stream(self.x.device)
        partials = [None] * len(self.ranges)
        for idx, (stream, (k0, k1), a_t) in enumerate(
            zip(self.streams, self.ranges, self.a_t_shards)
        ):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                partials[idx] = self.shard_mm_fn(self.x[:, k0:k1], a_t)
        for stream in self.streams:
            current.wait_stream(stream)
        return self.reduce_expand_fn(*partials)


def measure_one(fn, *, x, flush_buffer, config: ProbeConfig, cuda_profiler_range: bool) -> tuple[float, list[float]]:
    import torch

    prewarm_activation(x, fn)
    for _ in range(config.warmup_iters):
        fn()
    torch.cuda.synchronize()
    graph, _ = capture_cuda_graph(fn)
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        timings = replay_and_measure(
            graph,
            flush_buffer,
            config.warmup_iters,
            config.measure_iters,
        )
    finally:
        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
    return statistics.median(timings), timings


def run(args: argparse.Namespace) -> dict:
    add_repo_python_to_path()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_name, projection, raw_model = load_projection(args.model_config, args.projection)
    rank = int(args.rank or raw_model.get("lora_rank", 16))
    alpha = float(args.alpha or raw_model.get("lora_alpha", rank))
    dtype_name = str(args.dtype or raw_model.get("dtype", "bf16"))
    dtype = dtype_from_name(dtype_name)

    config = ProbeConfig(
        model=model_name,
        projection=projection.name,
        token_rows=args.tokens,
        rank=rank,
        alpha=alpha,
        dtype=dtype_name,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        l2_flush_mib=args.l2_flush_mib,
        compile_mode=args.compile_mode,
        split_align=args.split_align,
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    x = torch.randn((args.tokens, projection.in_features), device=device, dtype=dtype).contiguous()
    lora_a_cols = rank * len(projection.slice_sizes)
    lora_a = torch.randn((lora_a_cols, projection.in_features), device=device, dtype=dtype).contiguous()
    lora_b = torch.randn((projection.out_features, rank), device=device, dtype=dtype).contiguous()

    flush_buffer = None
    if args.l2_flush_mib > 0:
        elements = args.l2_flush_mib * 1024 * 1024 // torch.empty((), dtype=torch.int32).element_size()
        flush_buffer = torch.empty((elements,), device=device, dtype=torch.int32)

    variants = set(args.variant or ["torch-full", "tp-sequential", "tp-multistream"])
    rows: list[ProbeRow] = []
    for splits in args.splits:
        probe = SplitKLoRAProbe(
            x=x,
            lora_a=lora_a,
            lora_b=lora_b,
            projection=projection,
            rank=rank,
            scaling=alpha / rank,
            splits=splits,
            split_align=args.split_align,
            compile_mode=args.compile_mode,
            multistream="tp-multistream" in variants,
        )
        providers = {
            "torch-full": (
                probe.torch_full,
                "Single PyTorch GEMM shrink, then PyTorch GEMM expand.",
            ),
            "tp-sequential": (
                probe.tp_sequential,
                "K-sharded PyTorch GEMM shrink shards on the current stream, then local reduce and expand.",
            ),
            "tp-multistream": (
                probe.tp_multistream,
                "K-sharded PyTorch GEMM shrink shards on side streams, then local reduce and expand.",
            ),
        }
        for variant, (fn, note) in providers.items():
            if variant not in variants:
                continue
            print(f"measuring variant={variant} splits={splits}", file=sys.stderr, flush=True)
            median_us, timings = measure_one(
                fn,
                x=x,
                flush_buffer=flush_buffer,
                config=config,
                cuda_profiler_range=args.cuda_profiler_range,
            )
            rows.append(
                ProbeRow(
                    model=model_name,
                    projection=projection.name,
                    token_rows=args.tokens,
                    in_features=projection.in_features,
                    out_features=projection.out_features,
                    rank=rank,
                    variant=variant,
                    splits=splits,
                    latency_us=median_us,
                    p20_us=percentile(timings, 0.20),
                    p80_us=percentile(timings, 0.80),
                    min_us=min(timings),
                    max_us=max(timings),
                    note=note,
                )
            )

    payload = {
        "metadata": {
            "created_unix": time.time(),
            "config": asdict(config),
            "model_config": str(args.model_config),
            "projection_kind": projection.kind,
            "slice_sizes": list(projection.slice_sizes),
            "explanation": [
                "This is a no-custom-kernel split-K LoRA A probe.",
                "tp-multistream launches one PyTorch matmul per K shard on side streams and reduces the partial rank outputs.",
                "The measured path is captured and replayed with CUDA Graph.",
            ],
        },
        "measurements": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.plot_output is not None:
        write_plot(payload, args.plot_output)
    return payload


def write_plot(payload: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = payload["measurements"]
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])

    colors = {
        "torch-full": "#4C78A8",
        "tp-sequential": "#F28E2B",
        "tp-multistream": "#E15759",
    }
    labels = {
        "torch-full": "single cuBLAS shrink+expand",
        "tp-sequential": "TP split-K sequential",
        "tp-multistream": "TP split-K multi-stream",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for variant in variants:
        points = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: row["splits"],
        )
        ax.plot(
            [row["splits"] for row in points],
            [row["latency_us"] for row in points],
            marker="o",
            linewidth=2.0,
            color=colors.get(variant),
            label=labels.get(variant, variant),
        )

    ax.set_title(
        f"{payload['metadata']['config']['model']} {payload['metadata']['config']['projection']} "
        f"bs={payload['metadata']['config']['token_rows']} LoRA rank={payload['metadata']['config']['rank']}"
    )
    ax.set_xlabel("K splits")
    ax.set_ylabel("CUDA Graph replay median latency (us)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["splits"] for row in rows}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=OUT_DIR / "configs" / "qwen2.5-32b.json",
    )
    parser.add_argument("--projection", default="down")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=None)
    parser.add_argument("--splits", type=parse_csv_ints, default=parse_csv_ints("1,2,4,8,16"))
    parser.add_argument(
        "--variant",
        action="append",
        choices=["torch-full", "tp-sequential", "tp-multistream"],
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--l2-flush-mib", type=int, default=96)
    parser.add_argument(
        "--compile-mode",
        default="eager",
        help="torch.compile mode. Use 'default' for Inductor, 'eager' for Dynamo eager backend, or 'none'.",
    )
    parser.add_argument("--split-align", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "tp_splitk_probe.json",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=OUT_DIR / "tp_splitk_probe.png",
    )
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["measurements"], indent=2))


if __name__ == "__main__":
    main()
