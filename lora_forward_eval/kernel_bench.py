"""Kernel-level BF16, Marlin INT4, and Marlin+BF16-LoRA benchmark.

The benchmark intentionally treats torch.compile and CUDA Graph capture as hard
requirements. A provider that cannot be compiled and captured records a failure
instead of falling back to eager execution.

The INT4 baseline uses SGLang's GPTQ/Marlin W4A16 GEMM path with bf16
activations. This is the fast serving-kernel analogue for the QeRL comparison,
not bitsandbytes NF4.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.lora.triton_ops.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.srt.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.test_marlin_utils import marlin_quantize


DTYPE = torch.bfloat16
RANK = 16
GROUP_SIZE = 128
QUANT_TYPE = scalar_types.uint4b8


@torch.library.custom_op("lora_forward_eval::gptq_marlin_gemm", mutates_args=())
def compiled_gptq_marlin_gemm(
    a: torch.Tensor,
    q_w: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """Compile-friendly opaque wrapper around SGLang's JIT Marlin GEMM."""
    return gptq_marlin_gemm(
        a,
        None,
        q_w,
        scales,
        None,
        None,
        g_idx,
        sort_indices,
        workspace,
        QUANT_TYPE,
        a.shape[0],
        size_n,
        size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    )


@compiled_gptq_marlin_gemm.register_fake
def _compiled_gptq_marlin_gemm_fake(
    a: torch.Tensor,
    q_w: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    del q_w, scales, g_idx, sort_indices, workspace, size_k
    return torch.empty((a.shape[0], size_n), dtype=a.dtype, device=a.device)


@dataclass(frozen=True)
class ModelShape:
    name: str
    layers: int
    hidden: int
    kv_out: int
    intermediate: int


QWEN25_SHAPES = {
    "qwen2.5-14b": ModelShape(
        name="qwen2.5-14b",
        layers=48,
        hidden=5120,
        kv_out=1024,
        intermediate=13824,
    ),
    "qwen2.5-32b": ModelShape(
        name="qwen2.5-32b",
        layers=64,
        hidden=5120,
        kv_out=1024,
        intermediate=27648,
    ),
}


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    input_dim: int
    output_dims: tuple[int, ...]
    lora_b_kind: str

    @property
    def output_dim(self) -> int:
        return sum(self.output_dims)

    @property
    def stack_num(self) -> int:
        return len(self.output_dims)


@dataclass
class BenchResult:
    model: str
    projection: str
    provider: str
    tokens: int
    compiled: bool
    cuda_graph: bool
    hot_activation: bool = False
    median_us: float | None = None
    p20_us: float | None = None
    p80_us: float | None = None
    error: str | None = None


def projection_specs(shape: ModelShape) -> list[ProjectionSpec]:
    return [
        ProjectionSpec(
            "qkv",
            shape.hidden,
            (shape.hidden, shape.kv_out, shape.kv_out),
            "qkv",
        ),
        ProjectionSpec("out", shape.hidden, (shape.hidden,), "single"),
        ProjectionSpec(
            "gate_up",
            shape.hidden,
            (shape.intermediate, shape.intermediate),
            "gate_up",
        ),
        ProjectionSpec("down", shape.intermediate, (shape.hidden,), "single"),
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def flush_l2_buffer(device: torch.device, mib: int) -> torch.Tensor:
    return torch.empty(mib * 1024 * 1024, dtype=torch.int8, device=device)


def flush_l2(buf: torch.Tensor) -> None:
    # A write over a large buffer is a simple, explicit way to evict useful L2
    # lines before the measured replay. It is intentionally outside the graph.
    buf.zero_()


def quantize_marlin_weight(size_k: int, size_n: int, device: torch.device):
    w = torch.randn((size_k, size_n), dtype=DTYPE, device=device)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, QUANT_TYPE, GROUP_SIZE, False
    )
    return q_w, scales, g_idx, sort_indices


def make_lora_batch_info(tokens: int, device: torch.device) -> LoRABatchInfo:
    seg_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
    seg_indptr = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    weight_indices = torch.tensor([0], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([RANK], dtype=torch.int32, device=device)
    scalings = torch.tensor([1.0], dtype=torch.float32, device=device)
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=1,
        num_segments=1,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=tokens,
        seg_lens=seg_lens,
        permutation=None,
        expected_tokens=tokens,
        has_active_lora=True,
    )


def compile_fullgraph(fn: Callable[[], Any]) -> Callable[[], Any]:
    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    compiled()
    torch.cuda.synchronize()
    return compiled


def capture_graph(fn: Callable[[], Any]) -> Callable[[], Any]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def timed_replay(
    replay: Callable[[], Any],
    l2_buf: torch.Tensor,
    warmup: int,
    iters: int,
    pre_replay_fn: Callable[[], Any] | None = None,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        flush_l2(l2_buf)
        if pre_replay_fn is not None:
            pre_replay_fn()
        replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        flush_l2(l2_buf)
        if pre_replay_fn is not None:
            pre_replay_fn()  # re-warms activation into L2; weights stay cold
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)

    times.sort()
    median = statistics.median(times)
    p20 = times[max(0, math.floor(0.20 * (len(times) - 1)))]
    p80 = times[min(len(times) - 1, math.ceil(0.80 * (len(times) - 1)))]
    return median, p20, p80


def make_bf16_base_fn(spec: ProjectionSpec, x: torch.Tensor) -> Callable[[], Any]:
    weights = [
        torch.randn((spec.input_dim, out_dim), dtype=DTYPE, device=x.device)
        for out_dim in spec.output_dims
    ]

    def fn():
        outs = [torch.matmul(x, weight) for weight in weights]
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs, dim=-1)

    return fn


def make_marlin_base_fn(spec: ProjectionSpec, x: torch.Tensor) -> Callable[[], Any]:
    workspace = marlin_make_workspace(x.device)
    weights = [
        quantize_marlin_weight(spec.input_dim, out_dim, x.device)
        for out_dim in spec.output_dims
    ]

    def fn():
        outs = []
        for (q_w, scales, g_idx, sort_indices), out_dim in zip(
            weights, spec.output_dims
        ):
            outs.append(
                compiled_gptq_marlin_gemm(
                    x,
                    q_w,
                    scales,
                    g_idx,
                    sort_indices,
                    workspace,
                    out_dim,
                    spec.input_dim,
                )
            )
        return outs[0] if len(outs) == 1 else outs

    return fn


def make_qlora_fn(spec: ProjectionSpec, x: torch.Tensor) -> Callable[[], Any]:
    """INT4 base GEMM immediately followed by BF16 LoRA patch in one graph."""
    workspace = marlin_make_workspace(x.device)
    int4_weights = [
        quantize_marlin_weight(spec.input_dim, out_dim, x.device)
        for out_dim in spec.output_dims
    ]
    batch_info = make_lora_batch_info(x.shape[0], x.device)
    a_weight = torch.randn(
        (1, spec.stack_num * RANK, spec.input_dim), dtype=DTYPE, device=x.device
    ).contiguous()
    b_weight = torch.randn(
        (1, spec.output_dim, RANK), dtype=DTYPE, device=x.device
    ).contiguous()
    lora_delta = torch.empty((x.shape[0], spec.output_dim), dtype=DTYPE, device=x.device)

    if spec.lora_b_kind == "qkv":
        offsets = [0]
        for out_dim in spec.output_dims:
            offsets.append(offsets[-1] + out_dim)
        output_offset = torch.tensor(offsets, dtype=torch.int32, device=x.device)
        max_qkv_out_dim = max(spec.output_dims)

        def fn():
            outs = []
            for (q_w, scales, g_idx, sort_indices), out_dim in zip(
                int4_weights, spec.output_dims
            ):
                outs.append(
                    compiled_gptq_marlin_gemm(
                        x, q_w, scales, g_idx, sort_indices, workspace, out_dim, spec.input_dim
                    )
                )
            int4_out = torch.cat(outs, dim=-1)
            mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
            qkv_lora_b_fwd(mid, b_weight, batch_info, output_offset, max_qkv_out_dim, lora_delta)
            return int4_out.add_(lora_delta)

        return fn

    if spec.lora_b_kind == "gate_up":

        def fn():
            outs = []
            for (q_w, scales, g_idx, sort_indices), out_dim in zip(
                int4_weights, spec.output_dims
            ):
                outs.append(
                    compiled_gptq_marlin_gemm(
                        x, q_w, scales, g_idx, sort_indices, workspace, out_dim, spec.input_dim
                    )
                )
            int4_out = torch.cat(outs, dim=-1)
            mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
            gate_up_lora_b_fwd(mid, b_weight, batch_info, spec.output_dims[0], lora_delta)
            return int4_out.add_(lora_delta)

        return fn

    # single projection (out, down)
    (q_w, scales, g_idx, sort_indices) = int4_weights[0]
    out_dim = spec.output_dims[0]

    def fn():
        int4_out = compiled_gptq_marlin_gemm(
            x, q_w, scales, g_idx, sort_indices, workspace, out_dim, spec.input_dim
        )
        mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=1)
        sgemm_lora_b_fwd(mid, b_weight, batch_info, lora_delta)
        return int4_out.add_(lora_delta)

    return fn


def make_qlora_parallel_fn(spec: ProjectionSpec, x: torch.Tensor) -> Callable[[], Any]:
    """INT4 on main stream, lora_a+b on a side stream — both start from x concurrently."""
    workspace = marlin_make_workspace(x.device)
    int4_weights = [
        quantize_marlin_weight(spec.input_dim, out_dim, x.device)
        for out_dim in spec.output_dims
    ]
    batch_info = make_lora_batch_info(x.shape[0], x.device)
    a_weight = torch.randn(
        (1, spec.stack_num * RANK, spec.input_dim), dtype=DTYPE, device=x.device
    ).contiguous()
    b_weight = torch.randn(
        (1, spec.output_dim, RANK), dtype=DTYPE, device=x.device
    ).contiguous()
    lora_delta = torch.empty((x.shape[0], spec.output_dim), dtype=DTYPE, device=x.device)
    stream_lora = torch.cuda.Stream()

    if spec.lora_b_kind == "qkv":
        offsets = [0]
        for out_dim in spec.output_dims:
            offsets.append(offsets[-1] + out_dim)
        output_offset = torch.tensor(offsets, dtype=torch.int32, device=x.device)
        max_qkv_out_dim = max(spec.output_dims)

        def fn():
            stream_lora.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream_lora):
                mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
                qkv_lora_b_fwd(mid, b_weight, batch_info, output_offset, max_qkv_out_dim, lora_delta)
            outs = []
            for (q_w, scales, g_idx, sort_indices), out_dim in zip(int4_weights, spec.output_dims):
                outs.append(compiled_gptq_marlin_gemm(
                    x, q_w, scales, g_idx, sort_indices, workspace, out_dim, spec.input_dim
                ))
            int4_out = torch.cat(outs, dim=-1)
            torch.cuda.current_stream().wait_stream(stream_lora)
            return int4_out.add_(lora_delta)

        return fn

    if spec.lora_b_kind == "gate_up":

        def fn():
            stream_lora.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream_lora):
                mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
                gate_up_lora_b_fwd(mid, b_weight, batch_info, spec.output_dims[0], lora_delta)
            outs = []
            for (q_w, scales, g_idx, sort_indices), out_dim in zip(int4_weights, spec.output_dims):
                outs.append(compiled_gptq_marlin_gemm(
                    x, q_w, scales, g_idx, sort_indices, workspace, out_dim, spec.input_dim
                ))
            int4_out = torch.cat(outs, dim=-1)
            torch.cuda.current_stream().wait_stream(stream_lora)
            return int4_out.add_(lora_delta)

        return fn

    # single projection (out, down)
    (q_w, scales, g_idx, sort_indices) = int4_weights[0]
    out_dim = spec.output_dims[0]

    def fn():
        stream_lora.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream_lora):
            mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=1)
            sgemm_lora_b_fwd(mid, b_weight, batch_info, lora_delta)
        int4_out = compiled_gptq_marlin_gemm(
            x, q_w, scales, g_idx, sort_indices, workspace, out_dim, spec.input_dim
        )
        torch.cuda.current_stream().wait_stream(stream_lora)
        return int4_out.add_(lora_delta)

    return fn


def bench_qlora_parallel(
    *,
    model: str,
    spec: ProjectionSpec,
    x: torch.Tensor,
    l2_buf: torch.Tensor,
    warmup: int,
    iters: int,
) -> BenchResult:
    """Benchmark qlora with INT4 and lora_a/b overlapped on two CUDA streams.

    Skips torch.compile because stream context managers are not Dynamo-traceable,
    but still captures a multi-stream CUDA graph for clean GPU timing.
    """
    try:
        fn = make_qlora_parallel_fn(spec, x)
        # Warm up Triton JIT and Marlin extension before graph capture.
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        # Capture multi-stream CUDA graph (cudaStreamCaptureModeGlobal captures
        # all streams that fork from the capturing stream).
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        median, p20, p80 = timed_replay(graph.replay, l2_buf, warmup=warmup, iters=iters)
        return BenchResult(
            model=model,
            projection=spec.name,
            provider="qlora_parallel_streams",
            tokens=-1,
            compiled=False,
            cuda_graph=True,
            median_us=median,
            p20_us=p20,
            p80_us=p80,
        )
    except Exception:
        return BenchResult(
            model=model,
            projection=spec.name,
            provider="qlora_parallel_streams",
            tokens=-1,
            compiled=False,
            cuda_graph=False,
            error=traceback.format_exc(),
        )


def make_lora_patch_fn(spec: ProjectionSpec, x: torch.Tensor) -> Callable[[], Any]:
    batch_info = make_lora_batch_info(x.shape[0], x.device)
    a_weight = torch.randn(
        (1, spec.stack_num * RANK, spec.input_dim),
        dtype=DTYPE,
        device=x.device,
    ).contiguous()
    b_weight = torch.randn(
        (1, spec.output_dim, RANK),
        dtype=DTYPE,
        device=x.device,
    ).contiguous()
    base_output = torch.empty((x.shape[0], spec.output_dim), dtype=DTYPE, device=x.device)

    if spec.lora_b_kind == "qkv":
        offsets = [0]
        for out_dim in spec.output_dims:
            offsets.append(offsets[-1] + out_dim)
        output_offset = torch.tensor(offsets, dtype=torch.int32, device=x.device)
        max_qkv_out_dim = max(spec.output_dims)

        def fn():
            mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
            return qkv_lora_b_fwd(
                mid,
                b_weight,
                batch_info,
                output_offset,
                max_qkv_out_dim,
                base_output,
            )

        return fn

    if spec.lora_b_kind == "gate_up":

        def fn():
            mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
            return gate_up_lora_b_fwd(
                mid,
                b_weight,
                batch_info,
                spec.output_dims[0],
                base_output,
            )

        return fn

    def fn():
        mid = sgemm_lora_a_fwd(x, a_weight, batch_info, stack_num=spec.stack_num)
        return sgemm_lora_b_fwd(mid, b_weight, batch_info, base_output)

    return fn


def bench_provider(
    *,
    model: str,
    spec: ProjectionSpec,
    provider: str,
    fn_factory: Callable[[], Callable[[], Any]],
    l2_buf: torch.Tensor,
    warmup: int,
    iters: int,
    pre_replay_fn: Callable[[], Any] | None = None,
) -> BenchResult:
    try:
        fn = fn_factory()
        # Warm extension/JIT compilation before Dynamo compile and graph capture.
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        compiled = compile_fullgraph(fn)
        replay = capture_graph(compiled)
        median, p20, p80 = timed_replay(
            replay, l2_buf, warmup=warmup, iters=iters, pre_replay_fn=pre_replay_fn
        )
        return BenchResult(
            model=model,
            projection=spec.name,
            provider=provider,
            tokens=next(iter(fn.__closure__ or []), None).cell_contents.shape[0]
            if fn.__closure__
            and hasattr(next(iter(fn.__closure__)).cell_contents, "shape")
            else -1,
            compiled=True,
            cuda_graph=True,
            hot_activation=pre_replay_fn is not None,
            median_us=median,
            p20_us=p20,
            p80_us=p80,
        )
    except Exception:
        return BenchResult(
            model=model,
            projection=spec.name,
            provider=provider,
            tokens=-1,
            compiled=False,
            cuda_graph=False,
            hot_activation=pre_replay_fn is not None,
            error=traceback.format_exc(),
        )


def bench_one(
    shape: ModelShape,
    spec: ProjectionSpec,
    tokens: int,
    l2_buf: torch.Tensor,
    warmup: int,
    iters: int,
    hot_activation: bool = False,
) -> list[BenchResult]:
    x = torch.randn((tokens, spec.input_dim), dtype=DTYPE, device=l2_buf.device)
    pre_replay_fn = (lambda: x.sum()) if hot_activation else None

    rows = [
        bench_provider(
            model=shape.name,
            spec=spec,
            provider="bf16_full",
            fn_factory=lambda: make_bf16_base_fn(spec, x),
            l2_buf=l2_buf,
            warmup=warmup,
            iters=iters,
            pre_replay_fn=pre_replay_fn,
        ),
        bench_provider(
            model=shape.name,
            spec=spec,
            provider="marlin_int4_full",
            fn_factory=lambda: make_marlin_base_fn(spec, x),
            l2_buf=l2_buf,
            warmup=warmup,
            iters=iters,
            pre_replay_fn=pre_replay_fn,
        ),
        bench_provider(
            model=shape.name,
            spec=spec,
            provider="bf16_lora_patch",
            fn_factory=lambda: make_lora_patch_fn(spec, x),
            l2_buf=l2_buf,
            warmup=warmup,
            iters=iters,
            pre_replay_fn=pre_replay_fn,
        ),
        bench_provider(
            model=shape.name,
            spec=spec,
            provider="qlora_marlin_bf16",
            fn_factory=lambda: make_qlora_fn(spec, x),
            l2_buf=l2_buf,
            warmup=warmup,
            iters=iters,
            pre_replay_fn=pre_replay_fn,
        ),
        bench_qlora_parallel(
            model=shape.name,
            spec=spec,
            x=x,
            l2_buf=l2_buf,
            warmup=warmup,
            iters=iters,
        ),
    ]
    for row in rows:
        row.tokens = tokens
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5-14b", "qwen2.5-32b"],
        choices=sorted(QWEN25_SHAPES),
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        default=["qkv", "out", "gate_up", "down"],
        choices=["qkv", "out", "gate_up", "down"],
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--hot-activation",
        action="store_true",
        help="Pre-warm the activation tensor into L2 before each timed replay "
        "(simulates serving where activations are hot from the previous layer).",
    )
    parser.add_argument("--l2-flush-mib", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default=".codex-reports/lora_forward_eval/kernel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l2_buf = flush_l2_buffer(device, args.l2_flush_mib)

    metadata = {
        "created_unix": time.time(),
        "dtype": str(DTYPE),
        "rank": RANK,
        "group_size": GROUP_SIZE,
        "quant_type": str(QUANT_TYPE),
        "tokens": args.tokens,
        "warmup": args.warmup,
        "iters": args.iters,
        "l2_flush_mib": args.l2_flush_mib,
        "hot_activation": args.hot_activation,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(device),
        "models": [asdict(QWEN25_SHAPES[name]) for name in args.models],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    results: list[BenchResult] = []
    for model_name in args.models:
        shape = QWEN25_SHAPES[model_name]
        specs = [s for s in projection_specs(shape) if s.name in args.projections]
        for spec in specs:
            print(f"Benchmarking {shape.name} {spec.name} tokens={args.tokens}")
            results.extend(
                bench_one(
                    shape,
                    spec,
                    tokens=args.tokens,
                    l2_buf=l2_buf,
                    warmup=args.warmup,
                    iters=args.iters,
                    hot_activation=args.hot_activation,
                )
            )
            write_jsonl(out_dir / "kernel_results.jsonl", [asdict(r) for r in results])

    rows = [asdict(r) for r in results]
    write_jsonl(out_dir / "kernel_results.jsonl", rows)
    failures = [row for row in rows if row["error"]]
    summary = {
        "num_results": len(rows),
        "num_failures": len(failures),
        "failures": [
            {
                "model": row["model"],
                "projection": row["projection"],
                "provider": row["provider"],
                "error_tail": row["error"][-2000:] if row["error"] else None,
            }
            for row in failures
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        print(json.dumps(summary, indent=2))
        sys.exit(2)
    print(f"Wrote kernel results to {out_dir}")


if __name__ == "__main__":
    main()
