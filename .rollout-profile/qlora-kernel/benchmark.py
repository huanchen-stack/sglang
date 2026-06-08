#!/usr/bin/env python3
"""Benchmark SGLang-style QLoRA kernel latency.

This benchmark simulates the SGLang runtime LoRA forwarding path for one
projection group at a time:

* base projection: bf16 dense or int4 GPTQ Marlin
* LoRA patch: SGLang default ChunkedSGMV (`csgmv`) backend or plain BF16 Torch matmul
* QLoRA sequential: Marlin base, then LoRA accumulated into base output
* QLoRA two-stream: LoRA patch on a side stream while Marlin base runs

For Qwen-style layers, the faithful SGLang projection groups are `qkv`, `o`,
`gate_up`, and `down`. `qkv` and `gate_up` are fused in the LoRA wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[1]
DEFAULT_DATA = OUT_DIR / "qlora_kernel_perf.json"
DEFAULT_FAKE_DATA = OUT_DIR / "qlora_kernel_perf_fake.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int = 4096
    output_size: int = 11008
    lora_rank: int = 16
    lora_alpha: float = 16.0
    dtype: str = "bf16"
    quant_type: str = "uint4b8"
    group_size: int = 128
    act_order: bool = False
    warmup_iters: int = 10
    measure_iters: int = 30
    l2_flush_mib: int = 96
    max_loras_per_batch: int = 8
    max_lora_chunk_size: int = 16
    activation_prewarm: bool = True
    torch_compile: bool = True
    cuda_graph: bool = True
    compile_mode: str = "default"
    two_stream_reserve_sms: int = 1


@dataclass(frozen=True)
class ProjectionConfig:
    name: str
    kind: str
    in_features: int
    out_features: int
    slice_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    model: str
    layers: int | None
    hidden_size: int
    kv_out: int
    intermediate_size: int
    projections: tuple[ProjectionConfig, ...]
    lora_rank: int | None = None
    lora_alpha: float | None = None
    dtype: str | None = None
    quant_type: str | None = None
    group_size: int | None = None


@dataclass(frozen=True)
class Measurement:
    model: str | None
    projection: str | None
    projection_kind: str | None
    in_features: int | None
    out_features: int | None
    token_rows: int
    scheme: str
    latency_us: float | None
    median_us: float | None
    p20_us: float | None
    p80_us: float | None
    min_us: float | None
    max_us: float | None
    warmup_iters: int
    measure_iters: int
    compiled: bool
    cuda_graph: bool
    l2_flush_mib: int
    activation_prewarm: bool
    fake: bool
    error: str | None
    note: str


SCHEME_ORDER = [
    "bf16 dense base",
    "int4 Marlin base",
    "SGLang csgmv LoRA patch",
    "SGLang triton LoRA patch",
    "Torch matmul LoRA patch",
    "bf16 dense + csgmv sequential",
    "bf16 dense + triton sequential",
    "bf16 dense + torch matmul sequential",
    "bf16 dense + torch matmul two-stream",
    "SGLang QLoRA csgmv sequential",
    "SGLang QLoRA csgmv two-stream",
    "SGLang QLoRA triton sequential",
    "SGLang QLoRA triton two-stream",
    "Torch QLoRA matmul sequential",
    "Torch QLoRA matmul two-stream",
]


def add_repo_python_to_path() -> None:
    python_dir = REPO_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def parse_token_rows(value: str) -> list[int]:
    rows = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not rows or any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError(
            "tokens must be a comma-separated list of positive integers"
        )
    return rows


def dtype_from_name(name: str):
    import torch

    normalized = name.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}; expected bf16 or fp16")


def projection_from_dict(raw: dict) -> ProjectionConfig:
    slice_sizes = tuple(int(x) for x in raw.get("slice_sizes", [raw["out_features"]]))
    out_features = int(raw["out_features"])
    if sum(slice_sizes) != out_features:
        raise ValueError(
            f"{raw['name']} slice_sizes sum to {sum(slice_sizes)}, expected {out_features}"
        )
    return ProjectionConfig(
        name=str(raw["name"]),
        kind=str(raw.get("kind", "simple")),
        in_features=int(raw["in_features"]),
        out_features=out_features,
        slice_sizes=slice_sizes,
    )


def load_model_config(path: Path) -> ModelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ModelConfig(
        model=str(raw["model"]),
        layers=raw.get("layers"),
        hidden_size=int(raw["hidden_size"]),
        kv_out=int(raw["kv_out"]),
        intermediate_size=int(raw["intermediate_size"]),
        projections=tuple(projection_from_dict(item) for item in raw["projections"]),
        lora_rank=raw.get("lora_rank"),
        lora_alpha=raw.get("lora_alpha"),
        dtype=raw.get("dtype"),
        quant_type=raw.get("quant_type"),
        group_size=raw.get("group_size"),
    )


def make_single_projection_model(config: BenchmarkConfig) -> ModelConfig:
    return ModelConfig(
        model="single-linear",
        layers=None,
        hidden_size=config.hidden_size,
        kv_out=config.output_size,
        intermediate_size=config.output_size,
        projections=(
            ProjectionConfig(
                "single_linear",
                "simple",
                config.hidden_size,
                config.output_size,
                (config.output_size,),
            ),
        ),
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        dtype=config.dtype,
        quant_type=config.quant_type,
        group_size=config.group_size,
    )


def filter_projections(
    model_config: ModelConfig, projection_filter: set[str] | None
) -> tuple[ProjectionConfig, ...]:
    if projection_filter is None:
        return model_config.projections
    selected = tuple(
        projection
        for projection in model_config.projections
        if projection.name in projection_filter
    )
    missing = projection_filter.difference(
        {projection.name for projection in model_config.projections}
    )
    if missing:
        raise ValueError(
            f"{model_config.model} config does not define projections: {sorted(missing)}"
        )
    return selected


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def validate_shape(in_features: int, out_features: int, config: BenchmarkConfig) -> None:
    if in_features % 128 != 0:
        raise ValueError(f"in_features={in_features} must be divisible by 128")
    if out_features % 64 != 0:
        raise ValueError(f"out_features={out_features} must be divisible by 64")
    if config.group_size != -1 and in_features % config.group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={config.group_size}"
        )
    if config.lora_rank <= 0:
        raise ValueError("--rank must be positive")


def compile_callable(fn: Callable[[], object], mode: str) -> Callable[[], object]:
    import torch

    if mode == "default":
        return torch.compile(fn, fullgraph=False)
    return torch.compile(fn, fullgraph=False, mode=mode)


def compile_eager_callable(fn: Callable[[], object]) -> Callable[[], object]:
    """Use torch.compile's Dynamo wrapper without Inductor around custom kernels.

    SGLang csgmv already launches custom Triton kernels. Inductor compilation of
    that Python/Triton path is not the runtime behavior and can spend minutes in
    compile workers. The eager backend still exercises Dynamo wrapping while
    leaving the backend kernels intact for CUDA Graph timing.
    """

    import torch

    return torch.compile(fn, fullgraph=False, backend="eager")


def call_with_marlin_sm_reserve(fn: Callable[[], object], reserve_sms: int):
    previous = os.environ.get("SGLANG_MARLIN_RESERVE_SMS")
    try:
        if reserve_sms > 0:
            os.environ["SGLANG_MARLIN_RESERVE_SMS"] = str(reserve_sms)
        else:
            os.environ.pop("SGLANG_MARLIN_RESERVE_SMS", None)
        return fn()
    finally:
        if previous is None:
            os.environ.pop("SGLANG_MARLIN_RESERVE_SMS", None)
        else:
            os.environ["SGLANG_MARLIN_RESERVE_SMS"] = previous


def call_without_marlin_sm_reserve(fn: Callable[[], object]):
    return call_with_marlin_sm_reserve(fn, 0)


def prewarm_activation(x, touch_fn: Callable[[], object]) -> None:
    import torch

    with torch.no_grad():
        _ = x.sum()
        touch_fn()
        torch.cuda.synchronize()


def capture_cuda_graph(fn: Callable[[], object]):
    import torch

    graph = torch.cuda.CUDAGraph()
    holder: dict[str, object] = {}
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        holder["out"] = fn()
    return graph, holder


def replay_and_measure(
    graph,
    *,
    flush_buffer,
    warmup_iters: int,
    measure_iters: int,
) -> list[float]:
    import torch

    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    timings: list[float] = []
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


def make_csgmv_batch_info(num_tokens: int, config: BenchmarkConfig, device):
    import torch
    from sglang.srt.lora.utils import LoRABatchInfo

    chunk = config.max_lora_chunk_size
    num_segments = (num_tokens + chunk - 1) // chunk
    seg_lens = [chunk] * num_segments
    seg_lens[-1] = num_tokens - chunk * (num_segments - 1)

    seg_indptr_cpu = torch.empty(num_segments + 1, dtype=torch.int32)
    seg_indptr_cpu[0] = 0
    seg_indptr_cpu[1:] = torch.cumsum(torch.tensor(seg_lens, dtype=torch.int32), dim=0)

    lora_ranks = torch.zeros(config.max_loras_per_batch, dtype=torch.int32, device=device)
    lora_ranks[0] = config.lora_rank
    scalings = torch.zeros(config.max_loras_per_batch, dtype=torch.float32, device=device)
    scalings[0] = config.lora_alpha / config.lora_rank

    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=num_tokens,
        num_segments=num_segments,
        seg_indptr=seg_indptr_cpu.to(device=device),
        weight_indices=torch.zeros(num_segments, dtype=torch.int32, device=device),
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=chunk,
        seg_lens=torch.tensor(seg_lens, dtype=torch.int32, device=device),
        permutation=torch.arange(num_tokens, dtype=torch.int32, device=device),
        expected_tokens=num_tokens,
        has_active_lora=True,
        req_seg_indptr=torch.arange(num_tokens + 1, dtype=torch.int32, device=device),
        req_weight_indices=torch.zeros(num_tokens, dtype=torch.int32, device=device),
    )


def make_triton_sgemm_batch_info(num_tokens: int, config: BenchmarkConfig, device):
    import torch
    from sglang.srt.lora.utils import LoRABatchInfo

    lora_ranks = torch.zeros(config.max_loras_per_batch, dtype=torch.int32, device=device)
    lora_ranks[0] = config.lora_rank
    scalings = torch.zeros(config.max_loras_per_batch, dtype=torch.float32, device=device)
    scalings[0] = config.lora_alpha / config.lora_rank

    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=num_tokens,
        seg_lens=torch.tensor([num_tokens], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=num_tokens,
        has_active_lora=True,
        req_seg_indptr=torch.arange(num_tokens + 1, dtype=torch.int32, device=device),
        req_weight_indices=torch.zeros(num_tokens, dtype=torch.int32, device=device),
    )


class SGLangQLoRASim:
    """Small forward helper that directly uses SGLang's default csgmv LoRA backend."""

    def __init__(
        self,
        *,
        backend,
        projection: ProjectionConfig,
        dtype,
        device,
        rank: int,
    ):
        import torch

        self.backend = backend
        self.projection = projection
        self.device = device
        self.dtype = dtype
        self.rank = rank
        self.output_offset = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(projection.slice_sizes), dim=0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        self.max_slice_size = max(projection.slice_sizes)
        n_slices = len(projection.slice_sizes)
        self.lora_a = torch.randn(
            (1, n_slices * rank, projection.in_features),
            device=device,
            dtype=dtype,
        ).contiguous()
        self.lora_b = torch.randn(
            (1, projection.out_features, rank), device=device, dtype=dtype
        ).contiguous()

    def patch(self, x):
        if self.projection.kind == "qkv":
            return self.backend.run_qkv_lora(
                x=x,
                qkv_lora_a=self.lora_a,
                qkv_lora_b=self.lora_b,
                output_offset=self.output_offset,
                max_qkv_out_dim=self.max_slice_size,
                base_output=None,
                n_slices=len(self.projection.slice_sizes),
            )
        if self.projection.kind == "gate_up":
            assert len(self.projection.slice_sizes) == 2
            return self.backend.run_gate_up_lora(
                x=x,
                gate_up_lora_a=self.lora_a,
                gate_up_lora_b=self.lora_b,
                output_offset=self.output_offset,
                base_output=None,
            )

        lora_a_output = self.backend.run_lora_a_sgemm(x, self.lora_a)
        return self.backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lora_b,
            output_offset=self.output_offset,
            base_output=None,
        )

    def apply_to_base(self, base_output, x):
        if self.projection.kind == "qkv":
            return self.backend.run_qkv_lora(
                x=x,
                qkv_lora_a=self.lora_a,
                qkv_lora_b=self.lora_b,
                output_offset=self.output_offset,
                max_qkv_out_dim=self.max_slice_size,
                base_output=base_output,
                n_slices=len(self.projection.slice_sizes),
            )
        if self.projection.kind == "gate_up":
            return self.backend.run_gate_up_lora(
                x=x,
                gate_up_lora_a=self.lora_a,
                gate_up_lora_b=self.lora_b,
                output_offset=self.output_offset,
                base_output=base_output,
            )

        lora_a_output = self.backend.run_lora_a_sgemm(x, self.lora_a)
        return self.backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lora_b,
            output_offset=self.output_offset,
            base_output=base_output,
        )


class TorchMatmulLoRASim:
    """Plain BF16 matmul LoRA patch using the same fake weights as SGLangQLoRASim."""

    def __init__(
        self,
        *,
        projection: ProjectionConfig,
        lora_a,
        lora_b,
        rank: int,
        scaling: float,
    ):
        self.projection = projection
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.rank = rank
        self.scaling = scaling
        self.output_offsets = [0]
        for slice_size in projection.slice_sizes:
            self.output_offsets.append(self.output_offsets[-1] + slice_size)

    def _patch_only(self, x):
        import torch

        a_out = torch.matmul(x, self.lora_a[0].t())
        parts = []
        for idx, slice_size in enumerate(self.projection.slice_sizes):
            r0 = idx * self.rank
            r1 = r0 + self.rank
            o0 = self.output_offsets[idx]
            o1 = o0 + slice_size
            parts.append(torch.matmul(a_out[:, r0:r1], self.lora_b[0, o0:o1, :].t()))
        if len(parts) == 1:
            return parts[0] * self.scaling
        return torch.cat(parts, dim=-1) * self.scaling

    def patch(self, x):
        return self._patch_only(x)

    def apply_to_base(self, base_output, x):
        return base_output + self._patch_only(x)


def error_measurement(
    *,
    model: str,
    projection: ProjectionConfig,
    token_rows: int,
    scheme: str,
    config: BenchmarkConfig,
    error: BaseException,
    note: str,
) -> Measurement:
    return Measurement(
        model=model,
        projection=projection.name,
        projection_kind=projection.kind,
        in_features=projection.in_features,
        out_features=projection.out_features,
        token_rows=token_rows,
        scheme=scheme,
        latency_us=None,
        median_us=None,
        p20_us=None,
        p80_us=None,
        min_us=None,
        max_us=None,
        warmup_iters=config.warmup_iters,
        measure_iters=config.measure_iters,
        compiled=True,
        cuda_graph=True,
        l2_flush_mib=config.l2_flush_mib,
        activation_prewarm=config.activation_prewarm,
        fake=False,
        error=repr(error),
        note=note,
    )


def make_real_measurements(
    config: BenchmarkConfig,
    token_rows: Iterable[int],
    model_configs: Iterable[ModelConfig],
    projection_filter: set[str] | None,
    scheme_filter: set[str] | None,
    cuda_profiler_range: bool,
) -> list[Measurement]:
    add_repo_python_to_path()

    import torch
    from sgl_kernel.scalar_type import scalar_types

    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    from sglang.test.test_marlin_utils import marlin_quantize

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    workspace = marlin_make_workspace(device)
    backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=config.max_loras_per_batch,
        device=device,
        server_args=SimpleNamespace(max_lora_chunk_size=config.max_lora_chunk_size),
    )
    triton_backend = TritonLoRABackend(
        max_loras_per_batch=config.max_loras_per_batch,
        device=device,
    )

    flush_buffer = None
    if config.l2_flush_mib > 0:
        elements = (
            config.l2_flush_mib
            * 1024
            * 1024
            // torch.empty((), dtype=torch.int32).element_size()
        )
        flush_buffer = torch.empty((elements,), device=device, dtype=torch.int32)

    notes = {
        "bf16 dense base": "Dense bf16 torch.matmul base projection.",
        "int4 Marlin base": "SGLang GPTQ Marlin GEMM base projection only.",
        "SGLang csgmv LoRA patch": "SGLang default ChunkedSGMV LoRA patch only.",
        "SGLang triton LoRA patch": "SGLang Triton segmented-GEMM LoRA patch only.",
        "Torch matmul LoRA patch": "Plain bf16 torch.matmul LoRA patch only using one active adapter.",
        "bf16 dense + csgmv sequential": "Dense bf16 base followed by SGLang csgmv LoRA accumulated into base output.",
        "bf16 dense + triton sequential": "Dense bf16 base followed by SGLang Triton segmented-GEMM LoRA accumulated into base output.",
        "bf16 dense + torch matmul sequential": "Dense bf16 base followed by plain bf16 torch.matmul LoRA patch.",
        "bf16 dense + torch matmul two-stream": "Plain bf16 torch.matmul LoRA patch on a side stream while dense bf16 base runs.",
        "SGLang QLoRA csgmv sequential": "Marlin base followed by SGLang csgmv LoRA accumulated into base output.",
        "SGLang QLoRA csgmv two-stream": "SGLang csgmv LoRA patch on a side stream while Marlin base runs.",
        "SGLang QLoRA triton sequential": "Marlin base followed by SGLang Triton segmented-GEMM LoRA accumulated into base output.",
        "SGLang QLoRA triton two-stream": "SGLang Triton segmented-GEMM LoRA patch on a side stream while Marlin base runs.",
        "Torch QLoRA matmul sequential": "Marlin base followed by plain bf16 torch.matmul LoRA patch.",
        "Torch QLoRA matmul two-stream": "Plain bf16 torch.matmul LoRA patch on a side stream while Marlin base runs.",
    }
    rows: list[Measurement] = []

    for model_config in model_configs:
        effective = BenchmarkConfig(
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            lora_rank=int(model_config.lora_rank or config.lora_rank),
            lora_alpha=float(model_config.lora_alpha or config.lora_alpha),
            dtype=str(model_config.dtype or config.dtype),
            quant_type=str(model_config.quant_type or config.quant_type),
            group_size=int(model_config.group_size or config.group_size),
            act_order=config.act_order,
            warmup_iters=config.warmup_iters,
            measure_iters=config.measure_iters,
            l2_flush_mib=config.l2_flush_mib,
            max_loras_per_batch=config.max_loras_per_batch,
            max_lora_chunk_size=config.max_lora_chunk_size,
            activation_prewarm=config.activation_prewarm,
            torch_compile=config.torch_compile,
            cuda_graph=config.cuda_graph,
            compile_mode=config.compile_mode,
            two_stream_reserve_sms=config.two_stream_reserve_sms,
        )
        dtype = dtype_from_name(effective.dtype)
        quant_type = getattr(scalar_types, effective.quant_type)

        for projection in filter_projections(model_config, projection_filter):
            validate_shape(projection.in_features, projection.out_features, effective)
            print(
                f"Preparing {model_config.model}/{projection.name} "
                f"{projection.kind} ({projection.in_features}x{projection.out_features})...",
                file=sys.stderr,
                flush=True,
            )
            dense_weight = torch.randn(
                (projection.in_features, projection.out_features),
                device=device,
                dtype=dtype,
            ).contiguous()
            _, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
                dense_weight, quant_type, effective.group_size, effective.act_order
            )
            lora_sim = SGLangQLoRASim(
                backend=backend,
                projection=projection,
                dtype=dtype,
                device=device,
                rank=effective.lora_rank,
            )
            triton_lora_sim = SGLangQLoRASim(
                backend=triton_backend,
                projection=projection,
                dtype=dtype,
                device=device,
                rank=effective.lora_rank,
            )
            triton_lora_sim.lora_a = lora_sim.lora_a
            triton_lora_sim.lora_b = lora_sim.lora_b
            torch_lora_sim = TorchMatmulLoRASim(
                projection=projection,
                lora_a=lora_sim.lora_a,
                lora_b=lora_sim.lora_b,
                rank=effective.lora_rank,
                scaling=effective.lora_alpha / effective.lora_rank,
            )

            for token_rows_value in token_rows:
                print(
                    f"Token rows={token_rows_value} model={model_config.model} projection={projection.name}",
                    file=sys.stderr,
                    flush=True,
                )
                x = torch.randn(
                    (token_rows_value, projection.in_features),
                    device=device,
                    dtype=dtype,
                ).contiguous()
                backend.batch_info = make_csgmv_batch_info(
                    token_rows_value, effective, device
                )
                triton_backend.batch_info = make_triton_sgemm_batch_info(
                    token_rows_value, effective, device
                )

                def dense_base():
                    return torch.matmul(x, dense_weight)

                def marlin_base():
                    return gptq_marlin_gemm(
                        x,
                        None,
                        marlin_q_w,
                        marlin_s,
                        None,
                        None,
                        g_idx,
                        sort_indices,
                        workspace,
                        quant_type,
                        x.shape[0],
                        projection.out_features,
                        projection.in_features,
                        is_k_full=True,
                        use_atomic_add=False,
                        use_fp32_reduce=False,
                        is_zp_float=False,
                    )

                def marlin_base_regular():
                    return call_without_marlin_sm_reserve(marlin_base)

                def marlin_base_two_stream():
                    return call_with_marlin_sm_reserve(
                        marlin_base, effective.two_stream_reserve_sms
                    )

                def csgmv_patch():
                    return lora_sim.patch(x)

                def triton_patch():
                    return triton_lora_sim.patch(x)

                def torch_matmul_patch():
                    return torch_lora_sim.patch(x)

                compiled_dense_base = compile_callable(dense_base, effective.compile_mode)
                compiled_marlin_base = compile_callable(
                    marlin_base_regular, effective.compile_mode
                )
                compiled_marlin_base_two_stream = compile_callable(
                    marlin_base_two_stream, effective.compile_mode
                )
                compiled_csgmv_patch = csgmv_patch
                compiled_triton_patch = triton_patch
                compiled_torch_matmul_patch = compile_callable(
                    torch_matmul_patch, effective.compile_mode
                )

                def qlora_csgmv_sequential():
                    return lora_sim.apply_to_base(compiled_marlin_base(), x)

                def bf16_csgmv_sequential():
                    return lora_sim.apply_to_base(compiled_dense_base(), x)

                def bf16_triton_sequential():
                    return triton_lora_sim.apply_to_base(compiled_dense_base(), x)

                def bf16_torch_matmul_sequential():
                    return torch_lora_sim.apply_to_base(compiled_dense_base(), x)

                def bf16_torch_matmul_two_stream():
                    current_stream = torch.cuda.current_stream(device)
                    torch_matmul_side_stream.wait_stream(current_stream)
                    with torch.cuda.stream(torch_matmul_side_stream):
                        lora_out = compiled_torch_matmul_patch()
                    base_out = compiled_dense_base()
                    current_stream.wait_stream(torch_matmul_side_stream)
                    return base_out + lora_out

                def qlora_triton_sequential():
                    return triton_lora_sim.apply_to_base(compiled_marlin_base(), x)

                def qlora_torch_matmul_sequential():
                    return torch_lora_sim.apply_to_base(compiled_marlin_base(), x)

                csgmv_side_stream = torch.cuda.Stream(device=device)
                triton_side_stream = torch.cuda.Stream(device=device)
                torch_matmul_side_stream = torch.cuda.Stream(device=device)

                def qlora_csgmv_two_stream():
                    current_stream = torch.cuda.current_stream(device)
                    csgmv_side_stream.wait_stream(current_stream)
                    with torch.cuda.stream(csgmv_side_stream):
                        lora_out = compiled_csgmv_patch()
                    base_out = compiled_marlin_base_two_stream()
                    current_stream.wait_stream(csgmv_side_stream)
                    return base_out + lora_out

                def qlora_triton_two_stream():
                    current_stream = torch.cuda.current_stream(device)
                    triton_side_stream.wait_stream(current_stream)
                    with torch.cuda.stream(triton_side_stream):
                        lora_out = compiled_triton_patch()
                    base_out = compiled_marlin_base_two_stream()
                    current_stream.wait_stream(triton_side_stream)
                    return base_out + lora_out

                def qlora_torch_matmul_two_stream():
                    current_stream = torch.cuda.current_stream(device)
                    torch_matmul_side_stream.wait_stream(current_stream)
                    with torch.cuda.stream(torch_matmul_side_stream):
                        lora_out = compiled_torch_matmul_patch()
                    base_out = compiled_marlin_base_two_stream()
                    current_stream.wait_stream(torch_matmul_side_stream)
                    return base_out + lora_out

                providers: list[tuple[str, Callable[[], object]]] = [
                    ("bf16 dense base", compiled_dense_base),
                    ("int4 Marlin base", compiled_marlin_base),
                    ("SGLang csgmv LoRA patch", compiled_csgmv_patch),
                    ("SGLang triton LoRA patch", compiled_triton_patch),
                    ("Torch matmul LoRA patch", compiled_torch_matmul_patch),
                    (
                        "bf16 dense + csgmv sequential",
                        bf16_csgmv_sequential,
                    ),
                    (
                        "bf16 dense + triton sequential",
                        bf16_triton_sequential,
                    ),
                    (
                        "bf16 dense + torch matmul sequential",
                        bf16_torch_matmul_sequential,
                    ),
                    (
                        "bf16 dense + torch matmul two-stream",
                        bf16_torch_matmul_two_stream,
                    ),
                    (
                        "SGLang QLoRA csgmv sequential",
                        qlora_csgmv_sequential,
                    ),
                    ("SGLang QLoRA csgmv two-stream", qlora_csgmv_two_stream),
                    (
                        "SGLang QLoRA triton sequential",
                        qlora_triton_sequential,
                    ),
                    ("SGLang QLoRA triton two-stream", qlora_triton_two_stream),
                    (
                        "Torch QLoRA matmul sequential",
                        qlora_torch_matmul_sequential,
                    ),
                    ("Torch QLoRA matmul two-stream", qlora_torch_matmul_two_stream),
                ]
                if scheme_filter is not None:
                    providers = [
                        (scheme, fn)
                        for scheme, fn in providers
                        if scheme in scheme_filter
                    ]

                for scheme, fn in providers:
                    print(f"  measuring {scheme}", file=sys.stderr, flush=True)
                    try:
                        prewarm_activation(x, fn)
                        for _ in range(effective.warmup_iters):
                            fn()
                        torch.cuda.synchronize()
                        graph, _ = capture_cuda_graph(fn)
                        if cuda_profiler_range:
                            torch.cuda.cudart().cudaProfilerStart()
                        try:
                            timings = replay_and_measure(
                                graph,
                                flush_buffer=flush_buffer,
                                warmup_iters=effective.warmup_iters,
                                measure_iters=effective.measure_iters,
                            )
                        finally:
                            if cuda_profiler_range:
                                torch.cuda.cudart().cudaProfilerStop()
                        median_us = statistics.median(timings)
                        rows.append(
                            Measurement(
                                model=model_config.model,
                                projection=projection.name,
                                projection_kind=projection.kind,
                                in_features=projection.in_features,
                                out_features=projection.out_features,
                                token_rows=token_rows_value,
                                scheme=scheme,
                                latency_us=median_us,
                                median_us=median_us,
                                p20_us=percentile(timings, 0.20),
                                p80_us=percentile(timings, 0.80),
                                min_us=min(timings),
                                max_us=max(timings),
                                warmup_iters=effective.warmup_iters,
                                measure_iters=effective.measure_iters,
                                compiled=True,
                                cuda_graph=True,
                                l2_flush_mib=effective.l2_flush_mib,
                                activation_prewarm=True,
                                fake=False,
                                error=None,
                                note=notes[scheme],
                            )
                        )
                    except Exception as exc:
                        torch.cuda.synchronize()
                        rows.append(
                            error_measurement(
                                model=model_config.model,
                                projection=projection,
                                token_rows=token_rows_value,
                                scheme=scheme,
                                config=effective,
                                error=exc,
                                note=notes[scheme],
                            )
                        )

    torch.cuda.synchronize()
    return rows


def fake_measurements(token_rows: Iterable[int]) -> list[Measurement]:
    rows = []
    for n in token_rows:
        bf16 = 18.0 + 0.430 * n
        marlin = 28.0 + 0.215 * n
        patch = 16.0 + 0.050 * n
        sequential = marlin + patch * 0.82
        two_stream = max(marlin, patch) + 5.0 + 0.012 * n
        for scheme, value, note in [
            ("bf16 dense base", bf16, "Dense bf16 base projection."),
            ("int4 Marlin base", marlin, "Marlin int4 base projection."),
            ("SGLang csgmv LoRA patch", patch, "SGLang csgmv patch-only path."),
            ("SGLang QLoRA csgmv sequential", sequential, "Marlin base plus csgmv LoRA."),
            ("SGLang QLoRA csgmv two-stream", two_stream, "Overlapped csgmv LoRA patch and Marlin base."),
        ]:
            rows.append(
                Measurement(
                    model=None,
                    projection=None,
                    projection_kind=None,
                    in_features=None,
                    out_features=None,
                    token_rows=n,
                    scheme=scheme,
                    latency_us=round(value, 3),
                    median_us=round(value, 3),
                    p20_us=None,
                    p80_us=None,
                    min_us=None,
                    max_us=None,
                    warmup_iters=0,
                    measure_iters=0,
                    compiled=False,
                    cuda_graph=False,
                    l2_flush_mib=0,
                    activation_prewarm=False,
                    fake=True,
                    error=None,
                    note=note,
                )
            )
    return rows


def write_data(
    path: Path,
    config: BenchmarkConfig,
    measurements: list[Measurement],
    *,
    fake: bool,
    model_configs: Iterable[ModelConfig] | None = None,
) -> None:
    serialized_model_configs = []
    if model_configs is not None:
        for model_config in model_configs:
            serialized_model_configs.append(
                {
                    **asdict(model_config),
                    "projections": [
                        asdict(projection) for projection in model_config.projections
                    ],
                }
            )
    payload = {
        "metadata": {
            "fake": fake,
            "created_unix": time.time(),
            "config": asdict(config),
            "model_configs": serialized_model_configs,
            "scheme_order": SCHEME_ORDER,
            "research_summary": [
                "SGLang default LoRA backend is csgmv/ChunkedSGMV.",
                "This benchmark directly constructs LoRABatchInfo and calls the SGLang csgmv backend.",
                "QKV and gate/up use fused SGLang LoRA projection groups.",
                "csgmv rows launch the SGLang backend directly inside CUDA Graph; base Marlin and dense callables use torch.compile.",
                "Two-stream is still an experimental overlap comparison, not default SGLang behavior.",
                "SM reservation is applied only inside two-stream Marlin callables; regular base and sequential paths clear SGLANG_MARLIN_RESERVE_SMS.",
            ],
        },
        "measurements": [asdict(row) for row in measurements],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-config", type=Path, action="append", default=None)
    parser.add_argument("--projection", default="all")
    parser.add_argument(
        "--scheme",
        action="append",
        default=None,
        help="Benchmark only this scheme. May be passed more than once.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap the replay measurement window in cudaProfilerStart/Stop.",
    )
    parser.add_argument("--fake", action="store_true")
    parser.add_argument(
        "--tokens",
        type=parse_token_rows,
        default=parse_token_rows("1,4,8,16,32,64,128,256"),
    )
    parser.add_argument("--hidden-size", type=int, default=BenchmarkConfig.hidden_size)
    parser.add_argument("--output-size", type=int, default=BenchmarkConfig.output_size)
    parser.add_argument("--rank", type=int, default=BenchmarkConfig.lora_rank)
    parser.add_argument("--lora-alpha", type=float, default=BenchmarkConfig.lora_alpha)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=BenchmarkConfig.dtype)
    parser.add_argument("--quant-type", default=BenchmarkConfig.quant_type)
    parser.add_argument("--group-size", type=int, default=BenchmarkConfig.group_size)
    parser.add_argument("--warmup", type=int, default=BenchmarkConfig.warmup_iters)
    parser.add_argument("--iters", type=int, default=BenchmarkConfig.measure_iters)
    parser.add_argument("--l2-flush-mib", type=int, default=BenchmarkConfig.l2_flush_mib)
    parser.add_argument("--max-lora-chunk-size", type=int, default=BenchmarkConfig.max_lora_chunk_size)
    parser.add_argument("--compile-mode", default=BenchmarkConfig.compile_mode)
    parser.add_argument(
        "--two-stream-reserve-sms",
        type=int,
        default=BenchmarkConfig.two_stream_reserve_sms,
        help="Temporarily reserve this many SMs only for Marlin calls inside two-stream schemes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fake:
        config = BenchmarkConfig()
        output = args.output if args.output != DEFAULT_DATA else DEFAULT_FAKE_DATA
        write_data(
            output,
            config,
            fake_measurements([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            fake=True,
        )
        print(f"Wrote fake QLoRA kernel data to {output}")
        return

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    config = BenchmarkConfig(
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        lora_rank=args.rank,
        lora_alpha=args.lora_alpha,
        dtype=args.dtype,
        quant_type=args.quant_type,
        group_size=args.group_size,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        l2_flush_mib=args.l2_flush_mib,
        max_lora_chunk_size=args.max_lora_chunk_size,
        compile_mode=args.compile_mode,
        two_stream_reserve_sms=args.two_stream_reserve_sms,
    )
    model_configs = (
        [load_model_config(path) for path in args.model_config]
        if args.model_config
        else [make_single_projection_model(config)]
    )
    projection_filter = None
    if args.projection != "all":
        projection_filter = {
            item.strip() for item in args.projection.split(",") if item.strip()
        }
    scheme_filter = set(args.scheme) if args.scheme else None
    measurements = make_real_measurements(
        config,
        args.tokens,
        model_configs,
        projection_filter,
        scheme_filter,
        args.cuda_profiler_range,
    )
    write_data(args.output, config, measurements, fake=False, model_configs=model_configs)
    print(f"Wrote QLoRA kernel benchmark data to {args.output}")


if __name__ == "__main__":
    main()
