#!/usr/bin/env python3
"""Benchmark a synthetic Qwen-style decode transformer block."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_ENV_PYTHON = Path("/data/huanchen/miniforge3/envs/sglang/bin/python")
PYTHON = DEFAULT_ENV_PYTHON if DEFAULT_ENV_PYTHON.exists() else Path(sys.executable)
QLORA_PRECISIONS = {"qlora", "qlora-sequential"}
_QLORA_CUSTOM_OP = None
_QLORA_BENCHMARK_REGISTRY: dict[int, object] = {}
_NEXT_QLORA_BENCHMARK_ID = 0


@dataclass(frozen=True)
class ProjectionConfig:
    name: str
    kind: str
    in_features: int
    out_features: int
    slice_sizes: tuple[int, ...]
    bias: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model: str
    layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str
    dtype: str
    quant_type: str
    group_size: int
    lora_rank: int
    lora_alpha: float
    projections: tuple[ProjectionConfig, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    precision: str
    scope: str
    batch_size: int
    kv_len: int
    warmup_iters: int
    measure_iters: int
    l2_flush_mib: int
    torch_compile: bool
    cuda_graph: bool
    compile_mode: str
    two_stream_reserve_sms: int
    two_stream_layout: str
    qlora_base_wait_lora_a: bool
    cache_mode: str
    workspace_size: int
    workspace_l2_factor: float
    projection_precision_overrides: Optional[dict[str, str]] = None
    precision_config_name: Optional[str] = None


@dataclass(frozen=True)
class Result:
    model: str
    tp_size: int
    precision: str
    precision_config_name: Optional[str]
    scope: str
    batch_size: int
    kv_len: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    torch_compile: bool
    cuda_graph: bool
    l2_flush_mib: int
    cache_mode: str
    timing_source: str
    warmup_iters: int
    measure_iters: int
    attention_backend: str
    projection_precisions: Optional[dict[str, str]]
    workspace_size: int
    workspace_slot_bytes: int
    device_l2_bytes: int
    qlora_base_wait_lora_a: bool
    mean_us: float
    std_us: float
    median_us: float
    p20_us: float
    p80_us: float
    min_us: float
    max_us: float


@dataclass
class MarlinCaptureBuffers:
    output: object
    c_tmp: object
    a_tmp: object
    empty_dtype: object
    empty_int32: object


def register_qlora_benchmark_instance(benchmark: object) -> int:
    global _NEXT_QLORA_BENCHMARK_ID

    handle = _NEXT_QLORA_BENCHMARK_ID
    _NEXT_QLORA_BENCHMARK_ID += 1
    _QLORA_BENCHMARK_REGISTRY[handle] = benchmark
    return handle


def get_qlora_projection_custom_op():
    global _QLORA_CUSTOM_OP

    if _QLORA_CUSTOM_OP is not None:
        return _QLORA_CUSTOM_OP

    import torch

    @torch.library.custom_op(
        "sglang_bench::qlora_projection",
        mutates_args=(),
        schema="(Tensor x, int benchmark_handle, str projection_name) -> Tensor",
    )
    def qlora_projection(
        x: torch.Tensor,
        benchmark_handle: int,
        projection_name: str,
    ) -> torch.Tensor:
        benchmark = _QLORA_BENCHMARK_REGISTRY[benchmark_handle]
        return benchmark._run_multistream_qlora_projection(projection_name, x)

    @qlora_projection.register_fake
    def _(
        x: torch.Tensor,
        benchmark_handle: int,
        projection_name: str,
    ) -> torch.Tensor:
        benchmark = _QLORA_BENCHMARK_REGISTRY[benchmark_handle]
        output_size = benchmark.projection_output_size(projection_name)
        return x.new_empty((x.shape[0], output_size))

    _QLORA_CUSTOM_OP = qlora_projection
    return _QLORA_CUSTOM_OP


def add_repo_python_to_path() -> None:
    python_dir = REPO_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def ensure_single_rank_tp() -> None:
    import torch
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.distributed.is_initialized():
        if world_size == 1 and "MASTER_PORT" not in os.environ:
            rendezvous_path = (
                Path(tempfile.gettempdir()) / f"sglang-tp1-{uuid.uuid4().hex}.rdzv"
            )
            distributed_init_method = f"file://{rendezvous_path}"
        else:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT")
            if master_port is None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind((master_addr, 0))
                    master_port = str(sock.getsockname()[1])
            distributed_init_method = f"tcp://{master_addr}:{master_port}"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=local_rank,
            backend="nccl",
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )


def dtype_from_name(name: str):
    import torch

    normalized = name.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def parse_projection(raw: dict) -> ProjectionConfig:
    slice_sizes = tuple(int(x) for x in raw["slice_sizes"])
    out_features = int(raw["out_features"])
    if sum(slice_sizes) != out_features:
        raise ValueError(
            f"{raw['name']} slice_sizes sum to {sum(slice_sizes)}, expected {out_features}"
        )
    return ProjectionConfig(
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        in_features=int(raw["in_features"]),
        out_features=out_features,
        slice_sizes=slice_sizes,
        bias=bool(raw.get("bias", False)),
    )


def load_model_config(path: Path) -> ModelConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ModelConfig(
        model=str(raw["model"]),
        layers=int(raw["layers"]),
        hidden_size=int(raw["hidden_size"]),
        intermediate_size=int(raw["intermediate_size"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw["head_dim"]),
        max_position_embeddings=int(raw["max_position_embeddings"]),
        rope_theta=float(raw["rope_theta"]),
        rms_norm_eps=float(raw["rms_norm_eps"]),
        hidden_act=str(raw.get("hidden_act", "silu")),
        dtype=str(raw.get("dtype", "bf16")),
        quant_type=str(raw.get("quant_type", "uint4b8")),
        group_size=int(raw.get("group_size", 128)),
        lora_rank=int(raw.get("lora_rank", 16)),
        lora_alpha=int(raw.get("lora_alpha", 16)),
        projections=tuple(parse_projection(item) for item in raw["projections"]),
    )


def normalize_projection_name(name: str) -> str:
    if name == "up":
        return "gate_up"
    return name


def display_projection_name(name: str) -> str:
    if name == "gate_up":
        return "up"
    return name


def load_dynamic_projection_precisions(path: Path, batch_size: int) -> tuple[str, dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_batch = raw["projection_precision_by_batch"]
    key = str(batch_size)
    if key not in by_batch:
        raise ValueError(f"{path} does not define projection precision for batch size {batch_size}")
    assignment = {
        normalize_projection_name(name): str(precision)
        for name, precision in by_batch[key].items()
    }
    expected = {"qkv", "o", "gate_up", "down"}
    if set(assignment) != expected:
        raise ValueError(
            f"{path} batch {batch_size} defines {sorted(assignment)}, expected {sorted(expected)}"
        )
    for precision in assignment.values():
        if precision not in {"bf16", *QLORA_PRECISIONS}:
            raise ValueError(
                f"{path} batch {batch_size} uses unsupported dynamic precision {precision!r}"
            )
    return str(raw.get("name", path.stem)), assignment


def make_marlin_quant_config(group_size: int):
    from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig

    return GPTQMarlinConfig(
        weight_bits=4,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={
            "bits": 4,
            "group_size": group_size,
            "desc_act": False,
            "sym": True,
        },
    )


def pack_marlin_params(layer, dense_weight: object, group_size: int) -> None:
    import torch
    from sglang.srt.layers.quantization.gptq.gptq import scalar_types
    from sglang.srt.layers.quantization.utils import gptq_quantize_weights, pack_rows

    _, q_w, scales, g_idx, _ = gptq_quantize_weights(
        dense_weight,
        scalar_types.uint4b8,
        group_size=group_size,
        act_order=False,
    )
    qweight = pack_rows(
        q_w,
        num_bits=4,
        size_k=dense_weight.shape[0],
        size_n=dense_weight.shape[1],
    )
    layer.weight_loader(layer.qweight, qweight)
    layer.weight_loader(layer.scales, scales)
    if g_idx.numel() == 0:
        g_idx = (
            torch.arange(dense_weight.shape[0], device=dense_weight.device, dtype=torch.int32)
            // group_size
        )
    layer.weight_loader(layer.g_idx, g_idx.to(torch.int32))
    layer.quant_method.process_weights_after_loading(layer)


def load_linear_module_weight(
    layer,
    *,
    precision: str,
    logical_shape: tuple[int, int],
    group_size: int,
) -> None:
    import torch

    if precision in {"marlin", "qlora"}:
        dense_weight = torch.randn(logical_shape, device="cuda", dtype=torch.bfloat16)
        pack_marlin_params(layer, dense_weight, group_size)
    else:
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.data.normal_()
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data.normal_()
        layer.quant_method.process_weights_after_loading(layer)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def compile_callable(fn: Callable, enabled: bool, mode: str):
    if not enabled:
        return fn
    import torch

    if mode == "default":
        return torch.compile(fn, fullgraph=False)
    return torch.compile(fn, fullgraph=False, mode=mode)


def call_with_marlin_sm_reserve(fn: Callable, reserve_sms: int):
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
    start_event = torch.cuda.Event(enable_timing=True, external=True)
    end_event = torch.cuda.Event(enable_timing=True, external=True)
    torch.cuda.synchronize()
    with torch.no_grad(), torch.cuda.graph(graph):
        start_event.record()
        holder["out"] = fn()
        end_event.record()
    return graph, holder, start_event, end_event


def replay_and_measure(
    graph,
    *,
    flush_buffer,
    warmup_iters: int,
    measure_iters: int,
    start_event=None,
    end_event=None,
    timing_source: str = "replay",
):
    import torch

    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings: list[float] = []
    for i in range(measure_iters):
        if flush_buffer is not None:
            flush_buffer.fill_(i)
        if timing_source == "captured":
            if start_event is None or end_event is None:
                raise ValueError("Captured timing requires graph events")
            graph.replay()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event) * 1000.0)
        else:
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000.0)
    return timings


def tensor_nbytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def unique_tensor_nbytes(objects: list[object]) -> int:
    import torch

    seen: set[int] = set()
    total = 0

    def visit(obj: object) -> None:
        nonlocal total
        if obj is None:
            return
        if isinstance(obj, torch.Tensor):
            ptr = obj.untyped_storage().data_ptr()
            if ptr in seen:
                return
            seen.add(ptr)
            total += tensor_nbytes(obj)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                visit(value)
            return
        if hasattr(obj, "parameters") and hasattr(obj, "buffers"):
            for value in obj.parameters():
                visit(value)
            for value in obj.buffers():
                visit(value)
            return

    for obj in objects:
        visit(obj)
    return total


class DenseProjection:
    def __init__(self, projection: ProjectionConfig, dtype, torch_compile: bool, mode: str):
        import torch

        self.weight = torch.randn(
            (projection.in_features, projection.out_features),
            device="cuda",
            dtype=dtype,
        ).contiguous()
        self.bias = (
            torch.randn((projection.out_features,), device="cuda", dtype=dtype).contiguous()
            if projection.bias
            else None
        )

        def _forward(x):
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            return out

        self._forward = compile_callable(_forward, torch_compile, mode)

    def __call__(self, x):
        return self._forward(x)


class MarlinProjection:
    def __init__(
        self,
        projection: ProjectionConfig,
        *,
        dtype,
        quant_type_name: str,
        group_size: int,
        torch_compile: bool,
        mode: str,
    ):
        import torch
        from sgl_kernel.scalar_type import scalar_types
        from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        from sglang.test.test_marlin_utils import marlin_quantize

        self.out_features = projection.out_features
        self.in_features = projection.in_features
        self.quant_type = getattr(scalar_types, quant_type_name)
        dense_weight = torch.randn(
            (projection.in_features, projection.out_features),
            device="cuda",
            dtype=dtype,
        ).contiguous()
        _, self.qweight, self.scales, self.g_idx, self.sort_indices, _ = marlin_quantize(
            dense_weight, self.quant_type, group_size, False
        )
        self.workspace = marlin_make_workspace(torch.device("cuda"))
        self.bias = (
            torch.randn((projection.out_features,), device="cuda", dtype=dtype).contiguous()
            if projection.bias
            else None
        )

        def _forward(x):
            out = gptq_marlin_gemm(
                x,
                None,
                self.qweight,
                self.scales,
                None,
                None,
                self.g_idx,
                self.sort_indices,
                self.workspace,
                self.quant_type,
                x.shape[0],
                self.out_features,
                self.in_features,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=False,
                is_zp_float=False,
            )
            if self.bias is not None:
                out = out + self.bias
            return out

        self._forward = compile_callable(_forward, torch_compile, mode)

    def __call__(self, x):
        return self._forward(x)


class Torch2SLoRAPatch:
    def __init__(
        self,
        projection: ProjectionConfig,
        *,
        dtype,
        rank: int,
        alpha: float,
        torch_compile: bool,
        mode: str,
    ):
        import torch

        self.slice_sizes = projection.slice_sizes
        self.rank = rank
        self.scaling = alpha / rank
        self.output_offsets = [0]
        for slice_size in projection.slice_sizes:
            self.output_offsets.append(self.output_offsets[-1] + slice_size)
        n_slices = len(projection.slice_sizes)
        self.lora_a = torch.randn(
            (n_slices * rank, projection.in_features),
            device="cuda",
            dtype=dtype,
        ).contiguous()
        self.lora_b = torch.randn(
            (projection.out_features, rank),
            device="cuda",
            dtype=dtype,
        ).contiguous()

        def _patch_only(x):
            a_out = torch.matmul(x, self.lora_a.t())
            parts = []
            for idx, slice_size in enumerate(self.slice_sizes):
                r0 = idx * self.rank
                r1 = r0 + self.rank
                o0 = self.output_offsets[idx]
                o1 = o0 + slice_size
                parts.append(torch.matmul(a_out[:, r0:r1], self.lora_b[o0:o1, :].t()))
            if len(parts) == 1:
                return parts[0] * self.scaling
            return torch.cat(parts, dim=-1) * self.scaling

        self._patch_only = compile_callable(_patch_only, torch_compile, mode)

    def patch(self, x):
        return self._patch_only(x)


def make_column_lora_weights(
    projection: ProjectionConfig,
    *,
    rank: int,
    alpha: float,
    tp_world_size: int,
    dtype,
    device,
):
    import torch

    local_slice_sizes = tuple(size // tp_world_size for size in projection.slice_sizes)
    weights = {
        "slice_sizes": local_slice_sizes,
        "rank": rank,
        "scaling": alpha / rank,
        "a": torch.randn(
            (rank, projection.in_features), device=device, dtype=dtype
        ).contiguous(),
        "b": torch.randn(
            (sum(local_slice_sizes), rank), device=device, dtype=dtype
        ).contiguous(),
    }
    return weights


def make_row_lora_weights(
    projection: ProjectionConfig,
    *,
    rank: int,
    alpha: float,
    tp_world_size: int,
    dtype,
    device,
):
    import torch

    local_in_features = projection.in_features // tp_world_size
    weights = {
        "scaling": alpha / rank,
        "a": torch.randn((rank, local_in_features), device=device, dtype=dtype).contiguous(),
        "b": torch.randn((projection.out_features, rank), device=device, dtype=dtype).contiguous(),
    }
    return weights


def column_lora_a(x, weights):
    import torch

    return torch.matmul(x, weights["a"].transpose(0, 1))


def column_lora_b(a_out, weights):
    import torch

    return torch.matmul(a_out, weights["b"].transpose(0, 1)) * weights["scaling"]


def row_lora_a(x, weights):
    import torch

    return torch.matmul(x, weights["a"].transpose(0, 1))


def row_lora_b(x, weights):
    import torch

    return torch.matmul(x, weights["b"].transpose(0, 1)) * weights["scaling"]


class SyntheticTorchNativeAttnBackend:
    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        key_cache,
        value_cache,
        req_to_token_pool,
        token_to_kv_pool,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        import torch
        from torch.nn.functional import scaled_dot_product_attention

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        q_heads = q.view(self.batch_size, layer.tp_q_head_num, layer.qk_head_dim)
        o_heads = o.view(self.batch_size, layer.tp_q_head_num, layer.v_head_dim)

        if save_kv_cache and k is not None and v is not None:
            self.key_cache[:, -1].copy_(k.view(self.batch_size, layer.tp_k_head_num, layer.qk_head_dim))
            self.value_cache[:, -1].copy_(v.view(self.batch_size, layer.tp_v_head_num, layer.v_head_dim))

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        # This benchmark models fixed-shape decode batches: every request has the
        # same cached prefix length, so attention should run as one batched SDPA
        # call rather than per-request launches.
        batched_query = q_heads.unsqueeze(2)
        batched_key = self.key_cache.permute(0, 2, 1, 3)
        batched_value = self.value_cache.permute(0, 2, 1, 3)
        batched_out = scaled_dot_product_attention(
            batched_query,
            batched_key,
            batched_value,
            enable_gqa=use_gqa,
            scale=layer.scaling,
            is_causal=False,
        )
        o_heads.copy_(batched_out.squeeze(2))
        return o


class SyntheticDecodeAttention:
    def __init__(self, attn_module, batch_size: int, kv_len: int, dtype):
        import torch
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

        qk_head_dim = attn_module.head_dim
        tp_q_head_num = attn_module.num_heads
        tp_kv_head_num = attn_module.num_kv_heads
        total_tokens = batch_size * (kv_len + 1)
        self.req_to_token_pool = ReqToTokenPool(
            size=batch_size,
            max_context_len=kv_len + 1,
            device="cuda",
            enable_memory_saver=False,
        )
        self.token_to_kv_pool = MHATokenToKVPool(
            size=total_tokens,
            page_size=1,
            dtype=dtype,
            head_num=tp_kv_head_num,
            head_dim=qk_head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        self.attn_layer = RadixAttention(
            num_heads=tp_q_head_num,
            head_dim=qk_head_dim,
            scaling=attn_module.scaling,
            num_kv_heads=tp_kv_head_num,
            layer_id=0,
            quant_config=None,
            prefix="synthetic.attn",
        )

        req_pool_indices = torch.arange(1, batch_size + 1, dtype=torch.int64, device="cuda")
        token_slots = (
            torch.arange(1, total_tokens + 1, dtype=torch.int64, device="cuda")
            .view(batch_size, kv_len + 1)
            .contiguous()
        )
        self.req_to_token_pool.req_to_token[req_pool_indices] = token_slots.to(torch.int32)

        self.key_cache = torch.randn(
            (
                batch_size,
                kv_len + 1,
                tp_kv_head_num,
                qk_head_dim,
            ),
            device="cuda",
            dtype=dtype,
        ).contiguous()
        self.value_cache = torch.randn_like(self.key_cache)
        self.key_cache[:, -1].zero_()
        self.value_cache[:, -1].zero_()

        if kv_len > 0:
            self.token_to_kv_pool.set_kv_buffer(
                self.attn_layer,
                token_slots[:, :-1].reshape(-1),
                self.key_cache[:, :-1].reshape(-1, tp_kv_head_num, qk_head_dim),
                self.value_cache[:, :-1].reshape(-1, tp_kv_head_num, qk_head_dim),
            )

        self.backend = SyntheticTorchNativeAttnBackend(
            batch_size=batch_size,
            seq_len=kv_len + 1,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
        )
        self.forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=torch.zeros((batch_size,), dtype=torch.int64, device="cuda"),
            req_pool_indices=req_pool_indices,
            seq_lens=torch.full((batch_size,), kv_len + 1, dtype=torch.int32, device="cuda"),
            out_cache_loc=token_slots[:, -1].contiguous(),
            seq_lens_sum=batch_size * (kv_len + 1),
            seq_lens_cpu=torch.full((batch_size,), kv_len + 1, dtype=torch.int32, device="cpu"),
            positions=torch.full((batch_size,), kv_len, dtype=torch.int64, device="cuda"),
            num_token_non_padded=torch.tensor(batch_size, dtype=torch.int32, device="cuda"),
            mrope_positions=None,
        )

    def __call__(self, q, k, v):
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

        with forward_context(ForwardContext(attn_backend=self.backend)):
            return self.attn_layer(q, k, v, self.forward_batch)


class TransformerBlockBenchmark:
    def _cast_layer_modules(self, layer) -> None:
        for module in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            module.to(dtype=self.dtype)

    def _resolve_projection_precisions(self) -> dict[str, str]:
        if self.bench_config.projection_precision_overrides is not None:
            return dict(self.bench_config.projection_precision_overrides)
        return {
            "qkv": self.bench_config.precision,
            "o": self.bench_config.precision,
            "gate_up": self.bench_config.precision,
            "down": self.bench_config.precision,
        }

    def _build_projection_map(self) -> dict[str, object]:
        projection_sources = {
            "qkv": (
                self.shared_layer.self_attn.qkv_proj,
                None if self.quantized_layer is None else self.quantized_layer.self_attn.qkv_proj,
            ),
            "o": (
                self.shared_layer.self_attn.o_proj,
                None if self.quantized_layer is None else self.quantized_layer.self_attn.o_proj,
            ),
            "gate_up": (
                self.shared_layer.mlp.gate_up_proj,
                None if self.quantized_layer is None else self.quantized_layer.mlp.gate_up_proj,
            ),
            "down": (
                self.shared_layer.mlp.down_proj,
                None if self.quantized_layer is None else self.quantized_layer.mlp.down_proj,
            ),
        }
        projections: dict[str, object] = {}
        for name, precision in self.projection_precisions.items():
            bf16_module, quantized_module = projection_sources[name]
            if precision == "bf16":
                projections[name] = bf16_module
            else:
                if quantized_module is None:
                    raise RuntimeError(f"{name} requested {precision} but quantized layer is unavailable")
                projections[name] = quantized_module
        return projections

    def __init__(self, model_config: ModelConfig, bench_config: BenchmarkConfig):
        add_repo_python_to_path()
        import torch
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer
        from sglang.srt.server_args import set_global_server_args_for_scheduler

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this benchmark")

        set_global_server_args_for_scheduler(
            SimpleNamespace(
                rl_on_policy_target=None,
                enable_aiter_allreduce_fusion=False,
                attention_backend="torch_native",
                piecewise_cuda_graph_compiler="none",
                enable_symm_mem=False,
            )
        )

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        torch.set_grad_enabled(False)
        torch.cuda.set_device(self.local_rank)
        ensure_single_rank_tp()
        self.device = torch.device("cuda", self.local_rank)
        self.dtype = dtype_from_name(model_config.dtype)
        self.model_config = model_config
        self.bench_config = bench_config
        decoder_cfg = SimpleNamespace(
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            hidden_act=model_config.hidden_act,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            max_position_embeddings=model_config.max_position_embeddings,
            rope_theta=model_config.rope_theta,
            rms_norm_eps=model_config.rms_norm_eps,
        )
        self.shared_layer = Qwen2DecoderLayer(
            config=decoder_cfg,
            layer_id=0,
            quant_config=None,
            prefix="synthetic.layer.bf16",
        ).cuda().eval()
        self._cast_layer_modules(self.shared_layer)

        self.projection_precisions = self._resolve_projection_precisions()
        needs_marlin_layer = any(
            precision in {"marlin", *QLORA_PRECISIONS}
            for precision in self.projection_precisions.values()
        )
        self.quantized_layer = None
        if needs_marlin_layer:
            self.quantized_layer = Qwen2DecoderLayer(
                config=decoder_cfg,
                layer_id=0,
                quant_config=make_marlin_quant_config(model_config.group_size),
                prefix="synthetic.layer.quant",
            ).cuda().eval()
            self._cast_layer_modules(self.quantized_layer)

        self.attn = SyntheticDecodeAttention(
            self.shared_layer.self_attn,
            batch_size=bench_config.batch_size,
            kv_len=bench_config.kv_len,
            dtype=self.dtype,
        )
        self._init_layer_weights(self.shared_layer, precision="bf16")
        if self.quantized_layer is not None:
            self._init_layer_weights(self.quantized_layer, precision="qlora")

        self.input_layernorm = self.shared_layer.input_layernorm
        self.post_attention_layernorm = self.shared_layer.post_attention_layernorm
        self.self_attn = self.shared_layer.self_attn
        self.mlp = self.shared_layer.mlp
        self.rotary_emb = self.self_attn.rotary_emb
        self.act_fn = self.mlp.act_fn

        self.projections = self._build_projection_map()
        self.has_any_qlora_projection = any(
            precision in QLORA_PRECISIONS
            for precision in self.projection_precisions.values()
        )
        self.has_multistream_qlora_projection = any(
            precision == "qlora" for precision in self.projection_precisions.values()
        )
        self.qlora_custom_op = None
        self.qlora_custom_op_handle = None
        if self.has_multistream_qlora_projection:
            self.qlora_custom_op = get_qlora_projection_custom_op()
            self.qlora_custom_op_handle = register_qlora_benchmark_instance(self)
        self.column_lora_weights = {}
        self.row_lora_weights = {}
        self.column_lora_a_fns = {}
        self.column_lora_b_fns = {}
        self.row_lora_a_fns = {}
        self.row_lora_b_fns = {}
        self.column_base_fns = {}
        self.row_base_fns = {}
        self.marlin_capture_buffers: dict[str, MarlinCaptureBuffers] = {}
        self.qlora_side_stream = None
        self.qlora_comm_stream = None
        self.qlora_column_a_event = None
        self.qlora_row_a_event = None
        self.qlora_side_done_event = None
        self.qlora_comm_done_event = None
        if self.has_any_qlora_projection:
            if bench_config.cuda_graph:
                for name in ("qkv", "gate_up"):
                    if self.projection_precisions[name] in QLORA_PRECISIONS:
                        self.column_base_fns[name] = (
                            lambda tensor, name=name: self._run_preallocated_marlin_base(name, tensor)
                        )
                for name in ("o", "down"):
                    if self.projection_precisions[name] in QLORA_PRECISIONS:
                        self.row_base_fns[name] = (
                            lambda tensor, name=name: self._run_preallocated_marlin_base(name, tensor)
                        )
            else:
                for name in ("qkv", "gate_up"):
                    if self.projection_precisions[name] in QLORA_PRECISIONS:
                        self.column_base_fns[name] = (
                            lambda tensor, name=name: self.projections[name](tensor)[0]
                        )
                for name in ("o", "down"):
                    if self.projection_precisions[name] in QLORA_PRECISIONS:
                        self.row_base_fns[name] = (
                            lambda tensor, name=name: self.projections[name].quant_method.apply(
                                self.projections[name], tensor, None
                            )
                        )
            if self.has_multistream_qlora_projection:
                self.qlora_side_stream = torch.cuda.Stream(device=self.device)
                self.qlora_comm_stream = torch.cuda.Stream(device=self.device)
                # For CUDA-graph capture we want these to become internal
                # cross-stream dependencies, not standalone external event nodes.
                self.qlora_column_a_event = torch.cuda.Event()
                self.qlora_row_a_event = torch.cuda.Event()
                self.qlora_side_done_event = torch.cuda.Event()
                self.qlora_comm_done_event = torch.cuda.Event()
            for projection in model_config.projections:
                if self.projection_precisions[projection.name] not in QLORA_PRECISIONS:
                    continue
                if projection.name in {"qkv", "gate_up"}:
                    weights = make_column_lora_weights(
                        projection,
                        rank=model_config.lora_rank,
                        alpha=model_config.lora_alpha,
                        tp_world_size=self.world_size,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self.column_lora_weights[projection.name] = weights
                    self.column_lora_a_fns[projection.name] = compile_callable(
                        lambda tensor, weights=weights: column_lora_a(tensor, weights),
                        bench_config.torch_compile,
                        bench_config.compile_mode,
                    )
                    self.column_lora_b_fns[projection.name] = compile_callable(
                        lambda tensor, weights=weights: column_lora_b(tensor, weights),
                        bench_config.torch_compile,
                        bench_config.compile_mode,
                    )
                else:
                    weights = make_row_lora_weights(
                        projection,
                        rank=model_config.lora_rank,
                        alpha=model_config.lora_alpha,
                        tp_world_size=self.world_size,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self.row_lora_weights[projection.name] = weights
                    self.row_lora_a_fns[projection.name] = compile_callable(
                        lambda tensor, weights=weights: row_lora_a(tensor, weights),
                        bench_config.torch_compile,
                        bench_config.compile_mode,
                    )
                    self.row_lora_b_fns[projection.name] = compile_callable(
                        lambda tensor, weights=weights: row_lora_b(tensor, weights),
                        bench_config.torch_compile,
                        bench_config.compile_mode,
                    )

        self.hidden_states = torch.randn(
            (bench_config.batch_size, model_config.hidden_size),
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self.residual_states = torch.randn_like(self.hidden_states)
        self.o_inputs = torch.randn(
            (bench_config.batch_size, model_config.hidden_size // self.world_size),
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self.down_inputs = torch.randn(
            (bench_config.batch_size, model_config.intermediate_size // self.world_size),
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self.positions = torch.full(
            (bench_config.batch_size,),
            bench_config.kv_len,
            device=self.device,
            dtype=torch.int64,
        )
        device_props = torch.cuda.get_device_properties(self.device)
        l2_cache_size = getattr(device_props, "l2_cache_size", None)
        if l2_cache_size is None:
            l2_cache_size = getattr(device_props, "L2_cache_size")
        self.device_l2_bytes = int(l2_cache_size)

        self.flush_buffer = None
        if bench_config.cache_mode == "flush" and bench_config.l2_flush_mib > 0:
            elements = (
                bench_config.l2_flush_mib
                * 1024
                * 1024
                // torch.empty((), dtype=torch.int32).element_size()
            )
            self.flush_buffer = torch.empty((elements,), device=self.device, dtype=torch.int32)
        if self.has_any_qlora_projection:
            if bench_config.cuda_graph:
                for name in ("qkv", "gate_up"):
                    if self.projection_precisions[name] in QLORA_PRECISIONS:
                        self.marlin_capture_buffers[name] = self._allocate_marlin_capture_buffers(
                            self.projections[name], self.hidden_states.shape[0]
                        )
                for name in ("o", "down"):
                    if self.projection_precisions[name] in QLORA_PRECISIONS:
                        input_rows = self.o_inputs.shape[0] if name == "o" else self.down_inputs.shape[0]
                        self.marlin_capture_buffers[name] = self._allocate_marlin_capture_buffers(
                            self.projections[name], input_rows
                        )
    def _init_layer_weights(self, layer, *, precision: str) -> None:
        import torch

        torch.manual_seed(1234)
        load_linear_module_weight(
            layer.self_attn.qkv_proj,
            precision=precision,
            logical_shape=(
                self.model_config.hidden_size,
                self.model_config.hidden_size
                + 2 * self.model_config.num_key_value_heads * self.model_config.head_dim,
            ),
            group_size=self.model_config.group_size,
        )
        load_linear_module_weight(
            layer.self_attn.o_proj,
            precision=precision,
            logical_shape=(self.model_config.hidden_size, self.model_config.hidden_size),
            group_size=self.model_config.group_size,
        )
        load_linear_module_weight(
            layer.mlp.gate_up_proj,
            precision=precision,
            logical_shape=(
                self.model_config.hidden_size,
                2 * self.model_config.intermediate_size,
            ),
            group_size=self.model_config.group_size,
        )
        load_linear_module_weight(
            layer.mlp.down_proj,
            precision=precision,
            logical_shape=(
                self.model_config.intermediate_size,
                self.model_config.hidden_size,
            ),
            group_size=self.model_config.group_size,
        )
        if layer.self_attn.qkv_proj.bias is not None:
            layer.self_attn.qkv_proj.bias.data.normal_()
        layer.input_layernorm.weight.data.fill_(1)
        layer.post_attention_layernorm.weight.data.fill_(1)

    def _allocate_marlin_capture_buffers(self, module, rows: int) -> MarlinCaptureBuffers:
        import torch

        g_idx = getattr(module, "g_idx", None)
        has_act_order = g_idx is not None and g_idx.numel() > 0
        output_size = getattr(module, "output_size_per_partition", None) or module.output_size
        input_size = getattr(module, "input_size_per_partition", None) or module.input_size
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        max_m_block = min(((rows + 15) // 16) * 16, 64)
        return MarlinCaptureBuffers(
            output=torch.empty(
                (rows, output_size),
                device=self.device,
                dtype=self.dtype,
            ),
            c_tmp=torch.empty(
                sms * max_m_block * 256,
                device=self.device,
                dtype=torch.float32,
            ),
            a_tmp=(
                torch.empty(
                    (rows, input_size),
                    device=self.device,
                    dtype=self.dtype,
                )
                if has_act_order
                else torch.empty(0, device=self.device, dtype=self.dtype)
            ),
            empty_dtype=torch.empty(0, device=self.device, dtype=self.dtype),
            empty_int32=torch.empty(0, device=self.device, dtype=torch.int32),
        )

    def _run_preallocated_marlin_base(self, name: str, x):
        module = self.projections[name]
        buffers = self.marlin_capture_buffers[name]
        return module.scheme.kernel.apply(
            module,
            x,
            None,
            output=buffers.output,
            c_tmp=buffers.c_tmp,
            a_tmp=buffers.a_tmp,
            empty_dtype=buffers.empty_dtype,
            empty_int32=buffers.empty_int32,
        )

    def projection_output_size(self, name: str) -> int:
        module = self.projections[name]
        output_size = getattr(module, "output_size_per_partition", None) or module.output_size
        return int(output_size)

    def _run_multistream_qlora_projection(self, name: str, x):
        import torch
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        current_stream = torch.cuda.current_stream(self.device)
        assert self.qlora_side_stream is not None
        assert self.qlora_comm_stream is not None
        assert self.qlora_column_a_event is not None
        assert self.qlora_row_a_event is not None
        assert self.qlora_side_done_event is not None
        assert self.qlora_comm_done_event is not None
        base_wait_lora_a = self.bench_config.qlora_base_wait_lora_a

        if name in {"qkv", "gate_up"}:
            self.qlora_side_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.qlora_side_stream):
                lora_a_output = self.column_lora_a_fns[name](x)
                if base_wait_lora_a:
                    self.qlora_column_a_event.record(self.qlora_side_stream)
                lora_out = self.column_lora_b_fns[name](lora_a_output)
                self.qlora_side_done_event.record(self.qlora_side_stream)

            if base_wait_lora_a:
                current_stream.wait_event(self.qlora_column_a_event)
            base_out = call_with_marlin_sm_reserve(
                lambda: self.column_base_fns[name](x),
                self.bench_config.two_stream_reserve_sms,
            )
            current_stream.wait_event(self.qlora_side_done_event)
            return base_out + lora_out

        self.qlora_side_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.qlora_side_stream):
            lora_a_output = self.row_lora_a_fns[name](x)
            if base_wait_lora_a or self.world_size > 1:
                self.qlora_row_a_event.record(self.qlora_side_stream)

        if self.world_size > 1:
            self.qlora_comm_stream.wait_event(self.qlora_row_a_event)
            with torch.cuda.stream(self.qlora_comm_stream):
                lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
                self.qlora_comm_done_event.record(self.qlora_comm_stream)

            if base_wait_lora_a:
                current_stream.wait_event(self.qlora_row_a_event)
            output_parallel = call_with_marlin_sm_reserve(
                lambda: self.row_base_fns[name](x),
                self.bench_config.two_stream_reserve_sms,
            )

            with torch.cuda.stream(self.qlora_side_stream):
                self.qlora_side_stream.wait_event(self.qlora_comm_done_event)
                lora_out = self.row_lora_b_fns[name](lora_a_output)
                self.qlora_side_done_event.record(self.qlora_side_stream)

            current_stream.wait_event(self.qlora_comm_done_event)
            output_parallel = tensor_model_parallel_all_reduce(output_parallel)
            current_stream.wait_event(self.qlora_side_done_event)
        else:
            with torch.cuda.stream(self.qlora_side_stream):
                lora_out = self.row_lora_b_fns[name](lora_a_output)
                self.qlora_side_done_event.record(self.qlora_side_stream)
            if base_wait_lora_a:
                current_stream.wait_event(self.qlora_row_a_event)
            output_parallel = call_with_marlin_sm_reserve(
                lambda: self.row_base_fns[name](x),
                self.bench_config.two_stream_reserve_sms,
            )
            current_stream.wait_event(self.qlora_side_done_event)
        return output_parallel + lora_out

    def _run_projection(self, name: str, x, *, forward_batch=None):
        module = self.projections[name]

        def base_forward():
            out, _ = module(x, forward_batch=forward_batch) if forward_batch is not None else module(x)
            return out

        projection_precision = self.projection_precisions[name]
        if projection_precision not in QLORA_PRECISIONS:
            return base_forward()

        if projection_precision == "qlora-sequential":
            from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

            if name in {"qkv", "gate_up"}:
                lora_a_output = self.column_lora_a_fns[name](x)
                lora_out = self.column_lora_b_fns[name](lora_a_output)
                base_out = self.column_base_fns[name](x)
                return base_out + lora_out

            lora_a_output = self.row_lora_a_fns[name](x)
            if self.world_size > 1:
                lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
            lora_out = self.row_lora_b_fns[name](lora_a_output)
            output_parallel = self.row_base_fns[name](x)
            if self.world_size > 1:
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
            return output_parallel + lora_out

        assert self.qlora_custom_op is not None
        assert self.qlora_custom_op_handle is not None
        return self.qlora_custom_op(x, self.qlora_custom_op_handle, name)

    def qkv_only(self, x):
        return self._run_projection("qkv", x)

    def o_only(self, x):
        return self._run_projection("o", x)

    def up_only(self, x):
        return self._run_projection("gate_up", x)

    def down_only(self, x):
        return self._run_projection("down", x, forward_batch=self.attn.forward_batch)

    def _attn_mid_from_qkv(self, qkv):
        q_size = self.self_attn.q_size
        kv_size = self.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = self.rotary_emb(self.positions, q, k)
        return self.attn(q, k, v)

    def scope_input_tensor(self):
        if self.bench_config.scope == "o":
            return self.o_inputs
        if self.bench_config.scope == "down":
            return self.down_inputs
        return self.hidden_states

    def scope_slot_bytes(self) -> int:
        scope = self.bench_config.scope
        objects: list[object] = []
        if scope == "qkv":
            objects.extend([self.hidden_states, self.projections["qkv"]])
            if self.projection_precisions["qkv"] in QLORA_PRECISIONS:
                objects.append(self.column_lora_weights.get("qkv"))
        elif scope == "o":
            objects.extend([self.o_inputs, self.projections["o"]])
            if self.projection_precisions["o"] in QLORA_PRECISIONS:
                objects.append(self.row_lora_weights.get("o"))
        elif scope == "up":
            objects.extend([self.hidden_states, self.projections["gate_up"]])
            if self.projection_precisions["gate_up"] in QLORA_PRECISIONS:
                objects.append(self.column_lora_weights.get("gate_up"))
        elif scope == "down":
            objects.extend([self.down_inputs, self.projections["down"]])
            if self.projection_precisions["down"] in QLORA_PRECISIONS:
                objects.append(self.row_lora_weights.get("down"))
        elif scope == "attn":
            objects.extend(
                [
                    self.hidden_states,
                    self.positions,
                    self.input_layernorm,
                    self.projections["qkv"],
                    self.projections["o"],
                    self.attn.key_cache,
                    self.attn.value_cache,
                    self.attn.req_to_token_pool.req_to_token,
                    self.attn.forward_batch.input_ids,
                    self.attn.forward_batch.req_pool_indices,
                    self.attn.forward_batch.seq_lens,
                    self.attn.forward_batch.out_cache_loc,
                    self.attn.forward_batch.positions,
                    self.attn.forward_batch.num_token_non_padded,
                ]
            )
            if self.projection_precisions["qkv"] in QLORA_PRECISIONS:
                objects.append(self.column_lora_weights.get("qkv"))
            if self.projection_precisions["o"] in QLORA_PRECISIONS:
                objects.append(self.row_lora_weights.get("o"))
        elif scope == "mlp":
            objects.extend(
                [
                    self.hidden_states,
                    self.residual_states,
                    self.post_attention_layernorm,
                    self.projections["gate_up"],
                    self.projections["down"],
                ]
            )
            if self.projection_precisions["gate_up"] in QLORA_PRECISIONS:
                objects.append(self.column_lora_weights.get("gate_up"))
            if self.projection_precisions["down"] in QLORA_PRECISIONS:
                objects.append(self.row_lora_weights.get("down"))
        elif scope == "block":
            objects.extend(
                [
                    self.hidden_states,
                    self.residual_states,
                    self.positions,
                    self.input_layernorm,
                    self.post_attention_layernorm,
                    self.projections["qkv"],
                    self.projections["o"],
                    self.projections["gate_up"],
                    self.projections["down"],
                    self.attn.key_cache,
                    self.attn.value_cache,
                    self.attn.req_to_token_pool.req_to_token,
                    self.attn.forward_batch.input_ids,
                    self.attn.forward_batch.req_pool_indices,
                    self.attn.forward_batch.seq_lens,
                    self.attn.forward_batch.out_cache_loc,
                    self.attn.forward_batch.positions,
                    self.attn.forward_batch.num_token_non_padded,
                ]
            )
            if self.projection_precisions["qkv"] in QLORA_PRECISIONS:
                objects.append(self.column_lora_weights.get("qkv"))
            if self.projection_precisions["o"] in QLORA_PRECISIONS:
                objects.append(self.row_lora_weights.get("o"))
            if self.projection_precisions["gate_up"] in QLORA_PRECISIONS:
                objects.append(self.column_lora_weights.get("gate_up"))
            if self.projection_precisions["down"] in QLORA_PRECISIONS:
                objects.append(self.row_lora_weights.get("down"))
        else:
            raise ValueError(f"Unsupported scope {scope!r}")
        return unique_tensor_nbytes(objects)

    def compute_workspace_size(self) -> int:
        if self.bench_config.workspace_size > 0:
            return self.bench_config.workspace_size
        slot_bytes = max(1, self.scope_slot_bytes())
        l2_target = max(1, math.ceil(self.device_l2_bytes * self.bench_config.workspace_l2_factor))
        return max(2, 1 + math.ceil(l2_target / slot_bytes))

    def attn_core(self, x):
        qkv = self._run_projection("qkv", x)
        attn_out = self._attn_mid_from_qkv(qkv)
        return self._run_projection("o", attn_out)

    def attn_scope(self, x):
        hidden = self.input_layernorm(x)
        return self.attn_core(hidden)

    def mlp_core(self, x):
        gate_up = self._run_projection("gate_up", x)
        hidden = self.act_fn(gate_up)
        return self._run_projection("down", hidden, forward_batch=self.attn.forward_batch)

    def mlp_scope(self, x, residual):
        hidden, _ = self.post_attention_layernorm(x, residual)
        return self.mlp_core(hidden)

    def block_scope(self, x):
        residual = x
        hidden = self.input_layernorm(x)
        attn_out = self.attn_core(hidden)
        hidden, residual = self.post_attention_layernorm(attn_out, residual)
        mlp_out = self.mlp_core(hidden)
        return mlp_out, residual

    def scope_callable(self):
        scope = self.bench_config.scope
        if scope == "qkv":
            return lambda: self.qkv_only(self.hidden_states)
        if scope == "o":
            return lambda: self.o_only(self.o_inputs)
        if scope == "up":
            return lambda: self.up_only(self.hidden_states)
        if scope == "down":
            return lambda: self.down_only(self.down_inputs)
        if scope == "attn":
            return lambda: self.attn_scope(self.hidden_states)
        if scope == "mlp":
            return lambda: self.mlp_scope(self.hidden_states, self.residual_states)
        if scope == "block":
            return lambda: self.block_scope(self.hidden_states)
        raise ValueError(f"Unsupported scope {scope!r}")

    def measured_callable(self):
        return compile_callable(
            self.scope_callable(),
            self.bench_config.torch_compile,
            self.bench_config.compile_mode,
        )

    def result_from_timings(
        self,
        timings: list[float],
        *,
        timing_source: str,
        workspace_size: int,
        workspace_slot_bytes: int,
    ) -> Result:
        mean_us = statistics.fmean(timings)
        std_us = statistics.pstdev(timings) if len(timings) > 1 else 0.0
        return Result(
            model=self.model_config.model,
            tp_size=self.world_size,
            precision=self.bench_config.precision,
            precision_config_name=self.bench_config.precision_config_name,
            scope=self.bench_config.scope,
            batch_size=self.bench_config.batch_size,
            kv_len=self.bench_config.kv_len,
            hidden_size=self.model_config.hidden_size,
            intermediate_size=self.model_config.intermediate_size,
            num_attention_heads=self.model_config.num_attention_heads,
            num_key_value_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            torch_compile=self.bench_config.torch_compile,
            cuda_graph=self.bench_config.cuda_graph,
            l2_flush_mib=self.bench_config.l2_flush_mib,
            cache_mode=self.bench_config.cache_mode,
            timing_source=timing_source,
            warmup_iters=self.bench_config.warmup_iters,
            measure_iters=self.bench_config.measure_iters,
            attention_backend="torch_native",
            projection_precisions={
                display_projection_name(name): precision
                for name, precision in self.projection_precisions.items()
            },
            workspace_size=workspace_size,
            workspace_slot_bytes=workspace_slot_bytes,
            device_l2_bytes=self.device_l2_bytes,
            qlora_base_wait_lora_a=self.bench_config.qlora_base_wait_lora_a,
            mean_us=mean_us,
            std_us=std_us,
            median_us=statistics.median(timings),
            p20_us=percentile(timings, 0.2),
            p80_us=percentile(timings, 0.8),
            min_us=min(timings),
            max_us=max(timings),
        )

    def measure_workspace(self, cuda_profiler_range: bool) -> Result:
        import torch

        workspace_size = self.compute_workspace_size()
        workspace_slot_bytes = self.scope_slot_bytes()
        benches = [self]
        for _ in range(workspace_size - 1):
            benches.append(type(self)(self.model_config, self.bench_config))

        callables = [bench.measured_callable() for bench in benches]
        for bench, fn in zip(benches, callables):
            prewarm_activation(bench.scope_input_tensor(), fn)
            prewarm_activation(bench.scope_input_tensor(), fn)

        graphs: list[object | None] = []
        start_events: list[object | None] = []
        end_events: list[object | None] = []
        if self.bench_config.cuda_graph:
            for fn in callables:
                graph, _, start_event, end_event = capture_cuda_graph(fn)
                graphs.append(graph)
                start_events.append(start_event)
                end_events.append(end_event)
        else:
            graphs = [None] * workspace_size
            start_events = [None] * workspace_size
            end_events = [None] * workspace_size

        for i in range(self.bench_config.warmup_iters):
            slot = i % workspace_size
            if graphs[slot] is not None:
                graphs[slot].replay()
            else:
                callables[slot]()
        torch.cuda.synchronize()

        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()

        timings: list[float] = []
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for i in range(self.bench_config.measure_iters):
                slot = i % workspace_size
                if graphs[slot] is not None:
                    graphs[slot].replay()
                    torch.cuda.synchronize()
                    timings.append(start_events[slot].elapsed_time(end_events[slot]) * 1000.0)
                else:
                    start.record()
                    callables[slot]()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end) * 1000.0)
        finally:
            if cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStop()
        return self.result_from_timings(
            timings,
            timing_source="captured" if self.bench_config.cuda_graph else "replay",
            workspace_size=workspace_size,
            workspace_slot_bytes=workspace_slot_bytes,
        )

    def measure(self, cuda_profiler_range: bool) -> Result:
        import torch

        if self.bench_config.cache_mode == "workspace":
            return self.measure_workspace(cuda_profiler_range)

        fn = self.measured_callable()
        prewarm_activation(self.scope_input_tensor(), fn)

        for _ in range(self.bench_config.warmup_iters):
            fn()
        torch.cuda.synchronize()

        graph = None
        graph_start_event = None
        graph_end_event = None
        if self.bench_config.cuda_graph:
            graph, _, graph_start_event, graph_end_event = capture_cuda_graph(fn)

        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()

        try:
            if graph is not None:
                timings = replay_and_measure(
                    graph,
                    flush_buffer=self.flush_buffer,
                    warmup_iters=self.bench_config.warmup_iters,
                    measure_iters=self.bench_config.measure_iters,
                    start_event=graph_start_event,
                    end_event=graph_end_event,
                    timing_source="replay",
                )
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                timings = []
                for i in range(self.bench_config.measure_iters):
                    if self.flush_buffer is not None:
                        self.flush_buffer.fill_(i)
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end) * 1000.0)
        finally:
            if cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStop()

        return self.result_from_timings(
            timings,
            timing_source="replay",
            workspace_size=1,
            workspace_slot_bytes=self.scope_slot_bytes(),
        )


def run_nsys_profile(args: argparse.Namespace) -> None:
    stem = args.output.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        str(PYTHON),
        str(Path(__file__).resolve()),
        "--model-config",
        str(args.model_config),
        "--precision",
        args.precision,
        "--scope",
        args.scope,
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
        "--cache-mode",
        args.cache_mode,
        "--workspace-size",
        str(args.workspace_size),
        "--workspace-l2-factor",
        str(args.workspace_l2_factor),
        "--two-stream-reserve-sms",
        str(args.two_stream_reserve_sms),
        "--two-stream-layout",
        args.two_stream_layout,
        "--output",
        str(args.output),
        "--profile-nsys-internal",
    ]
    if args.precision_config is not None:
        cmd.extend(["--precision-config", str(args.precision_config)])
    if args.no_cuda_graph:
        cmd.append("--no-cuda-graph")
    if args.no_torch_compile:
        cmd.append("--no-torch-compile")
    if args.no_qlora_base_wait_lora_a:
        cmd.append("--no-qlora-base-wait-lora-a")

    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT / "python")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(stem.with_suffix(".sqlite")),
            str(stem.with_suffix(".nsys-rep")),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=["bf16", "marlin", "qlora", "qlora-sequential", "dynamic"],
        required=True,
    )
    parser.add_argument("--precision-config", type=Path)
    parser.add_argument(
        "--scope",
        choices=["block", "attn", "mlp", "qkv", "o", "up", "down"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--l2-flush-mib", type=int, default=96)
    parser.add_argument("--cache-mode", choices=["flush", "workspace"], default="flush")
    parser.add_argument("--workspace-size", type=int, default=0)
    parser.add_argument("--workspace-l2-factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    # With the no-base-wait schedule, Marlin already overlaps LoRA work; reserving
    # one SM did not improve the workspace CUDA-graph block benchmark.
    parser.add_argument("--two-stream-reserve-sms", type=int, default=0)
    parser.add_argument("--two-stream-layout", choices=["flipped", "standard"], default="flipped")
    parser.add_argument("--no-qlora-base-wait-lora-a", action="store_true")
    parser.add_argument("--profile-nsys", action="store_true")
    parser.add_argument("--profile-nsys-internal", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if args.profile_nsys and not args.profile_nsys_internal:
        run_nsys_profile(args)
        return

    if args.precision == "dynamic":
        if args.precision_config is None:
            raise ValueError("--precision dynamic requires --precision-config")
        if args.scope != "block":
            raise ValueError("dynamic precision is currently supported only for --scope block")
        precision_config_name, projection_precision_overrides = load_dynamic_projection_precisions(
            args.precision_config,
            args.batch_size,
        )
    else:
        if args.precision_config is not None:
            raise ValueError("--precision-config is only valid with --precision dynamic")
        precision_config_name = None
        projection_precision_overrides = None

    model_config = load_model_config(args.model_config)
    bench_config = BenchmarkConfig(
        precision=args.precision,
        scope=args.scope,
        batch_size=args.batch_size,
        kv_len=args.kv_len,
        warmup_iters=args.warmup,
        measure_iters=args.iters,
        l2_flush_mib=args.l2_flush_mib,
        torch_compile=not args.no_torch_compile,
        cuda_graph=not args.no_cuda_graph,
        compile_mode=args.compile_mode,
        two_stream_reserve_sms=args.two_stream_reserve_sms,
        two_stream_layout=args.two_stream_layout,
        qlora_base_wait_lora_a=not args.no_qlora_base_wait_lora_a,
        cache_mode=args.cache_mode,
        workspace_size=args.workspace_size,
        workspace_l2_factor=args.workspace_l2_factor,
        projection_precision_overrides=projection_precision_overrides,
        precision_config_name=precision_config_name,
    )
    bench = TransformerBlockBenchmark(model_config, bench_config)
    result = bench.measure(cuda_profiler_range=args.profile_nsys_internal)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
