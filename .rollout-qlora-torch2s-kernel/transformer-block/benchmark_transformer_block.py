#!/usr/bin/env python3
"""Benchmark a synthetic Qwen-style decode transformer block."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_ENV_PYTHON = Path("/data/huanchen/miniforge3/envs/sglang/bin/python")
PYTHON = DEFAULT_ENV_PYTHON if DEFAULT_ENV_PYTHON.exists() else Path(sys.executable)


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


@dataclass(frozen=True)
class Result:
    model: str
    precision: str
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
    warmup_iters: int
    measure_iters: int
    attention_backend: str
    median_us: float
    p20_us: float
    p80_us: float
    min_us: float
    max_us: float


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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
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
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        holder["out"] = fn()
    return graph, holder


def replay_and_measure(graph, *, flush_buffer, warmup_iters: int, measure_iters: int):
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
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0)
    return timings


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
    dtype,
    device,
):
    import torch

    n_slices = len(projection.slice_sizes)
    return {
        "slice_sizes": projection.slice_sizes,
        "rank": rank,
        "scaling": alpha / rank,
        "a": torch.randn(
            (n_slices * rank, projection.in_features), device=device, dtype=dtype
        ).contiguous(),
        "b": torch.randn(
            (projection.out_features, rank), device=device, dtype=dtype
        ).contiguous(),
    }


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
    weights["b"].mul_(weights["scaling"])
    return weights


def column_lora_a(x, weights):
    import torch

    return torch.matmul(x, weights["a"].transpose(0, 1))


def column_lora_b(a_out, weights):
    import torch

    rank = weights["rank"]
    offsets = [0]
    for size in weights["slice_sizes"]:
        offsets.append(offsets[-1] + size)
    parts = []
    for idx, slice_size in enumerate(weights["slice_sizes"]):
        r0 = idx * rank
        r1 = r0 + rank
        o0 = offsets[idx]
        o1 = o0 + slice_size
        parts.append(torch.matmul(a_out[:, r0:r1], weights["b"][o0:o1].transpose(0, 1)))
    return torch.cat(parts, dim=-1) * weights["scaling"]


def row_lora_a(x, weights):
    import torch

    return torch.matmul(x, weights["a"].transpose(0, 1))


def row_lora_b(x, weights):
    import torch

    return torch.matmul(x, weights["b"].transpose(0, 1))


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
        for req_idx in range(self.batch_size):
            per_req_query = q_heads[req_idx : req_idx + 1].transpose(0, 1).unsqueeze(0)
            per_req_key = self.key_cache[req_idx].transpose(0, 1).unsqueeze(0)
            per_req_value = self.value_cache[req_idx].transpose(0, 1).unsqueeze(0)
            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query,
                    per_req_key,
                    per_req_value,
                    enable_gqa=use_gqa,
                    scale=layer.scaling,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            o_heads[req_idx : req_idx + 1].copy_(per_req_out)

        return o


class SyntheticDecodeAttention:
    def __init__(self, model_config: ModelConfig, batch_size: int, kv_len: int, dtype):
        import torch
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

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
            head_num=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        self.attn_layer = RadixAttention(
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.head_dim,
            scaling=model_config.head_dim**-0.5,
            num_kv_heads=model_config.num_key_value_heads,
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
                model_config.num_key_value_heads,
                model_config.head_dim,
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
                self.key_cache[:, :-1].reshape(-1, model_config.num_key_value_heads, model_config.head_dim),
                self.value_cache[:, :-1].reshape(-1, model_config.num_key_value_heads, model_config.head_dim),
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

        torch.set_grad_enabled(False)
        torch.cuda.set_device(0)
        ensure_single_rank_tp()
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        self.device = torch.device("cuda")
        self.dtype = dtype_from_name(model_config.dtype)
        self.model_config = model_config
        self.bench_config = bench_config
        quant_config = (
            make_marlin_quant_config(model_config.group_size)
            if bench_config.precision in {"marlin", "qlora"}
            else None
        )
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
        self.layer = Qwen2DecoderLayer(
            config=decoder_cfg,
            layer_id=0,
            quant_config=quant_config,
            prefix="synthetic.layer",
        ).cuda().eval()
        for module in (
            self.layer.input_layernorm,
            self.layer.post_attention_layernorm,
            self.layer.self_attn.qkv_proj,
            self.layer.self_attn.o_proj,
            self.layer.mlp.gate_up_proj,
            self.layer.mlp.down_proj,
        ):
            module.to(dtype=self.dtype)

        self.attn = SyntheticDecodeAttention(
            model_config,
            batch_size=bench_config.batch_size,
            kv_len=bench_config.kv_len,
            dtype=self.dtype,
        )
        self._init_layer_weights()

        self.input_layernorm = self.layer.input_layernorm
        self.post_attention_layernorm = self.layer.post_attention_layernorm
        self.self_attn = self.layer.self_attn
        self.mlp = self.layer.mlp
        self.rotary_emb = self.self_attn.rotary_emb
        self.act_fn = self.mlp.act_fn

        self.projections = {
            "qkv": self.self_attn.qkv_proj,
            "o": self.self_attn.o_proj,
            "gate_up": self.mlp.gate_up_proj,
            "down": self.mlp.down_proj,
        }
        self.compiled_input_layernorm = compile_callable(
            lambda x: self.input_layernorm(x),
            bench_config.torch_compile and bench_config.precision == "qlora",
            bench_config.compile_mode,
        )
        self.compiled_post_attention_layernorm = compile_callable(
            lambda x, residual: self.post_attention_layernorm(x, residual),
            bench_config.torch_compile and bench_config.precision == "qlora",
            bench_config.compile_mode,
        )
        self.compiled_act_fn = compile_callable(
            lambda x: self.act_fn(x),
            bench_config.torch_compile and bench_config.precision == "qlora",
            bench_config.compile_mode,
        )
        self.compiled_attn_mid = compile_callable(
            lambda qkv: self._attn_mid_from_qkv(qkv),
            bench_config.torch_compile and bench_config.precision == "qlora",
            bench_config.compile_mode,
        )
        self.column_lora_weights = {}
        self.row_lora_weights = {}
        self.column_lora_a_fns = {}
        self.column_lora_b_fns = {}
        self.row_lora_a_fns = {}
        self.row_lora_b_fns = {}
        self.column_base_fns = {}
        self.row_base_fns = {}
        self.qlora_base_stream = None
        self.qlora_comm_stream = None
        if bench_config.precision == "qlora":
            maybe_compile = torch.compile if bench_config.torch_compile else lambda fn, **_: fn
            self.column_base_fns["qkv"] = maybe_compile(
                lambda tensor: self.projections["qkv"](tensor)[0],
                fullgraph=False,
            )
            self.column_base_fns["gate_up"] = maybe_compile(
                lambda tensor: self.projections["gate_up"](tensor)[0],
                fullgraph=False,
            )
            self.row_base_fns["o"] = maybe_compile(
                lambda tensor: self.projections["o"].quant_method.apply(
                    self.projections["o"], tensor, None
                ),
                fullgraph=False,
            )
            self.row_base_fns["down"] = maybe_compile(
                lambda tensor: self.projections["down"].quant_method.apply(
                    self.projections["down"], tensor, None
                ),
                fullgraph=False,
            )
            self.qlora_base_stream = torch.cuda.Stream(device=self.device)
            self.qlora_comm_stream = torch.cuda.Stream(device=self.device)
            for projection in model_config.projections:
                if projection.name in {"qkv", "gate_up"}:
                    weights = make_column_lora_weights(
                        projection,
                        rank=model_config.lora_rank,
                        alpha=model_config.lora_alpha,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self.column_lora_weights[projection.name] = weights
                    self.column_lora_a_fns[projection.name] = maybe_compile(
                        lambda tensor, weights=weights: column_lora_a(tensor, weights),
                        fullgraph=False,
                    )
                    self.column_lora_b_fns[projection.name] = maybe_compile(
                        lambda tensor, weights=weights: column_lora_b(tensor, weights),
                        fullgraph=False,
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
                    self.row_lora_a_fns[projection.name] = maybe_compile(
                        lambda tensor, weights=weights: row_lora_a(tensor, weights),
                        fullgraph=False,
                    )
                    self.row_lora_b_fns[projection.name] = maybe_compile(
                        lambda tensor, weights=weights: row_lora_b(tensor, weights),
                        fullgraph=False,
                    )

        self.hidden_states = torch.randn(
            (bench_config.batch_size, model_config.hidden_size),
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self.residual_states = torch.randn_like(self.hidden_states)
        self.mlp_inputs = torch.randn(
            (bench_config.batch_size, model_config.intermediate_size),
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self.positions = torch.full(
            (bench_config.batch_size,),
            bench_config.kv_len,
            device=self.device,
            dtype=torch.int64,
        )

        self.flush_buffer = None
        if bench_config.l2_flush_mib > 0:
            elements = (
                bench_config.l2_flush_mib
                * 1024
                * 1024
                // torch.empty((), dtype=torch.int32).element_size()
            )
            self.flush_buffer = torch.empty((elements,), device=self.device, dtype=torch.int32)

    def _init_layer_weights(self) -> None:
        import torch

        torch.manual_seed(1234)
        load_linear_module_weight(
            self.layer.self_attn.qkv_proj,
            precision=self.bench_config.precision,
            logical_shape=(
                self.model_config.hidden_size,
                self.model_config.hidden_size
                + 2 * self.model_config.num_key_value_heads * self.model_config.head_dim,
            ),
            group_size=self.model_config.group_size,
        )
        load_linear_module_weight(
            self.layer.self_attn.o_proj,
            precision=self.bench_config.precision,
            logical_shape=(self.model_config.hidden_size, self.model_config.hidden_size),
            group_size=self.model_config.group_size,
        )
        load_linear_module_weight(
            self.layer.mlp.gate_up_proj,
            precision=self.bench_config.precision,
            logical_shape=(
                self.model_config.hidden_size,
                2 * self.model_config.intermediate_size,
            ),
            group_size=self.model_config.group_size,
        )
        load_linear_module_weight(
            self.layer.mlp.down_proj,
            precision=self.bench_config.precision,
            logical_shape=(
                self.model_config.intermediate_size,
                self.model_config.hidden_size,
            ),
            group_size=self.model_config.group_size,
        )
        if self.layer.self_attn.qkv_proj.bias is not None:
            self.layer.self_attn.qkv_proj.bias.data.normal_()
        self.layer.input_layernorm.weight.data.fill_(1)
        self.layer.post_attention_layernorm.weight.data.fill_(1)

    def _run_projection(self, name: str, x, *, forward_batch=None):
        import torch
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        module = self.projections[name]

        def base_forward():
            out, _ = module(x, forward_batch=forward_batch) if forward_batch is not None else module(x)
            return out

        def row_base_parallel():
            return module.quant_method.apply(module, x, None)

        if self.bench_config.precision != "qlora":
            return base_forward()

        current_stream = torch.cuda.current_stream(self.device)
        assert self.qlora_base_stream is not None
        assert self.qlora_comm_stream is not None

        if name in {"qkv", "gate_up"}:
            lora_a_output = self.column_lora_a_fns[name](x)
            self.qlora_base_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.qlora_base_stream):
                base_out = call_with_marlin_sm_reserve(
                    lambda: self.column_base_fns[name](x),
                    self.bench_config.two_stream_reserve_sms,
                )
            lora_out = self.column_lora_b_fns[name](lora_a_output)
            current_stream.wait_stream(self.qlora_base_stream)
            return base_out + lora_out

        self.qlora_base_stream.wait_stream(current_stream)
        lora_a_output = self.row_lora_a_fns[name](x)
        with torch.cuda.stream(self.qlora_base_stream):
            output_parallel = call_with_marlin_sm_reserve(
                lambda: self.row_base_fns[name](x),
                self.bench_config.two_stream_reserve_sms,
            )

        if self.world_size > 1:
            self.qlora_comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.qlora_comm_stream):
                lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)

            current_stream.wait_stream(self.qlora_comm_stream)
            lora_out = self.row_lora_b_fns[name](lora_a_output)

            with torch.cuda.stream(self.qlora_base_stream):
                self.qlora_base_stream.wait_stream(self.qlora_comm_stream)
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
            current_stream.wait_stream(self.qlora_base_stream)
        else:
            lora_out = self.row_lora_b_fns[name](lora_a_output)
            current_stream.wait_stream(self.qlora_base_stream)
        return output_parallel + lora_out

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

    def attn_core(self, x):
        qkv = self._run_projection("qkv", x)
        attn_out = (
            self.compiled_attn_mid(qkv)
            if self.bench_config.precision == "qlora"
            else self._attn_mid_from_qkv(qkv)
        )
        return self._run_projection("o", attn_out)

    def attn_scope(self, x):
        hidden = (
            self.compiled_input_layernorm(x)
            if self.bench_config.precision == "qlora"
            else self.input_layernorm(x)
        )
        return self.attn_core(hidden)

    def mlp_core(self, x):
        gate_up = self._run_projection("gate_up", x)
        hidden = (
            self.compiled_act_fn(gate_up)
            if self.bench_config.precision == "qlora"
            else self.act_fn(gate_up)
        )
        return self._run_projection("down", hidden, forward_batch=self.attn.forward_batch)

    def mlp_scope(self, x, residual):
        hidden, _ = (
            self.compiled_post_attention_layernorm(x, residual)
            if self.bench_config.precision == "qlora"
            else self.post_attention_layernorm(x, residual)
        )
        return self.mlp_core(hidden)

    def block_scope(self, x):
        residual = x
        hidden = (
            self.compiled_input_layernorm(x)
            if self.bench_config.precision == "qlora"
            else self.input_layernorm(x)
        )
        attn_out = self.attn_core(hidden)
        hidden, residual = (
            self.compiled_post_attention_layernorm(attn_out, residual)
            if self.bench_config.precision == "qlora"
            else self.post_attention_layernorm(attn_out, residual)
        )
        mlp_out = self.mlp_core(hidden)
        return mlp_out, residual

    def scope_callable(self):
        scope = self.bench_config.scope
        if scope == "qkv":
            raw = lambda: self.qkv_only(self.hidden_states)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        if scope == "o":
            raw = lambda: self.o_only(self.hidden_states)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        if scope == "up":
            raw = lambda: self.up_only(self.hidden_states)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        if scope == "down":
            raw = lambda: self.down_only(self.mlp_inputs)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        if scope == "attn":
            raw = lambda: self.attn_scope(self.hidden_states)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        if scope == "mlp":
            raw = lambda: self.mlp_scope(self.hidden_states, self.residual_states)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        if scope == "block":
            raw = lambda: self.block_scope(self.hidden_states)
            return (
                compile_callable(raw, self.bench_config.torch_compile, self.bench_config.compile_mode)
                if self.bench_config.precision != "qlora"
                else raw
            )
        raise ValueError(f"Unsupported scope {scope!r}")

    def measure(self, cuda_profiler_range: bool) -> Result:
        import torch

        fn = self.scope_callable()
        prewarm_activation(self.hidden_states, fn)

        for _ in range(self.bench_config.warmup_iters):
            fn()
        torch.cuda.synchronize()

        graph = None
        if self.bench_config.cuda_graph:
            graph, _ = capture_cuda_graph(fn)

        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()

        try:
            if graph is not None:
                timings = replay_and_measure(
                    graph,
                    flush_buffer=self.flush_buffer,
                    warmup_iters=self.bench_config.warmup_iters,
                    measure_iters=self.bench_config.measure_iters,
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

        return Result(
            model=self.model_config.model,
            precision=self.bench_config.precision,
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
            warmup_iters=self.bench_config.warmup_iters,
            measure_iters=self.bench_config.measure_iters,
            attention_backend="torch_native",
            median_us=statistics.median(timings),
            p20_us=percentile(timings, 0.2),
            p80_us=percentile(timings, 0.8),
            min_us=min(timings),
            max_us=max(timings),
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
        "--two-stream-reserve-sms",
        str(args.two_stream_reserve_sms),
        "--two-stream-layout",
        args.two_stream_layout,
        "--output",
        str(args.output),
        "--profile-nsys-internal",
    ]
    if args.no_cuda_graph:
        cmd.append("--no-cuda-graph")
    if args.no_torch_compile:
        cmd.append("--no-torch-compile")

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
    parser.add_argument("--precision", choices=["bf16", "marlin", "qlora"], required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--two-stream-reserve-sms", type=int, default=1)
    parser.add_argument("--two-stream-layout", choices=["flipped", "standard"], default="flipped")
    parser.add_argument("--profile-nsys", action="store_true")
    parser.add_argument("--profile-nsys-internal", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile_nsys and not args.profile_nsys_internal:
        run_nsys_profile(args)
        return

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
    )
    bench = TransformerBlockBenchmark(model_config, bench_config)
    result = bench.measure(cuda_profiler_range=args.profile_nsys_internal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
