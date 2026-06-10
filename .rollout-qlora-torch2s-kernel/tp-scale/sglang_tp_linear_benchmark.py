#!/usr/bin/env python3
"""Profile a real SGLang tensor-parallel up/down projection pair."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.jit_kernel.activation import silu_and_mul as sglang_silu_and_mul


@dataclass(frozen=True)
class Result:
    tp: int
    rank: int
    local_rank: int
    layer: str
    kind: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    gate_up_local_output_features: int | None
    down_local_input_features: int | None
    median_us: float
    min_us: float
    max_us: float
    warmup: int
    iters: int
    cuda_graph: bool
    torch_compile: bool
    l2_flush_mib: int
    uses_sglang_tp_linear: bool
    uses_merged_column_parallel_gate_up: bool
    uses_row_parallel_down: bool
    forced_nccl_all_reduce: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["bf16", "marlin", "qlora"], required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=27648)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--l2-flush-mib", type=int, default=96)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--two-stream-reserve-sms", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-torch-compile", action="store_true")
    parser.add_argument(
        "--force-nccl-all-reduce",
        action="store_true",
        help="Disable SGLang custom all-reduce so row-parallel TP communication profiles as NCCL.",
    )
    return parser.parse_args()


def init_sglang_tp(force_nccl_all_reduce: bool) -> tuple[int, int, int]:
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if force_nccl_all_reduce:
        set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return world_size, rank, local_rank


def make_quant_config():
    from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig

    return GPTQMarlinConfig(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={"bits": 4, "group_size": 128, "desc_act": False, "sym": True},
    )


def make_layers(args: argparse.Namespace):
    from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear

    quant_config = make_quant_config() if args.kind in {"marlin", "qlora"} else None
    gate_up = MergedColumnParallelLinear(
        input_size=args.hidden_size,
        output_sizes=[args.intermediate_size, args.intermediate_size],
        bias=False,
        gather_output=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="tp_scale.gate_up_proj",
    ).cuda()
    down = RowParallelLinear(
        input_size=args.intermediate_size,
        output_size=args.hidden_size,
        bias=False,
        input_is_parallel=True,
        reduce_results=True,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="tp_scale.down_proj",
    ).cuda()
    return gate_up, down


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    sglang_silu_and_mul(x, out)
    return out


@contextmanager
def marlin_sm_reserve(reserve_sms: int):
    previous = os.environ.get("SGLANG_MARLIN_RESERVE_SMS")
    try:
        if reserve_sms > 0:
            os.environ["SGLANG_MARLIN_RESERVE_SMS"] = str(reserve_sms)
        else:
            os.environ.pop("SGLANG_MARLIN_RESERVE_SMS", None)
        yield
    finally:
        if previous is None:
            os.environ.pop("SGLANG_MARLIN_RESERVE_SMS", None)
        else:
            os.environ["SGLANG_MARLIN_RESERVE_SMS"] = previous


def make_lora_weights(args: argparse.Namespace, tp: int, tp_rank: int, device: torch.device):
    dtype = torch.bfloat16
    scaling = args.lora_alpha / args.lora_rank
    gate_up_local = args.intermediate_size // tp
    down_local = args.intermediate_size // tp
    gate_up_b = torch.randn(
        (gate_up_local * 2, args.lora_rank), device=device, dtype=dtype
    ).contiguous()
    down_a = torch.randn(
        (args.lora_rank, down_local), device=device, dtype=dtype
    ).contiguous()
    if scaling != 1.0:
        gate_up_b.mul_(scaling)
        down_a.mul_(scaling)
    return {
        "gate_up_a": torch.randn(
            (args.lora_rank * 2, args.hidden_size), device=device, dtype=dtype
        ).contiguous(),
        "gate_up_b": gate_up_b,
        "down_a": down_a,
        "down_b": torch.randn(
            (args.hidden_size, args.lora_rank), device=device, dtype=dtype
        ).contiguous(),
    }


def gate_up_lora_a(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(x, weights["gate_up_a"].transpose(0, 1))


def gate_up_lora_b(
    a_out: torch.Tensor, weights: dict[str, torch.Tensor]
) -> torch.Tensor:
    rank = weights["down_a"].shape[0]
    half = weights["gate_up_b"].shape[0] // 2
    gate = torch.matmul(
        a_out[:, :rank],
        weights["gate_up_b"][:half].transpose(0, 1),
    )
    up = torch.matmul(
        a_out[:, rank:],
        weights["gate_up_b"][half:].transpose(0, 1),
    )
    return torch.cat([gate, up], dim=-1)


def down_lora_a(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(x, weights["down_a"].transpose(0, 1))


def pack_marlin_params(layer, dense_weight: torch.Tensor) -> None:
    from sglang.srt.layers.quantization.gptq.gptq import scalar_types
    from sglang.srt.layers.quantization.utils import gptq_quantize_weights, pack_rows

    _, q_w, scales, g_idx, _ = gptq_quantize_weights(
        dense_weight,
        scalar_types.uint4b8,
        group_size=128,
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
            // 128
        )
    layer.weight_loader(layer.g_idx, g_idx.to(torch.int32))
    layer.quant_method.process_weights_after_loading(layer)


def load_synthetic_weight(
    layer,
    *,
    kind: str,
    logical_shape: tuple[int, int],
    checkpoint_shape: tuple[int, int],
    device: torch.device,
) -> None:
    if kind in {"marlin", "qlora"}:
        dense_weight = torch.randn(logical_shape, device=device, dtype=torch.bfloat16)
        pack_marlin_params(layer, dense_weight)
    else:
        dense_weight = torch.randn(checkpoint_shape, device=device, dtype=torch.bfloat16)
        layer.weight_loader(layer.weight, dense_weight)
        layer.quant_method.process_weights_after_loading(layer)


def load_synthetic_weights(gate_up, down, args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(1234)
    load_synthetic_weight(
        gate_up,
        kind=args.kind,
        logical_shape=(args.hidden_size, args.intermediate_size * 2),
        checkpoint_shape=(args.intermediate_size * 2, args.hidden_size),
        device=device,
    )
    load_synthetic_weight(
        down,
        kind=args.kind,
        logical_shape=(args.intermediate_size, args.hidden_size),
        checkpoint_shape=(args.hidden_size, args.intermediate_size),
        device=device,
    )


def l2_flush_fn(mib: int, device: torch.device) -> Callable[[], None]:
    if mib <= 0:
        return lambda: None
    elements = mib * 1024 * 1024 // torch.empty((), dtype=torch.int32).element_size()
    buf = torch.empty((elements,), device=device, dtype=torch.int32)

    def flush() -> None:
        buf.fill_(1)

    return flush


def capture_graph(fn: Callable[[], torch.Tensor], device: torch.device):
    from sglang.srt.distributed import get_tp_group

    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(device)
    with get_tp_group().graph_capture(stream=stream):
        with torch.cuda.graph(graph, stream=stream):
            out = fn()
    torch.cuda.synchronize(device)
    return graph, out


def tp_cpu_barrier(world_size: int) -> None:
    if world_size <= 1:
        return
    from sglang.srt.distributed import get_tp_group

    torch.distributed.barrier(group=get_tp_group().cpu_group)


def main() -> None:
    args = parse_args()
    world_size, rank, local_rank = init_sglang_tp(args.force_nccl_all_reduce)
    device = torch.device("cuda", local_rank)

    gate_up, down = make_layers(args)
    gate_up.eval()
    down.eval()
    load_synthetic_weights(gate_up, down, args, device)
    lora_weights = None
    gate_up_lora_a_fn = None
    gate_up_lora_b_fn = None
    down_lora_a_fn = None
    down_lora_b_fn = None
    qlora_base_stream = None
    qlora_comm_stream = None
    if args.kind == "qlora":
        lora_weights = make_lora_weights(args, world_size, rank, device)
        qlora_base_stream = torch.cuda.Stream(device=device)
        qlora_comm_stream = torch.cuda.Stream(device=device)
        maybe_compile = torch.compile if not args.no_torch_compile else lambda fn, **_: fn
        gate_up_lora_a_fn = maybe_compile(
            lambda tensor: gate_up_lora_a(tensor, lora_weights),
            fullgraph=False,
        )
        gate_up_lora_b_fn = maybe_compile(
            lambda tensor: gate_up_lora_b(tensor, lora_weights),
            fullgraph=False,
        )
        down_lora_a_fn = maybe_compile(
            lambda tensor: down_lora_a(tensor, lora_weights),
            fullgraph=False,
        )
        down_lora_b_fn = maybe_compile(
            lambda tensor: torch.matmul(tensor, lora_weights["down_b"].transpose(0, 1)),
            fullgraph=False,
        )
    x = torch.randn((args.tokens, args.hidden_size), device=device, dtype=torch.bfloat16)

    def forward() -> torch.Tensor:
        if args.kind != "qlora":
            hidden, _ = gate_up(x)
            hidden = silu_and_mul(hidden)
            out, _ = down(hidden)
            return out

        assert lora_weights is not None
        assert gate_up_lora_a_fn is not None
        assert gate_up_lora_b_fn is not None
        assert down_lora_a_fn is not None
        assert down_lora_b_fn is not None

        assert qlora_base_stream is not None
        assert qlora_comm_stream is not None
        current_stream = torch.cuda.current_stream(device)
        # Column-parallel gate/up: run the full LoRA patch on the current stream
        # while Marlin runs on the side stream.
        lora_gate_up_a = gate_up_lora_a_fn(x)
        qlora_base_stream.wait_stream(current_stream)
        with torch.cuda.stream(qlora_base_stream):
            base_gate_up, _ = gate_up(x)
        lora_gate_up = gate_up_lora_b_fn(lora_gate_up_a)
        current_stream.wait_stream(qlora_base_stream)
        hidden = base_gate_up + lora_gate_up
        hidden = silu_and_mul(hidden)

        # Three-stream row TP experiment:
        # - current/main stream owns LoRA compute,
        # - base stream owns Marlin compute,
        # - comm stream owns LoRA's rank all-reduce.
        # This lets LoRA's rank all-reduce overlap Marlin down compute without
        # allowing the two TP collectives to race each other: the base stream
        # waits for the LoRA all-reduce before launching its output all-reduce.
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        qlora_base_stream.wait_stream(current_stream)
        lora_a_output = down_lora_a_fn(hidden)
        with torch.cuda.stream(qlora_base_stream):
            output_parallel = down.quant_method.apply(down, hidden, None)

        if world_size > 1:
            qlora_comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(qlora_comm_stream):
                lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)

            current_stream.wait_stream(qlora_comm_stream)
            lora_output = down_lora_b_fn(lora_a_output)

            with torch.cuda.stream(qlora_base_stream):
                qlora_base_stream.wait_stream(qlora_comm_stream)
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
            current_stream.wait_stream(qlora_base_stream)
        else:
            lora_output = down_lora_b_fn(lora_a_output)
            current_stream.wait_stream(qlora_base_stream)
        return output_parallel + lora_output

    fn: Callable[[], torch.Tensor] = forward
    if not args.no_torch_compile:
        if args.kind == "qlora":
            # Dynamo-wrapping the multi-stream QLoRA wrapper works for direct
            # execution, but it is not CUDA Graph capture safe: stream/event
            # edges can be replayed through the legacy stream. Keep graph
            # profiling on the raw wrapper while the LoRA matmul stages remain
            # individually compiled above.
            if args.no_cuda_graph:
                fn = torch.compile(fn, fullgraph=False, backend="eager")
        else:
            fn = torch.compile(fn, fullgraph=False)

    reserve_context = (
        marlin_sm_reserve(args.two_stream_reserve_sms)
        if args.kind == "qlora"
        else nullcontext()
    )
    with reserve_context:
        flush_l2 = l2_flush_fn(args.l2_flush_mib, device)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(device)

        graph = None
        if not args.no_cuda_graph:
            graph, _ = capture_graph(fn, device)

        tp_cpu_barrier(world_size)
        if args.cuda_profiler_range and rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        tp_cpu_barrier(world_size)

        timings: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            flush_l2()
            start.record()
            if graph is not None:
                graph.replay()
            else:
                fn()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end) * 1000.0)

        tp_cpu_barrier(world_size)
        if args.cuda_profiler_range and rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
        tp_cpu_barrier(world_size)

    result = Result(
        tp=world_size,
        rank=rank,
        local_rank=local_rank,
        layer="up_down",
        kind=args.kind,
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        gate_up_local_output_features=getattr(gate_up, "output_size_per_partition", None),
        down_local_input_features=getattr(down, "input_size_per_partition", None),
        median_us=statistics.median(timings),
        min_us=min(timings),
        max_us=max(timings),
        warmup=args.warmup,
        iters=args.iters,
        cuda_graph=not args.no_cuda_graph,
        torch_compile=not args.no_torch_compile,
        l2_flush_mib=args.l2_flush_mib,
        uses_sglang_tp_linear=True,
        uses_merged_column_parallel_gate_up=True,
        uses_row_parallel_down=True,
        forced_nccl_all_reduce=args.force_nccl_all_reduce,
    )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote SGLang TP linear benchmark data to {args.output}", flush=True)

    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
