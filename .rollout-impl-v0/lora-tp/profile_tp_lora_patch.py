#!/usr/bin/env python3
"""Profile TP-shaped LoRA patch work for rollout precision mixing.

This intentionally profiles only the adapter patch math, not the INT4 Marlin
base projection or the full SGLang server. Launch with torchrun under nsys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=13824)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--profile",
        choices=("both", "gate_up", "down"),
        default="both",
        help="Which TP LoRA patch path to replay inside the profiler range.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Use eager torch matmul callables instead of torch.compile.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Replay eager callables directly instead of capturing CUDA graphs.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around the graph replay loop.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def nvtx_push(name: str) -> None:
    torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


def capture_graph(name: str, fn, local_rank: int):
    graph = torch.cuda.CUDAGraph()
    holder = {}
    torch.cuda.synchronize()
    barrier(local_rank)
    nvtx_push(f"capture:{name}")
    try:
        with torch.cuda.graph(graph):
            holder["out"] = fn()
    finally:
        nvtx_pop()
    torch.cuda.synchronize()
    barrier(local_rank)
    return graph, holder


def barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def log_rank0(rank_id: int, message: str) -> None:
    if rank_id == 0:
        print(message, flush=True)


def maybe_compile(fn, enabled: bool):
    if not enabled:
        return fn
    return torch.compile(fn, fullgraph=False)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank_id = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise RuntimeError("WORLD_SIZE must be positive")
    if args.intermediate_size % world_size != 0:
        raise RuntimeError(
            f"intermediate_size={args.intermediate_size} is not divisible by "
            f"world_size={world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)
    torch.manual_seed(1000 + rank_id)
    log_rank0(rank_id, f"initialized TP world_size={world_size}")

    batch = args.batch_size
    lora_rank = args.rank
    scaling = args.lora_alpha / args.rank
    hidden = args.hidden_size
    intermediate_per_rank = args.intermediate_size // world_size

    # Column-parallel gate/up: A is replicated, B is output-sharded per TP rank.
    gate_up_x = torch.randn((batch, hidden), device=device, dtype=dtype).contiguous()
    gate_up_a = torch.randn(
        (2 * lora_rank, hidden), device=device, dtype=dtype
    ).contiguous()
    gate_up_b = torch.randn(
        (2 * intermediate_per_rank, lora_rank), device=device, dtype=dtype
    ).contiguous()

    def gate_up_patch_impl():
        a_out = torch.matmul(gate_up_x, gate_up_a.transpose(0, 1))
        gate = torch.matmul(
            a_out[:, :lora_rank],
            gate_up_b[:intermediate_per_rank, :].transpose(0, 1),
        )
        up = torch.matmul(
            a_out[:, lora_rank : 2 * lora_rank],
            gate_up_b[intermediate_per_rank:, :].transpose(0, 1),
        )
        return torch.cat((gate, up), dim=-1) * scaling

    # Row-parallel down: A is input-sharded, B is replicated. The LoRA-A output
    # must be summed across TP ranks before applying B.
    down_x = torch.randn(
        (batch, intermediate_per_rank), device=device, dtype=dtype
    ).contiguous()
    down_a = torch.randn(
        (lora_rank, intermediate_per_rank), device=device, dtype=dtype
    ).contiguous()
    down_b = torch.randn((hidden, lora_rank), device=device, dtype=dtype).contiguous()

    def down_a_impl():
        return torch.matmul(down_x, down_a.transpose(0, 1))

    def down_b_impl(a_out):
        return torch.matmul(a_out, down_b.transpose(0, 1)) * scaling

    gate_up_patch = maybe_compile(gate_up_patch_impl, not args.no_compile)
    down_a_patch = maybe_compile(down_a_impl, not args.no_compile)
    down_b_patch = maybe_compile(down_b_impl, not args.no_compile)

    def down_patch_impl():
        a_out = down_a_patch()
        dist.all_reduce(a_out, op=dist.ReduceOp.SUM)
        return down_b_patch(a_out)

    log_rank0(rank_id, "warming up callables")
    for _ in range(args.warmup):
        if args.profile in ("both", "gate_up"):
            gate_up_patch()
        if args.profile in ("both", "down"):
            down_patch_impl()
    torch.cuda.synchronize()
    barrier(local_rank)

    graphs = []
    holders = []
    replays = []
    if args.no_cuda_graph:
        if args.profile in ("both", "gate_up"):
            replays.append(("tp4_gate_up_col_local_lora_patch", gate_up_patch))
        if args.profile in ("both", "down"):
            replays.append(("tp4_down_row_lora_patch_allreduce", down_patch_impl))
    else:
        if args.profile in ("both", "gate_up"):
            log_rank0(rank_id, "capturing gate_up graph")
            graph, holder = capture_graph(
                "tp4_gate_up_col_local_lora_patch", gate_up_patch, local_rank
            )
            graphs.append(("tp4_gate_up_col_local_lora_patch", graph))
            holders.append(holder)
        if args.profile in ("both", "down"):
            log_rank0(rank_id, "capturing down graph")
            graph, holder = capture_graph(
                "tp4_down_row_lora_patch_allreduce", down_patch_impl, local_rank
            )
            graphs.append(("tp4_down_row_lora_patch_allreduce", graph))
            holders.append(holder)
        replays = [(name, graph.replay) for name, graph in graphs]

    log_rank0(rank_id, "warming up replay loop")
    for _ in range(args.warmup):
        for _, replay in replays:
            replay()
    torch.cuda.synchronize()
    barrier(local_rank)

    log_rank0(rank_id, "entering profiler range")
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iters):
        for name, replay in replays:
            nvtx_push(f"replay:{name}:iter={i}:rank={rank_id}")
            try:
                replay()
            finally:
                nvtx_pop()
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    barrier(local_rank)
    log_rank0(rank_id, "finished profiler range")

    if args.output_json is not None and rank_id == 0:
        payload = {
            "profile": args.profile,
            "world_size": world_size,
            "batch_size": batch,
            "dtype": "bf16",
            "compiled": not args.no_compile,
            "cuda_graph": not args.no_cuda_graph,
            "warmup": args.warmup,
            "iters": args.iters,
            "hidden_size": hidden,
            "intermediate_size": args.intermediate_size,
            "intermediate_per_tp_rank": intermediate_per_rank,
            "lora_rank": lora_rank,
            "lora_alpha": args.lora_alpha,
            "paths": {
                "gate_up": {
                    "tp_type": "column_parallel",
                    "x": [batch, hidden],
                    "A_local": [2 * lora_rank, hidden],
                    "B_local": [2 * intermediate_per_rank, lora_rank],
                    "math": "cat((x @ A_gate.T) @ B_gate_local.T, (x @ A_up.T) @ B_up_local.T)",
                    "collective": None,
                },
                "down": {
                    "tp_type": "row_parallel",
                    "x_local": [batch, intermediate_per_rank],
                    "A_local": [lora_rank, intermediate_per_rank],
                    "B_replicated": [hidden, lora_rank],
                    "math": "all_reduce(x_local @ A_local.T) @ B_replicated.T",
                    "collective": "ncclAllReduce on [batch_size, lora_rank]",
                },
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    # Keep holders live until after graph replay.
    if not args.no_cuda_graph and not holders:
        raise RuntimeError("No graphs were captured")
    log_rank0(rank_id, "force exiting after profile to avoid NCCL graph teardown wait")
    os._exit(0)


if __name__ == "__main__":
    main()
