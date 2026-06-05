"""Create synthetic rank-16 PEFT LoRA adapters for Qwen2.5 14B/32B.

The adapter is intended for performance measurement of the LoRA serving path.
Weights are bf16 zeros so the adapter does not intentionally perturb generation,
but SGLang still loads the adapter and executes the LoRA patch kernels for
requests that specify the adapter name.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file


# NOTE: qkv_proj is intentionally excluded. The separate-stream LoRA patch is
# only applied to the o/gate/up/down projections (see run_rollout_lora_forward_
# experiments.TARGET_MODULES); the adapter must therefore not carry qkv weights,
# or the server rejects it for a target-module mismatch.
TARGET_MODULES = [
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class Shape:
    layers: int
    hidden: int
    kv_out: int
    intermediate: int


SHAPES = {
    "qwen2.5-14b": Shape(layers=48, hidden=5120, kv_out=1024, intermediate=13824),
    "qwen2.5-32b": Shape(layers=64, hidden=5120, kv_out=1024, intermediate=27648),
}


def add_lora_pair(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    in_dim: int,
    out_dim: int,
    rank: int,
    dtype: torch.dtype,
) -> None:
    tensors[f"{prefix}.lora_A.weight"] = torch.zeros((rank, in_dim), dtype=dtype)
    tensors[f"{prefix}.lora_B.weight"] = torch.zeros((out_dim, rank), dtype=dtype)


def create_adapter(model_key: str, output_dir: Path, rank: int, dtype: torch.dtype) -> None:
    shape = SHAPES[model_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}

    # (module suffix, in_dim, out_dim) for every projection we may emit. Only
    # the modules listed in TARGET_MODULES are actually written, so excluding
    # q/k/v here keeps qkv_proj out of the adapter.
    proj_specs = [
        ("q_proj", "self_attn", shape.hidden, shape.hidden),
        ("k_proj", "self_attn", shape.hidden, shape.kv_out),
        ("v_proj", "self_attn", shape.hidden, shape.kv_out),
        ("o_proj", "self_attn", shape.hidden, shape.hidden),
        ("gate_proj", "mlp", shape.hidden, shape.intermediate),
        ("up_proj", "mlp", shape.hidden, shape.intermediate),
        ("down_proj", "mlp", shape.intermediate, shape.hidden),
    ]
    for layer in range(shape.layers):
        for proj, block, in_dim, out_dim in proj_specs:
            if proj not in TARGET_MODULES:
                continue
            prefix = f"base_model.model.model.layers.{layer}.{block}.{proj}"
            add_lora_pair(tensors, prefix, in_dim, out_dim, rank, dtype)

    adapter_config = {
        "base_model_name_or_path": "",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": TARGET_MODULES,
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
    )
    save_file(tensors, str(output_dir / "adapter_model.safetensors"))
    metadata = {
        "model_key": model_key,
        "rank": rank,
        "dtype": str(dtype),
        "num_tensors": len(tensors),
        "num_params": sum(t.numel() for t in tensors.values()),
        "target_modules": TARGET_MODULES,
    }
    (output_dir / "synthetic_adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metadata['num_params']} parameters to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(SHAPES), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    create_adapter(args.model_key, Path(args.output_dir), args.rank, dtype)


if __name__ == "__main__":
    main()
