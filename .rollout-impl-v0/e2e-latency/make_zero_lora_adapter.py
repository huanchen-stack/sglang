#!/usr/bin/env python3
"""Create a zero-valued Qwen2-style LoRA adapter for rollout experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype={name}") from exc


def build_state_dict(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    dtype = parse_dtype(args.dtype)
    head_dim = args.hidden_size // args.num_attention_heads
    kv_out = args.num_key_value_heads * head_dim

    projection_shapes = {
        "self_attn.q_proj": (args.hidden_size, args.hidden_size),
        "self_attn.k_proj": (args.hidden_size, kv_out),
        "self_attn.v_proj": (args.hidden_size, kv_out),
        "self_attn.o_proj": (args.hidden_size, args.hidden_size),
        "mlp.gate_proj": (args.hidden_size, args.intermediate_size),
        "mlp.up_proj": (args.hidden_size, args.intermediate_size),
        "mlp.down_proj": (args.intermediate_size, args.hidden_size),
    }

    state: dict[str, torch.Tensor] = {}
    for layer_idx in range(args.num_hidden_layers):
        for module_name, (in_features, out_features) in projection_shapes.items():
            prefix = f"base_model.model.model.layers.{layer_idx}.{module_name}"
            state[f"{prefix}.lora_A.weight"] = torch.zeros(
                (args.rank, in_features), dtype=dtype
            )
            state[f"{prefix}.lora_B.weight"] = torch.zeros(
                (out_features, args.rank), dtype=dtype
            )
    return state


def write_adapter_config(args: argparse.Namespace, out_dir: Path) -> None:
    payload = {
        "base_model_name_or_path": "",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": args.rank,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": args.rank,
        "target_modules": TARGET_MODULES,
        "task_type": "CAUSAL_LM",
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_metadata(args: argparse.Namespace, state: dict[str, torch.Tensor], out_dir: Path) -> None:
    payload = {
        "model_key": args.model_key,
        "rank": args.rank,
        "dtype": str(parse_dtype(args.dtype)),
        "num_tensors": len(state),
        "num_params": sum(t.numel() for t in state.values()),
        "target_modules": TARGET_MODULES,
        "num_hidden_layers": args.num_hidden_layers,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
    }
    (out_dir / "synthetic_adapter_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--num-hidden-layers", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--num-attention-heads", type=int, required=True)
    parser.add_argument("--num-key-value-heads", type=int, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_state_dict(args)
    write_adapter_config(args, out_dir)
    save_file(state, str(out_dir / "adapter_model.safetensors"))
    write_metadata(args, state, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
