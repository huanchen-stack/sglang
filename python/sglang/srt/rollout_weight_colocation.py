"""Helpers for rollout weight colocation.

The v0 colocation mode keeps two model copies resident:

* the primary BF16 model, used for prefill
* an INT4 shadow model, used as the decode base under Torch-native LoRA

This module stays intentionally small.  It only maps model module names to the
projection families that participate in the v0 decode swap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple


PROJECTIONS: Tuple[str, ...] = ("qkv", "o", "up", "down")
BF16_PRECISION = "bf16"
INT4_TORCH2S_PRECISION = "int4_torch2s"
SUPPORTED_PRECISIONS: Tuple[str, ...] = (BF16_PRECISION, INT4_TORCH2S_PRECISION)


def projection_from_module_name(module_name: Optional[str]) -> Optional[str]:
    if not module_name:
        return None

    leaf = module_name.split(".")[-1]
    if leaf in {"qkv_proj", "query_key_value", "W_pack"}:
        return "qkv"
    if leaf in {"o_proj", "out_proj", "dense"}:
        return "o"
    if leaf in {"gate_up_proj", "up_proj", "gate_proj", "w1", "w3"}:
        return "up"
    if leaf in {"down_proj", "w2"}:
        return "down"

    if "qkv" in leaf:
        return "qkv"
    if leaf.startswith("o_") or leaf.endswith("_o_proj"):
        return "o"
    if "gate_up" in leaf or "up" in leaf or "gate" in leaf:
        return "up"
    if "down" in leaf:
        return "down"

    return None


def _normalize_precision(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "")
    aliases = {
        "bf16": BF16_PRECISION,
        "bfloat16": BF16_PRECISION,
        "int4": INT4_TORCH2S_PRECISION,
        "int4_torch2s": INT4_TORCH2S_PRECISION,
        "int4+torch2s": INT4_TORCH2S_PRECISION,
        "int4+qloratorch2s": INT4_TORCH2S_PRECISION,
        "int4+qlora_torch2s": INT4_TORCH2S_PRECISION,
        "int4_qlora_torch2s": INT4_TORCH2S_PRECISION,
        "int4+torch_native_two_stream": INT4_TORCH2S_PRECISION,
        "int4_torch_native_two_stream": INT4_TORCH2S_PRECISION,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rollout precision '{value}'. Supported values are "
            f"{', '.join(SUPPORTED_PRECISIONS)}."
        ) from exc


def _parse_batch_key(key: str) -> Iterable[int]:
    key = str(key).strip()
    if "-" in key:
        start_s, end_s = key.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if start > end:
            raise ValueError(f"Invalid rollout precision batch range '{key}'.")
        return range(start, end + 1)
    return (int(key),)


class RolloutPrecisionPolicy:
    """Per-CUDA-graph-batch decode precision policy."""

    def __init__(self, by_batch_size: Mapping[int, Mapping[str, str]]):
        self._by_batch_size: Dict[int, Dict[str, str]] = {
            int(bs): dict(policy) for bs, policy in by_batch_size.items()
        }

    def precision_for(self, batch_size: int, projection: str) -> str:
        if projection not in PROJECTIONS:
            raise ValueError(f"Unknown rollout projection '{projection}'.")
        try:
            policy = self._by_batch_size[int(batch_size)]
        except KeyError as exc:
            raise KeyError(
                f"No rollout precision policy for CUDA graph batch size {batch_size}."
            ) from exc
        try:
            return policy[projection]
        except KeyError as exc:
            raise KeyError(
                f"No rollout precision configured for projection '{projection}' "
                f"at CUDA graph batch size {batch_size}."
            ) from exc

    @classmethod
    def from_file(cls, path: str) -> "RolloutPrecisionPolicy":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Rollout precision policy must be a JSON object.")

        raw_policies = data.get("batch_sizes", data)
        if not isinstance(raw_policies, dict):
            raise ValueError(
                "Rollout precision policy must be keyed by batch size or contain "
                "a 'batch_sizes' object."
            )

        by_batch_size: Dict[int, Dict[str, str]] = {}
        for raw_batch_key, raw_policy in raw_policies.items():
            if not isinstance(raw_policy, dict):
                raise ValueError(
                    f"Rollout precision entry for '{raw_batch_key}' must be an object."
                )
            missing = [
                projection for projection in PROJECTIONS if projection not in raw_policy
            ]
            if missing:
                raise ValueError(
                    f"Rollout precision entry for '{raw_batch_key}' is missing "
                    f"projection(s): {', '.join(missing)}."
                )
            policy = {
                projection: _normalize_precision(str(raw_policy[projection]))
                for projection in PROJECTIONS
            }
            for batch_size in _parse_batch_key(raw_batch_key):
                if batch_size in by_batch_size:
                    raise ValueError(
                        f"Duplicate rollout precision policy for batch size {batch_size}."
                    )
                by_batch_size[batch_size] = dict(policy)

        return cls(by_batch_size)
