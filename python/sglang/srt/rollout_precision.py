"""Decode-only rollout precision switching policy.

This module is intentionally small and CPU-side.  The policy decision is made
before a forward batch reaches model layers; kernels only receive simple boolean
metadata through the existing LoRA batch info object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


ProjectionName = Literal["qkv", "o", "up", "down"]
PrecisionChoice = Literal["bf16_merged", "int4_torch_twostream", "int4_csgmv"]

PROJECTIONS: tuple[ProjectionName, ...] = ("qkv", "o", "up", "down")
VALID_CHOICES: set[str] = {"bf16_merged", "int4_torch_twostream", "int4_csgmv"}


@dataclass(frozen=True)
class RolloutPrecisionDecision:
    """Selected precision choice for one live-batch window."""

    enabled: bool
    source: str
    batch_start: Optional[int]
    batch_end: Optional[int]
    projections: dict[str, PrecisionChoice]
    speedup_vs_bf16: Optional[float] = None
    speedup_vs_csgmv: Optional[float] = None

    @classmethod
    def disabled(cls) -> "RolloutPrecisionDecision":
        return cls(
            enabled=False,
            source="disabled",
            batch_start=None,
            batch_end=None,
            projections={projection: "bf16_merged" for projection in PROJECTIONS},
        )

    def choice_for_projection(self, projection: Optional[str]) -> PrecisionChoice:
        if not projection:
            return "bf16_merged"
        return self.projections.get(projection, "bf16_merged")

    def use_int4_torch_twostream(self, projection: Optional[str]) -> bool:
        return self.choice_for_projection(projection) == "int4_torch_twostream"

    def use_int4(self, projection: Optional[str]) -> bool:
        return self.choice_for_projection(projection) in {
            "int4_torch_twostream",
            "int4_csgmv",
        }

    def use_bf16_merged(self, projection: Optional[str]) -> bool:
        return self.choice_for_projection(projection) == "bf16_merged"

    def uses_any_int4(self) -> bool:
        return any(
            choice in {"int4_torch_twostream", "int4_csgmv"}
            for choice in self.projections.values()
        )


@dataclass(frozen=True)
class RolloutPrecisionWindow:
    """One live-batch interval in the rollout policy."""

    batch_start: int
    batch_end: int
    projections: dict[str, PrecisionChoice]
    speedup_vs_bf16: Optional[float] = None
    speedup_vs_csgmv: Optional[float] = None

    def contains(self, batch_size: int) -> bool:
        high = max(self.batch_start, self.batch_end)
        low = min(self.batch_start, self.batch_end)
        return low < batch_size <= high


class RolloutPrecisionPolicy:
    """Decode-only precision switching policy loaded from JSON."""

    def __init__(
        self,
        windows: Iterable[RolloutPrecisionWindow],
        *,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        source: str = "manual",
    ):
        self.windows = tuple(
            sorted(windows, key=lambda window: window.batch_start, reverse=True)
        )
        self.model = model
        self.dataset = dataset
        self.source = source

    @classmethod
    def from_file(cls, path: str | Path) -> "RolloutPrecisionPolicy":
        policy_path = Path(path)
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        return cls.from_payload(payload, source=str(policy_path))

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, source: str = "payload"
    ) -> "RolloutPrecisionPolicy":
        raw_windows = payload.get("windows", payload.get("segments"))
        if not isinstance(raw_windows, list) or not raw_windows:
            raise ValueError("rollout precision policy requires a non-empty windows list")

        windows = []
        for idx, raw in enumerate(raw_windows):
            if "batch_start" not in raw or "batch_end" not in raw:
                raise ValueError(
                    f"policy window {idx} must define batch_start and batch_end"
                )

            raw_projections = raw.get("projections")
            if raw_projections is None:
                raw_projections = {
                    projection: raw.get(projection, "bf16_merged")
                    for projection in PROJECTIONS
                }
            if not isinstance(raw_projections, dict):
                raise ValueError(f"policy window {idx} projections must be a mapping")

            projections: dict[str, PrecisionChoice] = {}
            for projection in PROJECTIONS:
                choice = raw_projections.get(projection, "bf16_merged")
                choice = _normalize_choice(choice)
                projections[projection] = choice

            windows.append(
                RolloutPrecisionWindow(
                    batch_start=int(raw["batch_start"]),
                    batch_end=int(raw["batch_end"]),
                    projections=projections,
                    speedup_vs_bf16=_optional_float(raw.get("speedup_vs_bf16")),
                    speedup_vs_csgmv=_optional_float(raw.get("speedup_vs_csgmv")),
                )
            )

        return cls(
            windows=windows,
            model=payload.get("model"),
            dataset=payload.get("dataset"),
            source=source,
        )

    def select(self, batch_size: int, *, is_decode: bool) -> RolloutPrecisionDecision:
        if not is_decode:
            return RolloutPrecisionDecision.disabled()
        for window in self.windows:
            if window.contains(batch_size):
                return RolloutPrecisionDecision(
                    enabled=True,
                    source=self.source,
                    batch_start=window.batch_start,
                    batch_end=window.batch_end,
                    projections=window.projections,
                    speedup_vs_bf16=window.speedup_vs_bf16,
                    speedup_vs_csgmv=window.speedup_vs_csgmv,
                )
        return RolloutPrecisionDecision.disabled()


def load_rollout_precision_policy(
    path: Optional[str],
) -> Optional[RolloutPrecisionPolicy]:
    if not path:
        return None
    return RolloutPrecisionPolicy.from_file(path)


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


def _normalize_choice(choice: Any) -> PrecisionChoice:
    if not isinstance(choice, str):
        raise ValueError(f"precision choice must be a string, got {choice!r}")
    normalized = choice.lower().replace("-", "_").replace("+", "_")
    aliases = {
        "bf16": "bf16_merged",
        "bf16_merged_lora": "bf16_merged",
        "int4": "int4_torch_twostream",
        "int4_torch": "int4_torch_twostream",
        "torch_twostream": "int4_torch_twostream",
        "int4_torch2s": "int4_torch_twostream",
        "int4_bf16_lora_torch2s": "int4_torch_twostream",
        "int4_csgmv_lora": "int4_csgmv",
        "csgmv": "int4_csgmv",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_CHOICES:
        raise ValueError(
            f"unsupported precision choice {choice!r}; expected one of {sorted(VALID_CHOICES)}"
        )
    return normalized  # type: ignore[return-value]


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
