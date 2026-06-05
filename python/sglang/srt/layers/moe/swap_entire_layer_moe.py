"""Swap-entire-layer MoE: per-layer binary BF16/INT4 swap by token threshold.

Companion to ``heter_moe.py``. Where ``HeterFusedMoE`` runs mixed precision
per forward (top-k experts in BF16, rest in INT4, two kernels summed),
``apply_swap_entire_layer_precision`` configures HeterFusedMoE with a
binary promotion lookup so each forward launches exactly one kernel:

  * num_tokens >= ``swap_threshold``  →  ``bf16_full`` (one BF16 kernel)
  * num_tokens <  ``swap_threshold``  →  ``int4_only`` (one Marlin kernel)

Applied only to the first ``swap_first_n`` MoE layers (default: half). The
remaining layers become all-INT4 (HeterFusedMoE with int4_only_experts =
every expert), so decode runs INT4-only end-to-end and BF16 weights are
loaded only where they get used.

Rationale: large-batch prefill is compute-bound where dequant overhead in
Marlin INT4 hurts; BF16 wins. Small-batch decode is memory-bound where
INT4 wins. The "mixed precision per layer" path always pays a 2-kernel
launch + reduction tax, even when one precision dominates. The swap-layer
strategy spends that tax only on layer-boundary precision changes (which
happen once per forward batch, not per layer per kernel).
"""

from __future__ import annotations

import gc
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.moe.heter_moe import (
    HeterFusedMoE,
    _parse_heter_config,
    _swap_attention_to_int4,
)

logger = logging.getLogger(__name__)

# Sentinel value the loader uses to identify a swap-entire-layer config.
SWAP_ENTIRE_LAYER_MODE = "swap_entire_layer"


# (group_ids, group_weights), one tuple per group. Same shape contract as
# EfficiencyPromotionPolicy.dispatch — the layer's forward expects this.
GroupDispatchTuple = Tuple[torch.Tensor, torch.Tensor]


class SwapEntireLayerPolicy:
    """Layer-level binary precision dispatch.

    Replaces ``EfficiencyPromotionPolicy`` for swap layers. The expert-
    level policy spends per-forward time on:

      * scatter_add token counts across experts
      * topk over those counts (when 0 < n_active < num_experts)
      * gather + ``torch.where`` to build per-group ``(ids, weights)``

    None of that is needed when the precision decision is per-layer
    (``num_tokens >= swap_threshold`` → all-BF16, else all-INT4): the
    "active" group gets the original ``(topk_ids, topk_weights)`` and
    the inactive group gets a sentinel-filled stub.

    The stub buffers are zero-initialised and shared across forward
    calls (their values never need to change — the layer's forward()
    short-circuits the inactive group on ``decision.n_active`` so the
    stubs are never actually consumed).
    """

    def __init__(
        self,
        num_experts: int,
        num_groups: int,
        device: Optional[torch.device] = None,
        threshold: int = 1152,
        int4_group_idx: int = 0,
        bf16_group_idx: int = 1,
    ):
        self._num_experts = num_experts
        self._num_groups = num_groups
        self._device = device or torch.device("cuda")
        self._threshold = threshold
        self._int4_group_idx = int4_group_idx
        self._bf16_group_idx = bf16_group_idx
        # Lazily-grown sentinel/zero buffers for the inactive group's
        # stub tensors. Avoids per-call allocation. Reallocated once
        # when num_tokens grows past the current capacity.
        self._stub_ids: Optional[torch.Tensor] = None
        self._stub_weights: Optional[torch.Tensor] = None

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_groups(self) -> int:
        return self._num_groups

    def _get_stubs(
        self,
        ids: torch.Tensor,
        weights: torch.Tensor,
        sentinel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._stub_ids is None
            or self._stub_ids.shape != ids.shape
            or self._stub_ids.dtype != ids.dtype
        ):
            self._stub_ids = torch.full_like(ids, sentinel)
        if (
            self._stub_weights is None
            or self._stub_weights.shape != weights.shape
            or self._stub_weights.dtype != weights.dtype
        ):
            self._stub_weights = torch.zeros_like(weights)
        return self._stub_ids, self._stub_weights

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        n_active: int,  # ignored — kept for interface compat with policy contract
        sentinel: int = -1,
    ) -> List[GroupDispatchTuple]:
        # The MoE layer drives the precision decision via
        # ``_select_promotion_decision`` (binary lookup) → forward
        # passes ``decision.n_active`` here. We honour that signal so
        # tests that monkeypatch ``_select_promotion_decision`` keep
        # working: n_active >= num_experts → BF16; else INT4.
        use_bf16 = n_active >= self._num_experts

        stub_ids, stub_w = self._get_stubs(
            token_selected_experts, token_final_scales, sentinel
        )

        results: List[GroupDispatchTuple] = []
        for group_idx in range(self._num_groups):
            active = (
                (group_idx == self._bf16_group_idx and use_bf16)
                or (group_idx == self._int4_group_idx and not use_bf16)
            )
            if active:
                results.append((token_selected_experts, token_final_scales))
            else:
                results.append((stub_ids, stub_w))
        return results


def _make_binary_promotion_lookup(threshold: int) -> Dict[str, Dict[str, Any]]:
    """Build a binary int4_only / bf16_full lookup centered on ``threshold``.

    The runtime ``_select_promotion_decision`` picks the nearest
    ``bs<N>`` row by absolute distance, ties going to the smaller key. We
    place one ``int4_only`` row at threshold-1 and one ``bf16_full`` row
    at threshold so the crossover is sharp at exactly ``threshold``.
    """
    if threshold < 2:
        raise ValueError(
            f"swap_threshold must be >= 2 (got {threshold}); "
            "decode batches need a sub-threshold key to map to int4_only."
        )
    return {
        "bs1": {"_overall_best": "int4_only"},
        f"bs{threshold - 1}": {"_overall_best": "int4_only"},
        f"bs{threshold}": {"_overall_best": "bf16_full"},
        "bs131072": {"_overall_best": "bf16_full"},
    }


def _make_dummy_bf16_cells() -> Dict[str, Dict[str, Dict[str, int]]]:
    """Minimal valid bf16-cells dict.

    bf16_full and int4_only rows don't reference cell_keys, so the parent
    only needs the file to parse. We still write a syntactically valid
    cell so HeterFusedMoE._validate_promotion_lookup_cells doesn't trip
    on a stray future row.
    """
    cell = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    return {"_unused_cell": {"up": cell, "down": cell}}


def _materialize_swap_lookup_files(
    threshold: int, target_dir: Path
) -> tuple[Path, Path]:
    """Write the synthetic lookup + bf16 cells to ``target_dir``.

    Returns (promotion_lookup_path, bf16_cells_path). Caller owns cleanup.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    lookup_path = target_dir / "swap_promotion_lookup.json"
    cells_path = target_dir / "swap_bf16_cells.json"
    with lookup_path.open("w") as f:
        json.dump(_make_binary_promotion_lookup(threshold), f, indent=2)
    with cells_path.open("w") as f:
        json.dump(_make_dummy_bf16_cells(), f, indent=2)
    return lookup_path, cells_path


def _validate_swap_config(cfg: Dict[str, Any], num_layers: int) -> Dict[str, Any]:
    """Validate swap-entire-layer fields and return a normalized copy."""
    out = dict(cfg)

    swap_first_n = out.get("swap_first_n")
    swap_first_fraction = out.get("swap_first_fraction")
    if swap_first_n is None and swap_first_fraction is None:
        swap_first_fraction = 0.5
    if swap_first_n is None:
        if not (isinstance(swap_first_fraction, (int, float))
                and 0.0 < swap_first_fraction <= 1.0):
            raise ValueError(
                f"swap_first_fraction must be in (0, 1], got {swap_first_fraction!r}"
            )
        swap_first_n = max(1, int(round(num_layers * float(swap_first_fraction))))
    if not (isinstance(swap_first_n, int) and 1 <= swap_first_n <= num_layers):
        raise ValueError(
            f"swap_first_n={swap_first_n} must be in [1, {num_layers}]"
        )
    out["swap_first_n"] = swap_first_n

    threshold = out.get("swap_threshold", 256)
    if not (isinstance(threshold, int) and threshold >= 2):
        raise ValueError(
            f"swap_threshold must be int >= 2, got {threshold!r}"
        )
    out["swap_threshold"] = threshold

    # Default: non-swap layers go all-INT4 (decode-fast, VRAM-light).
    # Set to False to leave them BF16 (test-only; does not match the
    # design intent of the method).
    out.setdefault("non_swap_int4_only", True)
    if not isinstance(out["non_swap_int4_only"], bool):
        raise ValueError("non_swap_int4_only must be a boolean")

    return out


def apply_swap_entire_layer_precision(
    model: nn.Module,
    config_path: str,
    device: torch.device,
) -> None:
    """Apply swap-entire-layer precision to ``model``'s MoE stack.

    For each MoE layer in [0, swap_first_n):
      - HeterFusedMoE with BF16+INT4 weights for every expert
      - synthetic binary promotion_lookup (int4_only below threshold,
        bf16_full at/above) so forward launches one kernel
    For each MoE layer in [swap_first_n, num_layers):
      - if non_swap_int4_only=True (default): HeterFusedMoE with
        int4_only_experts = every expert  (no BF16 weights loaded)
      - else: leave as the original FusedMoE (BF16-only)

    Optionally swaps attention to INT4 if heter_config sets
    attention_num_bits=4 (mirrors apply_heter_precision).
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    raw_cfg = _parse_heter_config(config_path)

    # Find INT4 group's checkpoint path (required: we always load INT4).
    int4_checkpoint = None
    for gcfg in raw_cfg["groups"]:
        if gcfg.get("num_bits", 16) == 4:
            int4_checkpoint = gcfg["checkpoint"]
            break
    if int4_checkpoint is None:
        raise ValueError(
            "swap-entire-layer requires an INT4 group with a 'checkpoint' "
            "entry in groups[]."
        )

    layers_module = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_module = model.model.layers
    elif hasattr(model, "layers"):
        layers_module = model.layers
    if layers_module is None:
        raise ValueError("Cannot find model layers for swap-entire-layer apply")

    # Count MoE layers up front (some models have non-MoE layers in the
    # stack; we count only those that match HeterFusedMoE's targeting).
    moe_layer_indices: List[int] = []
    for layer_id, layer in enumerate(layers_module):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            if isinstance(layer.mlp.experts, FusedMoE):
                moe_layer_indices.append(layer_id)
    num_moe = len(moe_layer_indices)
    if num_moe == 0:
        logger.warning("apply_swap_entire_layer_precision: no MoE layers found")
        return

    cfg = _validate_swap_config(raw_cfg, num_moe)
    swap_first_n = cfg["swap_first_n"]
    swap_threshold = cfg["swap_threshold"]
    non_swap_int4_only = cfg["non_swap_int4_only"]

    # Materialize synthetic lookup files in a stable location next to
    # config_path so logs and reproducible reruns can find them. Falls
    # back to the system tempdir if the config dir isn't writable.
    cfg_dir = Path(config_path).resolve().parent
    try:
        lookup_path, cells_path = _materialize_swap_lookup_files(
            swap_threshold, cfg_dir / ".swap_runtime"
        )
    except OSError:
        td = Path(tempfile.mkdtemp(prefix="swap_entire_layer_"))
        lookup_path, cells_path = _materialize_swap_lookup_files(
            swap_threshold, td
        )

    # Build the heter_config we'll hand to HeterFusedMoE. Override the
    # lookup paths so the parent's loader picks up our binary lookup.
    heter_for_moe = dict(raw_cfg)
    heter_for_moe["promotion_lookup_path"] = str(lookup_path)
    heter_for_moe["bf16_config_path"] = str(cells_path)
    # We inject _int4_only_by_layer per-layer below; clear any inherited
    # int4_only_experts_file mapping so it doesn't double-apply.
    heter_for_moe["_int4_only_by_layer"] = {}
    heter_for_moe["_bf16_only_by_layer"] = {}

    # Swap layer indices: positions in moe_layer_indices, not absolute
    # layer_id. (Almost always identical for dense MoE stacks, but the
    # distinction matters if non-MoE layers appear interleaved.)
    swap_layer_ids = set(moe_layer_indices[:swap_first_n])
    non_swap_layer_ids = set(moe_layer_indices[swap_first_n:])

    num_swap = 0
    num_non_swap = 0
    for layer_id, layer in enumerate(layers_module):
        if layer_id not in swap_layer_ids and layer_id not in non_swap_layer_ids:
            continue

        fused_moe = layer.mlp.experts
        if not isinstance(fused_moe, FusedMoE):
            continue

        if layer_id in swap_layer_ids:
            # All-experts heter (BF16+INT4); binary lookup → one kernel
            # per forward.
            heter_moe = HeterFusedMoE.from_fused_moe(
                fused_moe,
                heter_for_moe,
                int4_only_experts=None,
                bf16_only_experts=None,
                layer_id=layer_id,
            )
            heter_moe.load_int4_weights(int4_checkpoint, layer_id)
            heter_moe.repack_int4_to_marlin()
            # Replace the expert-level policy with the layer-level one.
            # No per-expert top-k, no slot-group dispatch — the binary
            # precision choice is made entirely in the lookup. Saves
            # the per-forward dispatch tax in CUDA-graph-friendly steps.
            heter_moe.policy = SwapEntireLayerPolicy(
                num_experts=heter_moe.num_experts,
                num_groups=heter_moe.num_groups,
                device=heter_moe.device,
                threshold=swap_threshold,
                int4_group_idx=heter_moe._int4_group_idx,
                bf16_group_idx=heter_moe._bf16_group_idx,
            )
            num_swap += 1
        else:
            if not non_swap_int4_only:
                continue  # leave as BF16 FusedMoE
            # All experts INT4-only: no BF16 weights kept, fast decode.
            E = fused_moe.w13_weight.shape[0]
            int4_only_experts = list(range(E))
            heter_moe = HeterFusedMoE.from_fused_moe(
                fused_moe,
                heter_for_moe,
                int4_only_experts=int4_only_experts,
                bf16_only_experts=None,
                layer_id=layer_id,
            )
            heter_moe.load_int4_weights(int4_checkpoint, layer_id)
            heter_moe.repack_int4_to_marlin()
            num_non_swap += 1

        # Replace and free the original to keep peak VRAM flat (mirrors
        # apply_heter_precision).
        layer.mlp.experts = heter_moe
        del fused_moe
        gc.collect()
        torch.cuda.empty_cache()

        if layer_id % 10 == 0:
            kind = "swap-heter" if layer_id in swap_layer_ids else "all-int4"
            logger.info(f"swap-entire-layer: layer {layer_id} → {kind}")

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        f"swap-entire-layer applied: swap_first_n={swap_first_n} "
        f"(threshold={swap_threshold}), {num_swap} heter swap layers + "
        f"{num_non_swap} all-INT4 layers"
    )

    if raw_cfg.get("attention_num_bits", 16) == 4:
        _swap_attention_to_int4(model, int4_checkpoint, device)
