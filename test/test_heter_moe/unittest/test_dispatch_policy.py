"""Dispatch tests for the single efficiency-promotion policy."""

import importlib.util
import sys
from pathlib import Path

import torch

_PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"


def _import_module_from_file(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(module_name, _PYTHON_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_policy_mod = _import_module_from_file(
    "heter_policy", "sglang/srt/layers/moe/heter_policy.py"
)
EfficiencyPromotionPolicy = _policy_mod.EfficiencyPromotionPolicy


def _make_policy(
    num_experts=4,
    int4_only_mask=None,
    bf16_only_mask=None,
):
    return EfficiencyPromotionPolicy(
        num_experts=num_experts,
        num_groups=2,
        device=torch.device("cpu"),
        int4_only_mask=int4_only_mask,
        int4_group_idx=0,
        bf16_only_mask=bf16_only_mask,
        bf16_group_idx=1,
    )


def _appearances(dispatches):
    return sum((weights != 0).long() for _, weights in dispatches)


def test_n_active_zero_routes_valid_slots_to_int4():
    policy = _make_policy()
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)

    dispatches = policy.dispatch(ids, weights, n_active=0, sentinel=-1)
    int4_ids, int4_weights = dispatches[0]
    bf16_ids, bf16_weights = dispatches[1]

    assert torch.equal(int4_ids, ids)
    assert torch.equal(int4_weights, weights)
    assert (bf16_ids == -1).all()
    assert (bf16_weights == 0).all()


def test_partial_promotion_uses_top_routed_counts_with_duplicates():
    policy = _make_policy()
    ids = torch.tensor(
        [[0, 2], [1, 2], [2, 3], [0, -1]],
        dtype=torch.long,
    )
    weights = torch.ones_like(ids, dtype=torch.float32)

    dispatches = policy.dispatch(ids, weights, n_active=1, sentinel=-1)
    int4_ids, _ = dispatches[0]
    bf16_ids, _ = dispatches[1]

    expert2 = ids == 2
    valid_not2 = (ids >= 0) & (ids != 2)
    assert (bf16_ids[expert2] == 2).all()
    assert (int4_ids[expert2] == -1).all()
    assert torch.equal(int4_ids[valid_not2], ids[valid_not2])
    assert (bf16_ids[valid_not2] == -1).all()


def test_full_promotion_routes_all_valid_slots_to_bf16():
    policy = _make_policy()
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)

    dispatches = policy.dispatch(ids, weights, n_active=99, sentinel=-1)
    int4_ids, int4_weights = dispatches[0]
    bf16_ids, bf16_weights = dispatches[1]

    assert (int4_ids == -1).all()
    assert (int4_weights == 0).all()
    assert torch.equal(bf16_ids, ids)
    assert torch.equal(bf16_weights, weights)


def test_forced_precision_masks_are_final_word():
    int4_mask = torch.tensor([False, False, True, False])
    bf16_mask = torch.tensor([False, True, False, False])
    policy = _make_policy(int4_only_mask=int4_mask, bf16_only_mask=bf16_mask)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)

    dispatches = policy.dispatch(ids, weights, n_active=4, sentinel=-1)
    int4_ids, _ = dispatches[0]
    bf16_ids, _ = dispatches[1]

    assert int4_ids[ids == 2].eq(2).all()
    assert bf16_ids[ids == 2].eq(-1).all()
    assert bf16_ids[ids == 1].eq(1).all()
    assert int4_ids[ids == 1].eq(-1).all()


def test_invalid_expert_ids_are_sentineled_and_zero_weighted():
    policy = _make_policy()
    ids = torch.tensor([[-1, 0], [4, 1]], dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)

    dispatches = policy.dispatch(ids, weights, n_active=2, sentinel=4)
    valid = (ids >= 0) & (ids < 4)
    appearances = _appearances(dispatches)

    assert (appearances[valid] == 1).all()
    assert (appearances[~valid] == 0).all()
    for group_ids, group_weights in dispatches:
        assert (group_ids[~valid] == 4).all()
        assert (group_weights[~valid] == 0).all()
