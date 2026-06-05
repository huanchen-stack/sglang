"""Runtime lookup selection for HeterFusedMoE efficiency promotion."""

import json

import pytest
import torch


def _write_json(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def _cell_config(block_m=32):
    return {
        "up": {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        },
        "down": {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        },
    }


def _make_layer(tmp_path, lookup):
    from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

    lookup_path = _write_json(tmp_path / "promotion_lookup.json", lookup)
    bf16_path = _write_json(
        tmp_path / "bf16_cells.json",
        {"n2_mpe8": _cell_config()},
    )
    cfg = {
        "groups": [
            {"name": "int4", "num_bits": 4, "group_size": 128},
            {"name": "bf16", "num_bits": 16},
        ],
        "promotion_lookup_path": lookup_path,
        "bf16_config_path": bf16_path,
    }
    return HeterFusedMoE(
        num_experts=4,
        hidden_size=128,
        intermediate_size=128,
        top_k=2,
        heter_config=cfg,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def test_selects_exact_and_nearest_promotion_rows(tmp_path):
    layer = _make_layer(
        tmp_path,
        {
            "bs4": {"_overall_best": "int4_only"},
            "bs8": {
                "_overall_best": "promotion",
                "n_active": 2,
                "cell_key": "n2_mpe8",
            },
            "bs16": {"_overall_best": "bf16_full"},
        },
    )

    exact = layer._select_promotion_decision(8)
    nearest = layer._select_promotion_decision(7)
    full = layer._select_promotion_decision(16)

    assert exact.mode == "promotion"
    assert exact.n_active == 2
    assert exact.cell_key == "n2_mpe8"
    assert exact.bf16_up_config["BLOCK_SIZE_M"] == 32
    assert nearest.batch_size_key == 8
    assert nearest.n_active == 2
    assert full.mode == "bf16_full"
    assert full.n_active == layer.num_experts
    assert full.bf16_up_config is None


def test_missing_promotion_lookup_file_fails(tmp_path):
    from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

    bf16_path = _write_json(
        tmp_path / "bf16_cells.json",
        {"n2_mpe8": _cell_config()},
    )
    cfg = {
        "groups": [
            {"name": "int4", "num_bits": 4, "group_size": 128},
            {"name": "bf16", "num_bits": 16},
        ],
        "promotion_lookup_path": str(tmp_path / "missing.json"),
        "bf16_config_path": bf16_path,
    }
    with pytest.raises(FileNotFoundError, match="promotion_lookup_path"):
        HeterFusedMoE(
            num_experts=4,
            hidden_size=128,
            intermediate_size=128,
            top_k=2,
            heter_config=cfg,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_missing_bf16_cell_key_fails(tmp_path):
    from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

    lookup_path = _write_json(
        tmp_path / "promotion_lookup.json",
        {
            "bs8": {
                "_overall_best": "promotion",
                "n_active": 2,
                "cell_key": "missing_cell",
            }
        },
    )
    bf16_path = _write_json(
        tmp_path / "bf16_cells.json",
        {"n2_mpe8": _cell_config()},
    )
    cfg = {
        "groups": [
            {"name": "int4", "num_bits": 4, "group_size": 128},
            {"name": "bf16", "num_bits": 16},
        ],
        "promotion_lookup_path": lookup_path,
        "bf16_config_path": bf16_path,
    }
    with pytest.raises(ValueError, match="missing BF16 cell_key"):
        HeterFusedMoE(
            num_experts=4,
            hidden_size=128,
            intermediate_size=128,
            top_k=2,
            heter_config=cfg,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
