import pytest
import torch

from sglang.srt.lora.layers import BaseLayerWithLoRA, ReplicatedLinearWithLoRA
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.rollout_precision import RolloutPrecisionDecision, RolloutPrecisionPolicy


def test_rollout_precision_policy_is_decode_only():
    policy = RolloutPrecisionPolicy.from_payload(
        {
            "windows": [
                {
                    "batch_start": 64,
                    "batch_end": 32,
                    "projections": {
                        "qkv": "int4_torch_twostream",
                        "o": "bf16_merged",
                        "up": "int4_torch_twostream",
                        "down": "int4_torch_twostream",
                    },
                    "speedup_vs_bf16": 2.3,
                    "speedup_vs_csgmv": 3.6,
                }
            ]
        }
    )

    prefill_decision = policy.select(48, is_decode=False)
    assert not prefill_decision.enabled
    assert prefill_decision.choice_for_projection("qkv") == "bf16_merged"

    decode_decision = policy.select(48, is_decode=True)
    assert decode_decision.enabled
    assert decode_decision.use_int4_torch_twostream("qkv")
    assert decode_decision.use_bf16_merged("o")
    assert decode_decision.speedup_vs_bf16 == 2.3


def test_demo_policy_file_loads():
    policy = RolloutPrecisionPolicy.from_file(
        ".rollout-impl-codex/policies/qwen2.5-14b-eurus-demo.json"
    )
    decision = policy.select(48, is_decode=True)
    assert decision.enabled
    assert decision.use_int4_torch_twostream("qkv")
    assert decision.use_bf16_merged("o")


def test_lora_layer_keeps_prefill_on_base_path():
    layer = _make_lora_layer("qkv", is_decode=False)

    assert not layer._apply_lora_this_pass()
    assert not layer._use_torch_twostream_lora(torch.empty(1, 4))


def test_lora_layer_requires_merged_flag_to_skip_bf16_lora():
    layer = _make_lora_layer(
        "o",
        is_decode=True,
        decision=RolloutPrecisionDecision(
            enabled=True,
            source="unit",
            batch_start=64,
            batch_end=32,
            projections={
                "qkv": "int4_torch_twostream",
                "o": "bf16_merged",
                "up": "int4_torch_twostream",
                "down": "int4_torch_twostream",
            },
        ),
    )

    layer.lora_backend.batch_info.rollout_precision_assume_merged_bf16 = False
    assert layer._apply_lora_this_pass()

    layer.lora_backend.batch_info.rollout_precision_assume_merged_bf16 = True
    assert not layer._apply_lora_this_pass()


def test_rollout_policy_overrides_global_twostream_for_bf16_projection():
    layer = _make_lora_layer(
        "o",
        is_decode=True,
        decision=RolloutPrecisionDecision(
            enabled=True,
            source="unit",
            batch_start=64,
            batch_end=32,
            projections={
                "qkv": "int4_torch_twostream",
                "o": "bf16_merged",
                "up": "int4_torch_twostream",
                "down": "int4_torch_twostream",
            },
        ),
    )
    layer.lora_torch_twostream = True

    assert not layer._use_torch_twostream_lora(_FakeCudaInput())


def test_rollout_policy_enables_twostream_for_int4_projection():
    layer = _make_lora_layer(
        "qkv",
        is_decode=True,
        decision=_int4_decision(),
    )
    layer.lora_torch_twostream = False

    assert layer._use_torch_twostream_lora(_FakeCudaInput())


def test_cuda_graph_capture_uses_active_lora_for_int4_policy():
    manager = LoRAManager.__new__(LoRAManager)
    manager.max_loras_per_batch = 1
    manager.loras = {"adapter-id": object()}
    manager.lora_modules = []
    manager.lora_refs = {}
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    manager.memory_pool = _FakeMemoryPool()

    lora_ids = manager.get_cuda_graph_capture_lora_ids(8, _int4_decision())

    assert lora_ids == ["adapter-id"] * 8
    assert manager.memory_pool.prepared_uids == {"adapter-id"}


def test_cuda_graph_capture_keeps_empty_lora_for_bf16_policy():
    manager = LoRAManager.__new__(LoRAManager)
    manager.max_loras_per_batch = 1
    manager.loras = {"adapter-id": object()}
    manager.memory_pool = _FakeMemoryPool()

    lora_ids = manager.get_cuda_graph_capture_lora_ids(8, _bf16_decision())

    assert lora_ids == [None] * 8
    assert manager.memory_pool.prepared_uids is None


def test_lora_layer_dispatches_int4_base_to_shadow_layer():
    layer = _make_replicated_lora_layer(
        "qkv",
        is_decode=True,
        decision=_int4_decision(),
        base_offset=1.0,
        shadow_offset=10.0,
    )
    x = torch.ones(2, 4)

    output, output_bias = layer.forward(x)

    assert output_bias is None
    assert torch.allclose(output, x + 10.0)


def test_lora_layer_keeps_prefill_on_bf16_base_even_with_shadow_layer():
    layer = _make_replicated_lora_layer(
        "qkv",
        is_decode=False,
        decision=_int4_decision(),
        base_offset=1.0,
        shadow_offset=10.0,
    )
    x = torch.ones(2, 4)

    output, _ = layer.forward(x)

    assert torch.allclose(output, x + 1.0)


def test_lora_layer_raises_when_int4_policy_has_no_shadow_layer():
    layer = _make_replicated_lora_layer(
        "qkv",
        is_decode=True,
        decision=_int4_decision(),
        base_offset=1.0,
        shadow_offset=None,
    )

    with pytest.raises(RuntimeError, match="no INT4 shadow layer"):
        layer.forward(torch.ones(2, 4))


def _int4_decision():
    return RolloutPrecisionDecision(
        enabled=True,
        source="unit",
        batch_start=64,
        batch_end=32,
        projections={
            "qkv": "int4_torch_twostream",
            "o": "bf16_merged",
            "up": "bf16_merged",
            "down": "bf16_merged",
        },
    )


def _bf16_decision():
    return RolloutPrecisionDecision(
        enabled=True,
        source="unit",
        batch_start=512,
        batch_end=128,
        projections={
            "qkv": "bf16_merged",
            "o": "bf16_merged",
            "up": "bf16_merged",
            "down": "bf16_merged",
        },
    )


def _make_lora_layer(
    projection,
    is_decode,
    decision=None,
):
    backend = _DummyBackend(
        batch_info=LoRABatchInfo(
            use_cuda_graph=False,
            bs=1,
            num_segments=1,
            seg_indptr=torch.tensor([0, 1], dtype=torch.int32),
            weight_indices=torch.tensor([0], dtype=torch.int32),
            lora_ranks=torch.tensor([8], dtype=torch.int32),
            scalings=torch.tensor([1.0], dtype=torch.float32),
            max_len=1,
            seg_lens=torch.tensor([1], dtype=torch.int32),
            permutation=None,
            has_active_lora=True,
            is_decode=is_decode,
            rollout_precision_decision=decision,
        )
    )
    layer = BaseLayerWithLoRA(torch.nn.Linear(4, 4, bias=False), backend)
    layer.set_lora = True
    layer.rollout_precision_projection = projection
    return layer


def _make_replicated_lora_layer(
    projection,
    is_decode,
    decision,
    base_offset,
    shadow_offset,
):
    backend = _DummyBackend(
        batch_info=LoRABatchInfo(
            use_cuda_graph=False,
            bs=1,
            num_segments=1,
            seg_indptr=torch.tensor([0, 1], dtype=torch.int32),
            weight_indices=torch.tensor([0], dtype=torch.int32),
            lora_ranks=torch.tensor([0], dtype=torch.int32),
            scalings=torch.tensor([1.0], dtype=torch.float32),
            max_len=1,
            seg_lens=torch.tensor([1], dtype=torch.int32),
            permutation=None,
            has_active_lora=False,
            is_decode=is_decode,
            rollout_precision_decision=decision,
        )
    )
    layer = ReplicatedLinearWithLoRA(_DummyLinear(base_offset), backend)
    layer.rollout_precision_projection = projection
    layer.rollout_precision_module_name = f"layers.0.{projection}"
    if shadow_offset is not None:
        layer.rollout_precision_int4_base_layer = _DummyLinear(shadow_offset)
    return layer


class _DummyBackend:
    name = "torch_native"

    def __init__(self, batch_info):
        self.batch_info = batch_info


class _FakeCudaInput:
    is_cuda = True


class _FakeMemoryPool:
    def __init__(self):
        self.prepared_uids = None

    def prepare_lora_batch(self, cur_uids, **kwargs):
        del kwargs
        self.prepared_uids = cur_uids


class _DummyQuantMethod:
    def __init__(self, offset):
        self.offset = offset

    def apply(self, layer, x, bias=None):
        del layer, bias
        return x + self.offset


class _DummyLinear(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.output_size = 4
        self.skip_bias_add = False
        self.bias = None
        self.quant_method = _DummyQuantMethod(offset)
