"""Heterogeneous dispatch policy for mixed BF16/INT4 MoE.

The runtime has one policy: ``EfficiencyPromotionPolicy``.  It does not read
JSON, choose batch-size rows, or pick kernel configs.  The MoE layer passes the
desired number of BF16 experts (``n_active``), and the policy promotes the
top-count experts for the current routed batch.
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# (experts, scales) pair for one group, both [N, K].
GroupDispatchTuple = Tuple[torch.Tensor, torch.Tensor]


class HeterDispatchPolicy:
    """Shared tensor plumbing for heterogeneous MoE dispatch."""

    def __init__(
        self,
        num_experts: int,
        num_groups: int,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        self._num_experts = num_experts
        self._num_groups = num_groups
        self._device = device or torch.device("cuda")
        self._int4_only_mask = int4_only_mask
        self._int4_group_idx = int4_group_idx
        self._bf16_only_mask = bf16_only_mask
        self._bf16_group_idx = bf16_group_idx

        self._expert_to_group_buf = torch.zeros(
            num_experts, dtype=torch.long, device=self._device)
        self._token_count_buf = torch.zeros(
            num_experts + 1, dtype=torch.float32, device=self._device)
        self._ones_buf = torch.ones(
            num_experts, dtype=torch.float32, device=self._device)

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_groups(self) -> int:
        return self._num_groups

    def _compute_token_counts(
        self,
        token_selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-expert routed-slot counts, ignoring invalid sentinels."""
        counts = self._token_count_buf
        counts.zero_()
        flat = token_selected_experts.reshape(-1).long()
        valid = (flat >= 0) & (flat < self._num_experts)
        safe = torch.where(valid, flat, self._num_experts)
        n = safe.shape[0]
        if self._ones_buf.shape[0] < n:
            self._ones_buf = torch.ones(
                n, dtype=torch.float32, device=self._device)
        counts.scatter_add_(0, safe, self._ones_buf[:n])
        return counts[:self._num_experts]

    def _dispatch_from_expert_to_group(
        self,
        expert_to_group: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        sentinel: int,
    ) -> List[GroupDispatchTuple]:
        """Build per-group fixed-shape ``(ids, weights)`` dispatch tensors."""
        if self._int4_only_mask is not None:
            expert_to_group[self._int4_only_mask] = self._int4_group_idx
        if self._bf16_only_mask is not None:
            expert_to_group[self._bf16_only_mask] = self._bf16_group_idx

        selected = token_selected_experts.long()
        valid = (selected >= 0) & (selected < self._num_experts)
        safe_selected = selected.clamp(min=0, max=self._num_experts - 1)
        slot_groups = torch.where(
            valid, expert_to_group[safe_selected], -1)

        results: List[GroupDispatchTuple] = []
        for group_idx in range(self._num_groups):
            in_group = slot_groups == group_idx
            group_ids = torch.where(in_group, token_selected_experts, sentinel)
            group_weights = torch.where(in_group, token_final_scales, 0.0)
            results.append((group_ids, group_weights))
        return results


class EfficiencyPromotionPolicy(HeterDispatchPolicy):
    """Promote the top ``n_active`` routed experts to BF16."""

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        n_active: int,
        sentinel: int = -1,
    ) -> List[GroupDispatchTuple]:
        assignment = self._expert_to_group_buf
        assignment.fill_(self._int4_group_idx)

        n_active = max(0, min(int(n_active), self._num_experts))
        if n_active >= self._num_experts:
            assignment.fill_(self._bf16_group_idx)
        elif n_active > 0:
            token_counts = self._compute_token_counts(token_selected_experts)
            _, promoted = torch.topk(token_counts, n_active)
            assignment.scatter_(0, promoted, self._bf16_group_idx)

        return self._dispatch_from_expert_to_group(
            assignment,
            token_selected_experts,
            token_final_scales,
            sentinel=sentinel,
        )
