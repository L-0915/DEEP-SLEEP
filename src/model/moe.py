"""Mixture of Experts (MoE) layer for the DeepSleep model.

Implements sparse MoE following the DeepSeek / Mixtral industry standard:
- Softmax over all experts first, then top-k selection with renormalization
- No token dropping: all tokens are routed to their selected experts
- DeepSeek-style load balancing auxiliary loss
- Router z-loss for numerical stability
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DeepSleepMLP

logger = logging.getLogger(__name__)


class DeepSleepMoE(nn.Module):
    """Mixture-of-Experts layer following DeepSeek-MoE / Mixtral conventions.

    Routing procedure (industry standard):
        1. Compute router logits for all experts
        2. Softmax over all experts to get routing probabilities
        3. Select top-k experts per token
        4. Renormalize selected expert weights to sum to 1
        5. Dispatch tokens to selected experts (no dropping)
        6. Compute DeepSeek-style load balancing loss on the full softmax probs

    Args:
        config: Model configuration with MoE-related attributes.
    """

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.config = config
        self.num_routed_experts = config.num_routed_experts
        self.num_shared_experts = config.num_shared_experts
        self.top_k = config.top_k
        self.aux_loss_coeff = config.aux_loss_coeff
        self.z_loss_coeff = config.z_loss_coeff
        self.router_jitter_noise = config.router_jitter_noise

        # Router: maps hidden dim -> num_routed_experts logits (no bias)
        self.gate = nn.Linear(config.d_model, self.num_routed_experts, bias=False)

        # Routed experts (sparse)
        expert_config = type("ExpertConfig", (), {
            "d_model": config.d_model,
            "hidden_size": config.d_model,
            "intermediate_size": config.moe_intermediate_size,
        })()
        self.experts = nn.ModuleList(
            [DeepSleepMLP(expert_config) for _ in range(self.num_routed_experts)]
        )

        # Shared expert (always active)
        if self.num_shared_experts > 0:
            self.shared_expert = DeepSleepMLP(expert_config)
        else:
            self.shared_expert = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the MoE layer.

        Args:
            hidden_states: Input of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tuple of (output, aux_losses_dict).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. Router logits: (batch, seq_len, num_routed_experts)
        router_logits = self.gate(hidden_states)

        # 2. Softmax over ALL experts (DeepSeek/Mixtral standard)
        routing_probs = F.softmax(router_logits.float(), dim=-1)  # (B, S, E)

        # 3. Top-k selection on probabilities
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # 4. Renormalize selected weights to sum to 1
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        # 5. Compute auxiliary losses (on full softmax probs, before top-k)
        aux_loss = self._compute_load_balance_loss(routing_probs, top_k_indices)
        z_loss = self._compute_z_loss(router_logits)

        # 6. Dispatch tokens to experts and combine (no token dropping)
        expert_output = self._dispatch_and_combine(
            hidden_states, top_k_weights, top_k_indices
        )

        # 7. Add shared expert if present
        if self.shared_expert is not None:
            expert_output = expert_output + self.shared_expert(hidden_states)

        aux_losses = {
            "aux_loss": aux_loss * self.aux_loss_coeff,
            "z_loss": z_loss * self.z_loss_coeff,
        }

        return expert_output, aux_losses

    def _compute_load_balance_loss(
        self,
        routing_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """DeepSeek-style load balancing auxiliary loss.

        Computes: ``aux_loss = num_experts * sum(f_i * P_i)``

        where:
        - f_i = fraction of tokens whose top-k set includes expert i (hard routing freq)
        - P_i = average routing probability for expert i (soft routing prob)

        When perfectly balanced: f_i = top_k/num_experts, P_i = 1/num_experts,
        loss = num_experts * (top_k/num_experts) * (1/num_experts) = top_k/num_experts.
        This is the minimum possible value.

        Args:
            routing_probs: Full softmax probabilities ``(batch, seq_len, num_experts)``.
            top_k_indices: Selected expert indices ``(batch, seq_len, top_k)``.

        Returns:
            Scalar load balance loss.
        """
        num_experts = self.num_routed_experts
        num_tokens = routing_probs.shape[0] * routing_probs.shape[1]

        # f_i: fraction of tokens routed to each expert (hard assignment)
        # Count how many times each expert appears in the top-k selections
        expert_mask = F.one_hot(top_k_indices, num_classes=num_experts).float()
        # expert_mask: (batch, seq_len, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=(0, 1, 2))  # (num_experts,)
        f_i = tokens_per_expert / (num_tokens * self.top_k)  # normalize

        # P_i: average routing probability per expert (soft probability from full softmax)
        P_i = routing_probs.mean(dim=(0, 1))  # (num_experts,)

        # Load balance loss
        aux_loss = num_experts * torch.sum(f_i * P_i)
        return aux_loss

    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss for numerical stability.

        ``z_loss = (1/T) * sum(log_sum_exp(logits)^2)``

        Prevents router logits from growing too large.

        Args:
            router_logits: Raw router logits ``(batch, seq_len, num_experts)``.

        Returns:
            Scalar z-loss.
        """
        log_z = torch.logsumexp(router_logits.float(), dim=-1)
        z_loss = torch.mean(log_z ** 2)
        return z_loss

    def _dispatch_and_combine(
        self,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to selected experts and combine results.

        No token dropping: every token is processed by all its top-k experts.

        Args:
            hidden_states: ``(batch, seq_len, d_model)``.
            top_k_weights: Renormalized weights ``(batch, seq_len, top_k)``.
            top_k_indices: Expert indices ``(batch, seq_len, top_k)``.

        Returns:
            Combined output ``(batch, seq_len, d_model)``.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Flatten to (num_tokens, ...)
        hidden_states_flat = hidden_states.view(num_tokens, hidden_dim)
        top_k_weights_flat = top_k_weights.view(num_tokens, self.top_k)
        top_k_indices_flat = top_k_indices.view(num_tokens, self.top_k)

        # Initialize output
        output_flat = torch.zeros(num_tokens, hidden_dim,
                                  dtype=hidden_states.dtype, device=hidden_states.device)

        # Process each expert: gather its tokens, run forward, scatter results back
        for expert_idx in range(self.num_routed_experts):
            expert = self.experts[expert_idx]

            # Find tokens routed to this expert across all top-k slots
            # expert_mask: (num_tokens, top_k) - True where this expert is selected
            expert_mask = (top_k_indices_flat == expert_idx)
            token_ids, k_ids = torch.where(expert_mask)

            if token_ids.numel() == 0:
                continue

            # Gather input tokens
            expert_input = hidden_states_flat[token_ids]

            # Forward through expert
            expert_output = expert(expert_input)

            # Weight by routing weight
            weights = top_k_weights_flat[token_ids, k_ids].unsqueeze(-1)
            expert_output = expert_output * weights

            # Scatter-add back to output
            output_flat.scatter_add_(
                dim=0,
                index=token_ids.unsqueeze(-1).expand(-1, hidden_dim),
                src=expert_output,
            )

        return output_flat.view(batch_size, seq_len, hidden_dim)
