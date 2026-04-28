"""Custom loss functions for DeepSleep training.

Provides loss computations for:
* Standard language modeling (cross-entropy)
* Mixture-of-Experts auxiliary losses (load-balance + z-loss)
* Direct Preference Optimization (DPO)

All loss functions return a scalar ``torch.Tensor`` suitable for
``.backward()``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Language Modeling Loss
# ---------------------------------------------------------------------------


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the standard language-modeling cross-entropy loss.

    Args:
        logits: Model output logits of shape ``(batch, seq_len, vocab_size)``.
        labels: Target token ids of shape ``(batch, seq_len)``.  Positions
            with value *ignore_index* are excluded from the loss.
        ignore_index: Label value that indicates padding / non-compute tokens.

    Returns:
        Scalar loss tensor (mean over non-ignored tokens).
    """
    # Shift so that predictions align with targets:
    # logits[B, S-1, V] -> labels[B, S-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


# ---------------------------------------------------------------------------
# MoE Auxiliary Loss
# ---------------------------------------------------------------------------


def compute_moe_aux_loss(
    moe_losses: List[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Aggregate MoE auxiliary losses from all MoE layers.

    Each element in *moe_losses* is a dictionary with keys such as
    ``"load_balance_loss"`` and ``"z_loss"``.  The returned value is the
    sum of all auxiliary losses across all MoE layers.

    This aggregated auxiliary loss should be added to the main LM loss (often
    with a small coefficient) during back-propagation.

    Args:
        moe_losses: List of per-layer auxiliary loss dictionaries.  Can be
            empty (returns zero).

    Returns:
        Scalar tensor (zero if *moe_losses* is empty).
    """
    if not moe_losses:
        return torch.tensor(0.0, device=_infer_device(moe_losses), dtype=torch.float32)

    total = torch.tensor(0.0, dtype=torch.float32)
    for layer_losses in moe_losses:
        for loss_name, loss_val in layer_losses.items():
            total = total + loss_val.float()

    return total


def _infer_device(moe_losses: List[Dict[str, torch.Tensor]]) -> torch.device:
    """Infer the device from the first available tensor in *moe_losses*.

    Returns ``cpu`` when the list is empty.
    """
    for layer_losses in moe_losses:
        for loss_val in layer_losses.values():
            return loss_val.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------


def log_prob_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probabilities from logits.

    Uses the numerically stable log-softmax formulation.

    Args:
        logits: Logits of shape ``(batch, seq_len, vocab_size)``.
        labels: Token ids of shape ``(batch, seq_len)``.

    Returns:
        Log-probabilities of shape ``(batch, seq_len)``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log-prob for each token in *labels*.
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute mean per-sequence log-probability.

    Sequences may have different lengths of non-ignored tokens.  The returned
    value is the *mean* log-probability over all non-ignored tokens for each
    sequence in the batch.

    Args:
        logits: ``[batch, seq_len, vocab_size]``
        labels: ``[batch, seq_len]``
        ignore_index: Label value to skip.

    Returns:
        Tensor of shape ``(batch,)`` with mean log-prob per sequence.
    """
    per_token_logps = log_prob_from_logits(logits, labels)
    mask = (labels != ignore_index).float()
    # Avoid division by zero for fully-masked sequences.
    token_count = mask.sum(dim=-1).clamp(min=1.0)
    return (per_token_logps * mask).sum(dim=-1) / token_count


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the Direct Preference Optimization (DPO) loss.

    The DPO loss is:

        L = -log(sigmoid(beta * (log_pi_chosen - log_pi_ref_chosen
                                  - log_pi_rejected + log_pi_ref_rejected)))

    where ``pi`` denotes the policy model and ``pi_ref`` the frozen reference
    model.

    Args:
        policy_chosen_logps: Log-probs of chosen completions under the policy.
            Shape ``(batch,)``.
        policy_rejected_logps: Log-probs of rejected completions under the
            policy.  Shape ``(batch,)``.
        reference_chosen_logps: Log-probs of chosen completions under the
            reference model.  Shape ``(batch,)``.
        reference_rejected_logps: Log-probs of rejected completions under the
            reference model.  Shape ``(batch,)``.
        beta: DPO temperature parameter (controls deviation from the
            reference policy).

    Returns:
        Scalar DPO loss tensor (mean over the batch).
    """
    logits_diff = (
        policy_chosen_logps
        - reference_chosen_logps
        - policy_rejected_logps
        + reference_rejected_logps
    )
    # Scaled logit difference
    scaled = beta * logits_diff
    # -log(sigmoid(x)) = log(1 + exp(-x)), but we use the stable version.
    loss = -F.logsigmoid(scaled)
    return loss.mean()
