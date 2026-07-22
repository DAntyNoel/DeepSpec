from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DSparkActionScoreOutput:
    token_log_probs: torch.Tensor
    action_log_scores: torch.Tensor
    action_mean_log_scores: torch.Tensor
    candidate_log_probs: torch.Tensor
    candidate_probs: torch.Tensor
    action_mask: torch.Tensor
    candidate_mask: torch.Tensor


def score_teacher_forced_actions(
    model,
    *,
    base_logits: torch.Tensor,
    first_prev_token_ids: torch.Tensor,
    action_token_ids: torch.Tensor,
    action_lengths: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> DSparkActionScoreOutput:
    """Score canonical actions with the DSpark sequential-head semantics.

    ``base_logits`` and optional ``hidden_states`` are the parallel-backbone
    outputs for one state. Every candidate reuses that state context, while
    the Markov head is teacher-forced with the candidate's own prefix.
    """

    if base_logits.ndim != 3:
        raise ValueError(
            "base_logits must have shape [batch, horizon, vocab], "
            f"got {tuple(base_logits.shape)}"
        )
    if action_token_ids.ndim != 3:
        raise ValueError(
            "action_token_ids must have shape [batch, candidates, length], "
            f"got {tuple(action_token_ids.shape)}"
        )
    batch_size, horizon, vocab_size = base_logits.shape
    action_batch, num_candidates, max_action_length = action_token_ids.shape
    if action_batch != batch_size:
        raise ValueError("base_logits and action_token_ids batch sizes must match")
    if max_action_length < 1 or max_action_length > horizon:
        raise ValueError(
            "action length must be in [1, horizon], "
            f"got length={max_action_length}, horizon={horizon}"
        )
    if action_lengths.shape != (batch_size, num_candidates):
        raise ValueError(
            "action_lengths must have shape [batch, candidates], "
            f"got {tuple(action_lengths.shape)}"
        )
    if first_prev_token_ids.shape != (batch_size,):
        raise ValueError(
            "first_prev_token_ids must have shape [batch], "
            f"got {tuple(first_prev_token_ids.shape)}"
        )
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    action_token_ids = action_token_ids.long()
    action_lengths = action_lengths.long()
    if bool(((action_lengths < 1) | (action_lengths > max_action_length)).any().item()):
        raise ValueError("every action length must be within the padded action width")
    if bool(((action_token_ids < 0) | (action_token_ids >= vocab_size)).any().item()):
        raise ValueError("action_token_ids contain ids outside the draft vocabulary")

    if candidate_mask is None:
        candidate_mask = torch.ones(
            batch_size,
            num_candidates,
            dtype=torch.bool,
            device=action_token_ids.device,
        )
    else:
        candidate_mask = candidate_mask.to(
            device=action_token_ids.device,
            dtype=torch.bool,
        )
        if candidate_mask.shape != (batch_size, num_candidates):
            raise ValueError(
                "candidate_mask must have shape [batch, candidates], "
                f"got {tuple(candidate_mask.shape)}"
            )
    if bool((~candidate_mask).all(dim=1).any().item()):
        raise ValueError("every batch row must contain at least one valid candidate")

    step_ids = torch.arange(max_action_length, device=action_token_ids.device)
    action_mask = step_ids.view(1, 1, -1) < action_lengths.unsqueeze(-1)
    first_prev = first_prev_token_ids.long().view(batch_size, 1, 1).expand(
        -1,
        num_candidates,
        -1,
    )
    prev_token_ids = torch.cat(
        [first_prev, action_token_ids[:, :, :-1]],
        dim=-1,
    )
    expanded_base_logits = base_logits[:, None, :max_action_length, :].expand(
        -1,
        num_candidates,
        -1,
        -1,
    )
    expanded_hidden = None
    if hidden_states is not None:
        if hidden_states.shape[:2] != (batch_size, horizon):
            raise ValueError(
                "hidden_states must have shape [batch, horizon, hidden], "
                f"got {tuple(hidden_states.shape)}"
            )
        expanded_hidden = hidden_states[:, None, :max_action_length, :].expand(
            -1,
            num_candidates,
            -1,
            -1,
        )

    if model.markov_head is None:
        corrected_logits = expanded_base_logits
    else:
        corrected_logits = model.markov_head.apply_block_logits(
            expanded_base_logits,
            token_ids=prev_token_ids,
            hidden_states=expanded_hidden,
        )
    token_log_probs = torch.log_softmax(corrected_logits.float(), dim=-1).gather(
        -1,
        action_token_ids.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = torch.where(
        action_mask,
        token_log_probs,
        torch.zeros_like(token_log_probs),
    )
    action_log_scores = token_log_probs.sum(dim=-1)
    action_mean_log_scores = action_log_scores / action_lengths.to(
        dtype=action_log_scores.dtype
    )
    policy_logits = action_mean_log_scores / float(temperature)
    policy_logits = policy_logits.masked_fill(~candidate_mask, float("-inf"))
    candidate_log_probs = torch.log_softmax(policy_logits, dim=-1)
    candidate_probs = candidate_log_probs.exp()
    return DSparkActionScoreOutput(
        token_log_probs=token_log_probs,
        action_log_scores=action_log_scores,
        action_mean_log_scores=action_mean_log_scores,
        candidate_log_probs=candidate_log_probs,
        candidate_probs=candidate_probs,
        action_mask=action_mask,
        candidate_mask=candidate_mask,
    )


__all__ = [
    "DSparkActionScoreOutput",
    "score_teacher_forced_actions",
]

