#!/usr/bin/env python
"""Tiny path-aware canonical-action scoring smoke."""

from __future__ import annotations

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from deepspec.modeling.dspark import score_teacher_forced_actions
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import build_draft_config


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


def main() -> None:
    torch.manual_seed(0)
    target_cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model_args = AttrDict(
        num_draft_layers=1,
        block_size=3,
        target_layer_ids=[0, 2],
        mask_token_id=31,
        num_anchors=2,
        markov_rank=4,
        markov_head_type="vanilla",
        confidence_head_alpha=0.0,
    )
    model = Qwen3DSparkModel(build_draft_config(target_cfg, model_args))
    base_logits = torch.randn(1, 3, 32, requires_grad=True)
    actions = torch.tensor([[[3, 4, 5], [3, 7, 0], [8, 0, 0]]])
    lengths = torch.tensor([[3, 2, 1]])
    scores = score_teacher_forced_actions(
        model,
        base_logits=base_logits,
        first_prev_token_ids=torch.tensor([2]),
        action_token_ids=actions,
        action_lengths=lengths,
        temperature=1.0,
    )
    assert scores.candidate_probs.shape == (1, 3)
    assert torch.allclose(scores.candidate_probs.sum(dim=-1), torch.ones(1))
    scores.candidate_log_probs[0, 0].backward()
    assert base_logits.grad is not None
    assert torch.isfinite(base_logits.grad).all()
    print("candidate_probs", scores.candidate_probs.detach().tolist())
    print("gradient_finite", True)


if __name__ == "__main__":
    main()
