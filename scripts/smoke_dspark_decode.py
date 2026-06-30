#!/usr/bin/env python
"""Minimal DSpark speculative-decoding smoke test.

This runs the DSpark decode loop with tiny randomly initialized Qwen3 target
and DSpark draft models. It does not download a target model, draft checkpoint,
or dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from deepspec.eval.base_evaluator import generate_decoding_sample
from deepspec.eval.dspark.draft_ops import (
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import build_draft_config


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


def main() -> None:
    torch.manual_seed(7)

    target_cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        eos_token_id=2,
        pad_token_id=0,
    )
    target_model = Qwen3ForCausalLM(target_cfg).eval()

    model_args = AttrDict(
        num_draft_layers=2,
        block_size=3,
        target_layer_ids=[0, 2],
        mask_token_id=63,
        num_anchors=4,
        markov_rank=0,
        confidence_head_alpha=0.0,
    )
    draft_cfg = build_draft_config(target_cfg, model_args)
    draft_cfg._attn_implementation = "eager"
    draft_model = Qwen3DSparkModel(draft_cfg).eval()

    def init_context(*, initial_output, **_) -> SimpleNamespace:
        return SimpleNamespace(
            past_key_values_draft=DynamicCache(),
            target_hidden_states=extract_context_feature(
                initial_output.hidden_states,
                draft_model.target_layer_ids,
            ),
        )

    def propose(
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ):
        del stop_token_ids
        draft_input_ids = torch.full(
            (output_ids.size(0), draft_model.block_size),
            int(draft_model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        block_hidden = forward_dspark_draft_block(
            draft_model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=draft_model.block_size,
        )
        return build_dspark_proposal(
            model=draft_model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=draft_model.block_size,
            temperature=0.0,
            confidence_threshold=0.0,
        )

    def update(context: SimpleNamespace, verification) -> None:
        verified_target_hidden = extract_context_feature(
            verification.target_output.hidden_states,
            draft_model.target_layer_ids,
        )
        context.target_hidden_states = verified_target_hidden[
            :,
            : verification.accepted_draft_tokens + 1,
            :,
        ]

    input_ids = torch.tensor([[5, 11, 17, 23]], dtype=torch.long)
    sample = generate_decoding_sample(
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=6,
        max_proposal_tokens=draft_model.block_size,
        temperature=0.0,
        stop_token_ids=None,
        init_context=init_context,
        propose=propose,
        update=update,
    )

    print(f"output_ids={sample.output_ids.tolist()}")
    print(f"num_output_tokens={sample.num_output_tokens}")
    print(f"verify_count={sample.verify_count}")
    print(f"proposal_lengths={sample.proposal_lengths}")
    print(f"accepted_draft_lengths={sample.accepted_draft_lengths}")
    print(f"acceptance_lengths={sample.acceptance_lengths}")


if __name__ == "__main__":
    main()
