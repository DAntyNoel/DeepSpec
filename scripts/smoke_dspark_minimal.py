#!/usr/bin/env python
"""Minimal DSpark smoke test with synthetic tensors.

This checks the DSpark Qwen3 draft model path without downloading target
weights or preparing the full target cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import build_draft_config


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


def main() -> None:
    torch.manual_seed(0)
    init_path = Path(__file__).with_name(f".smoke_dspark_dist_{os.getpid()}")
    dist_initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_path}",
            rank=0,
            world_size=1,
        )
        dist_initialized_here = True

    try:
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
        )
        model_args = AttrDict(
            num_draft_layers=2,
            block_size=3,
            target_layer_ids=[0, 2],
            mask_token_id=63,
            num_anchors=4,
            markov_rank=0,
            confidence_head_alpha=0.0,
        )
        cfg = build_draft_config(target_cfg, model_args)
        model = Qwen3DSparkModel(cfg).eval()

        batch_size = 2
        seq_len = 12
        input_ids = torch.randint(0, cfg.vocab_size - 1, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        target_hidden_states = torch.randn(
            batch_size,
            seq_len,
            len(cfg.target_layer_ids) * cfg.hidden_size,
        )
        target_last_hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                target_hidden_states=target_hidden_states,
                target_last_hidden_states=target_last_hidden_states,
                loss_mask=loss_mask,
            )
            loss = compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=0.0,
            )

        print(f"draft_logits={tuple(outputs.draft_logits.shape)}")
        print(f"target_ids={tuple(outputs.target_ids.shape)}")
        print(f"eval_tokens={int(outputs.eval_mask.sum().item())}")
        print(f"loss={float(loss.item()):.6f}")
    finally:
        if dist_initialized_here:
            dist.destroy_process_group()
        if init_path.exists():
            init_path.unlink()


if __name__ == "__main__":
    main()
