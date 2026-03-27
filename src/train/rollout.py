"""Rollout generation for GRPO training."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import BackoffConfig
from src.generation import TrajectoryRecord
from src.generation_batched import BatchedBackoffGenerator
from src.prompt import build_prompt


@torch.no_grad()
def generate_rollouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: dict[str, int],
    question: str,
    config: BackoffConfig,
    num_rollouts: int = 4,
) -> list[TrajectoryRecord]:
    """Generate G rollouts for a single question.

    Uses batched generation: one model forward pass per autoregressive
    step processes all G rollouts simultaneously.
    """
    gen = BatchedBackoffGenerator(model, tokenizer, token_ids, config)

    prompt_text = build_prompt(tokenizer, question, config.system_prompt)
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    return gen.generate_batch(prompt_ids, G=num_rollouts, temperature=config.temperature)
