"""Rollout generation for GRPO training."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import BackoffConfig
from src.generation import TrajectoryRecord
from src.generation_batched import BatchedBackoffGenerator


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

    messages = [{"role": "user", "content": (
        f"{config.system_prompt}\n\n{question}"
    )}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    return gen.generate_batch(prompt_ids, G=num_rollouts, temperature=config.temperature)
