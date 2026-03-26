"""Rollout generation for GRPO training."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import BackoffConfig
from src.generation import BackoffGenerator, TrajectoryRecord


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

    Uses temperature > 0 for diverse trajectories.
    """
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

    messages = [{"role": "user", "content": (
        f"{config.system_prompt}\n\n{question}"
    )}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    trajectories = []
    for _ in range(num_rollouts):
        traj = gen.generate(prompt_ids, temperature=config.temperature)
        trajectories.append(traj)

    return trajectories
