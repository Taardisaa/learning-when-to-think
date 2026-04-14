"""Pluggable action-injection strategies for RL rollouts.

Each strategy returns an HF `LogitsProcessor` that modifies logits during
generation. The rollout loop is strategy-agnostic — switching strategies
only requires changing the config.

Strategies:
    NoInjection: no modification; model samples freely from full vocab
    HardMask:    after `\\n\\n`, mask all non-action logits (force action choice)
    SoftBias:    after `\\n\\n`, add a bias to action token logits
"""

from __future__ import annotations

from typing import Protocol

import torch
from transformers import LogitsProcessor


# Tokens that end a paragraph / step in Qwen2.5 tokenizer.
# Each of these encodes a variant of "<punctuation>\n\n" or just "\n\n".
# When the model emits any of these, the next token is a candidate for action injection.
STEP_BOUNDARY_TOKEN_IDS = {
    271,   # "\n\n"
    382,   # ".\n\n"
    1447,  # ":\n\n"
    401,   # ";\n\n"
    1939,  # "?\n\n"
    2219,  # "!\n\n"
    1406,  # "\n\n\n"
    630,   # "}\n\n"
    692,   # ")\n\n"
    2533,  # "]\n\n"
}


class ActionStrategy(Protocol):
    """Interface for action-injection strategies."""

    name: str

    def logits_processor(self, action_token_ids: list[int]) -> LogitsProcessor | None:
        """Return an HF LogitsProcessor, or None for no modification."""
        ...


class NoInjection:
    """No-op strategy: natural sampling from the full vocabulary."""

    name = "no_injection"

    def logits_processor(self, action_token_ids: list[int]) -> LogitsProcessor | None:
        return None


def _boundary_mask(last_tokens: torch.Tensor, boundary_ids: set[int]) -> torch.Tensor:
    """Return boolean tensor marking batch rows where last token is a step boundary."""
    mask = torch.zeros_like(last_tokens, dtype=torch.bool)
    for bid in boundary_ids:
        mask |= (last_tokens == bid)
    return mask


class _HardMaskProcessor(LogitsProcessor):
    """Masks all non-action logits at step boundaries, until <terminate> is emitted.

    After the model emits <terminate>, masking is disabled for that sequence so
    it can freely generate \\boxed{answer} and <|im_end|>.
    """

    def __init__(self, action_token_ids: list[int], boundary_ids: set[int], terminate_id: int):
        self.action_ids = torch.tensor(action_token_ids, dtype=torch.long)
        self.boundary_ids = boundary_ids
        self.terminate_id = terminate_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            return scores

        # Disable masking for any sequence that has already emitted <terminate>
        already_terminated = (input_ids == self.terminate_id).any(dim=1)

        last_token = input_ids[:, -1]
        at_boundary = _boundary_mask(last_token, self.boundary_ids)

        # Apply mask only to rows that are at a boundary AND haven't terminated yet
        mask_rows = at_boundary & ~already_terminated
        if not mask_rows.any():
            return scores

        action_ids = self.action_ids.to(scores.device)
        vocab_allow = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        vocab_allow[action_ids] = True

        masked_scores = scores.clone()
        very_negative = torch.full_like(scores, float("-inf"))
        masked_scores[mask_rows] = torch.where(
            vocab_allow.unsqueeze(0).expand(mask_rows.sum(), -1),
            scores[mask_rows],
            very_negative[mask_rows],
        )
        return masked_scores


class HardMask:
    """After a step-boundary token, force the next token to be an action.

    Masking stops once <terminate> is emitted, so the model can freely write
    \\boxed{answer} and end the sequence.
    """

    name = "hard_mask"

    def __init__(self, boundary_ids: set[int] | None = None):
        self.boundary_ids = boundary_ids or STEP_BOUNDARY_TOKEN_IDS

    def logits_processor(self, action_token_ids: list[int]) -> LogitsProcessor:
        # Assume the third action token is <terminate> (matches src/pivot/tokens.py ordering)
        terminate_id = action_token_ids[-1]
        return _HardMaskProcessor(action_token_ids, self.boundary_ids, terminate_id)


class _SoftBiasProcessor(LogitsProcessor):
    """Adds a bias to action-token logits at positions following a step boundary."""

    def __init__(self, action_token_ids: list[int], bias: float, boundary_ids: set[int]):
        self.action_ids = torch.tensor(action_token_ids, dtype=torch.long)
        self.bias = bias
        self.boundary_ids = boundary_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            return scores

        last_token = input_ids[:, -1]
        bias_rows = _boundary_mask(last_token, self.boundary_ids)
        if not bias_rows.any():
            return scores

        action_ids = self.action_ids.to(scores.device)
        biased_scores = scores.clone()
        # Add bias to action logits only for rows at boundary
        biased_scores[bias_rows[:, None].expand(-1, scores.shape[-1]) & torch.zeros_like(scores, dtype=torch.bool).index_fill_(-1, action_ids, True)] += self.bias
        return biased_scores


class SoftBias:
    """After a step boundary, add a positive bias to action-token logits."""

    name = "soft_bias"

    def __init__(self, bias: float = 10.0, boundary_ids: set[int] | None = None):
        self.bias = bias
        self.boundary_ids = boundary_ids or STEP_BOUNDARY_TOKEN_IDS

    def logits_processor(self, action_token_ids: list[int]) -> LogitsProcessor:
        return _SoftBiasProcessor(action_token_ids, self.bias, self.boundary_ids)


_STRATEGIES: dict[str, type] = {
    "no_injection": NoInjection,
    "hard_mask": HardMask,
    "soft_bias": SoftBias,
}


def get_strategy(name: str, **kwargs) -> ActionStrategy:
    """Construct a strategy by name."""
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown action strategy '{name}'. Choices: {list(_STRATEGIES)}")
    return _STRATEGIES[name](**kwargs)
