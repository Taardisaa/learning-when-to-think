"""Shared types for the 3-action MDP pivot.

These dataclasses define the interface contract between:
- Member 1 (reward/eval)
- Member 2 (RL training)
- Member 3 (generation/experiments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal["continue", "refine", "terminate"]


@dataclass
class RolloutResult:
    """One generation rollout for a single prompt.

    Member 3 (generation) produces these; Member 1 (eval) consumes them.
    For baselines without explicit actions, ``actions`` is an empty list.
    """

    prompt_id: str
    question: str
    gold_answer: str
    level: int  # difficulty 1-5
    subject: str
    generated_text: str
    predicted_answer: str | None
    correct: bool
    num_tokens: int  # generated tokens only (not prompt)
    actions: list[Action] = field(default_factory=list)
    step_token_counts: list[int] = field(default_factory=list)


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics for a set of rollout results."""

    # Overall
    accuracy: float
    total_problems: int
    correct_count: int
    avg_tokens: float
    cost_per_correct: float  # total_tokens / correct_count (inf if 0)
    avg_steps: float

    # By difficulty level (1-5)
    accuracy_by_level: dict[int, float]
    avg_tokens_by_level: dict[int, float]
    count_by_level: dict[int, int]

    # By subject
    accuracy_by_subject: dict[str, float]

    # Action usage analysis (H3)
    action_counts_by_level: dict[int, dict[str, int]]
    action_fractions_by_level: dict[int, dict[str, float]]
