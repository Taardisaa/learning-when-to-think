"""Reward computation for GRPO training."""

from __future__ import annotations

from src.data.gsm8k import grade_answer
from src.generation import TrajectoryRecord


def compute_reward(
    trajectory: TrajectoryRecord,
    gold_answer: str,
    lambda_tok: float = 0.0001,
    lambda_explore: float = 0.0,
) -> float:
    """Compute reward for a trajectory.

    R(τ) = correctness - λ_tok * T_net + λ_explore * has_backoff

    Args:
        trajectory: the generated trajectory.
        gold_answer: gold answer string.
        lambda_tok: token cost coefficient.
        lambda_explore: exploration bonus (annealed, 0 in Phase 3).
    """
    correctness = 1.0 if grade_answer(trajectory.answer_number, gold_answer) else 0.0
    t_net = trajectory.final_kv_length
    has_backoff = 1.0 if trajectory.backoff_count > 0 else 0.0

    # Exploration bonus only when backoff contributed to a correct answer.
    # Prevents reward hacking where the model always backs off for free reward.
    explore_bonus = lambda_explore * has_backoff * correctness

    return correctness - lambda_tok * t_net + explore_bonus


def anneal_lambda_explore(
    base_lambda: float, step: int, anneal_steps: int
) -> float:
    """Linear annealing: base_lambda → 0 over anneal_steps."""
    if anneal_steps <= 0:
        return 0.0
    return base_lambda * max(0.0, 1.0 - step / anneal_steps)
