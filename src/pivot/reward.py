"""Adaptive Length Penalty (ALP) reward for 3-action MDP.

Formula:  R_k = r_acc,k - beta * max(0, SR(q)) * n_tokens,k / L_max

Where SR(q) = (1/K) * sum_k(r_acc,k) is the group solve rate.

Pure Python — no torch dependency. Takes primitives, not TrajectoryRecord.
"""

from __future__ import annotations


def compute_group_solve_rate(correctness: list[bool]) -> float:
    """SR(q) = (1/K) * sum(r_acc,k)."""
    if not correctness:
        return 0.0
    return sum(correctness) / len(correctness)


def compute_alp_rewards(
    correctness: list[bool],
    token_counts: list[int],
    beta: float = 0.05,
    l_max: int = 2048,
) -> list[float]:
    """Compute ALP rewards for K rollouts on the same prompt.

    Args:
        correctness: Whether each rollout got the right answer.
        token_counts: Number of generated tokens per rollout.
        beta: Length penalty coefficient.
        l_max: Normalization cap for token counts.

    Returns:
        List of K reward floats.
    """
    if len(correctness) != len(token_counts):
        raise ValueError(
            f"correctness ({len(correctness)}) and token_counts "
            f"({len(token_counts)}) must have the same length"
        )
    if not correctness:
        return []

    sr = compute_group_solve_rate(correctness)
    rewards = []
    for correct, n_tokens in zip(correctness, token_counts):
        r_acc = 1.0 if correct else 0.0
        penalty = beta * max(0.0, sr) * n_tokens / l_max
        rewards.append(r_acc - penalty)
    return rewards


def compute_alp_rewards_batch(
    groups: list[list[tuple[bool, int]]],
    beta: float = 0.05,
    l_max: int = 2048,
) -> list[list[float]]:
    """Compute ALP rewards for multiple prompt groups.

    Convenience wrapper for GRPO training (Q prompts x K rollouts).

    Args:
        groups: List of groups. Each group is a list of (correct, n_tokens)
                tuples for K rollouts on the same prompt.
        beta: Length penalty coefficient.
        l_max: Normalization cap for token counts.

    Returns:
        List of lists of reward floats, one inner list per group.
    """
    results = []
    for group in groups:
        correctness = [c for c, _ in group]
        token_counts = [n for _, n in group]
        results.append(compute_alp_rewards(correctness, token_counts, beta, l_max))
    return results
