"""ALP reward wrapper for trl's GRPOTrainer.

trl's `reward_funcs` interface expects a callable with signature:
    func(prompts: list, completions: list, **kwargs) -> list[float]

where `kwargs` receives extra dataset columns. We pass `gold_answer` through
the dataset so the reward function can grade each completion.

Within a single GRPOTrainer step, `prompts` / `completions` contain
num_prompts * num_generations items, ordered so all K rollouts for prompt i
are contiguous. We group them, compute ALP rewards per group, and flatten.
"""

from __future__ import annotations

from typing import Callable

from src.data.math import extract_boxed_answer, grade_math_answer
from src.pivot.reward import compute_alp_rewards


def make_alp_reward_func(
    beta: float = 0.05,
    l_max: int = 2048,
    num_generations: int = 4,
) -> Callable:
    """Build an ALP reward function compatible with trl's GRPOTrainer.

    Args:
        beta: length penalty coefficient
        l_max: token count normalization cap
        num_generations: K rollouts per prompt

    Returns:
        A callable `(prompts, completions, gold_answer, completion_ids, **_) -> list[float]`
    """

    def alp_reward(prompts, completions, completion_ids=None, gold_answer=None, **kwargs):
        if gold_answer is None:
            raise ValueError("ALP reward requires `gold_answer` column in the dataset")

        # trl passes completions as list[str] or list[list[message_dict]] depending on
        # whether the format is conversational. Normalize to list[str].
        texts: list[str] = []
        for c in completions:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list) and c and isinstance(c[0], dict):
                texts.append(c[-1].get("content", ""))
            else:
                texts.append(str(c))

        n = len(texts)
        assert len(gold_answer) == n, (
            f"gold_answer length {len(gold_answer)} != completions {n}"
        )
        # completion_ids is a list of lists of token ids
        if completion_ids is None:
            # Fallback: approximate token count from text length
            token_counts = [len(t) // 4 + 1 for t in texts]
        else:
            token_counts = [len(ids) for ids in completion_ids]

        # Grade each completion
        correctness = []
        for t, gold in zip(texts, gold_answer):
            pred = extract_boxed_answer(t)
            correctness.append(bool(grade_math_answer(pred, gold)))

        # Group by prompt (each consecutive num_generations items = same prompt)
        rewards = [0.0] * n
        for i in range(0, n, num_generations):
            group_correct = correctness[i : i + num_generations]
            group_tokens = token_counts[i : i + num_generations]
            group_rewards = compute_alp_rewards(
                group_correct, group_tokens, beta=beta, l_max=l_max
            )
            for j, r in enumerate(group_rewards):
                rewards[i + j] = r

        return rewards

    alp_reward.__name__ = "alp_reward"
    return alp_reward
