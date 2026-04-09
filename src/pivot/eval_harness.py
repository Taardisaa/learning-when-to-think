"""Evaluation harness for the 3-action MDP pivot.

Computes accuracy, efficiency, and action-usage metrics from RolloutResults.
Supports all three hypotheses from the proposal:
  H1: Pareto frontier (accuracy vs avg_tokens)
  H2: Token-difficulty correlation
  H3: Action distribution shifts with difficulty
"""

from __future__ import annotations

from collections import defaultdict

from src.pivot.eval_types import EvalMetrics, RolloutResult


def evaluate(results: list[RolloutResult]) -> EvalMetrics:
    """Compute all evaluation metrics from rollout results.

    Expects one RolloutResult per problem (single-pass evaluation).
    """
    if not results:
        raise ValueError("Cannot evaluate empty results list")

    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    total_tokens = sum(r.num_tokens for r in results)
    total_steps = sum(len(r.actions) for r in results)

    accuracy = correct_count / total
    avg_tokens = total_tokens / total
    cost_per_correct = total_tokens / correct_count if correct_count > 0 else float("inf")
    avg_steps = total_steps / total

    # By level
    by_level: dict[int, list[RolloutResult]] = defaultdict(list)
    for r in results:
        by_level[r.level].append(r)

    accuracy_by_level = {}
    avg_tokens_by_level = {}
    count_by_level = {}
    for lvl, group in sorted(by_level.items()):
        count_by_level[lvl] = len(group)
        accuracy_by_level[lvl] = sum(1 for r in group if r.correct) / len(group)
        avg_tokens_by_level[lvl] = sum(r.num_tokens for r in group) / len(group)

    # By subject
    by_subject: dict[str, list[RolloutResult]] = defaultdict(list)
    for r in results:
        by_subject[r.subject].append(r)

    accuracy_by_subject = {
        subj: sum(1 for r in group if r.correct) / len(group)
        for subj, group in sorted(by_subject.items())
    }

    # Action counts by level (H3)
    action_counts_by_level: dict[int, dict[str, int]] = {}
    action_fractions_by_level: dict[int, dict[str, float]] = {}
    has_actions = any(r.actions for r in results)

    if has_actions:
        for lvl, group in sorted(by_level.items()):
            counts: dict[str, int] = defaultdict(int)
            for r in group:
                for a in r.actions:
                    counts[a] += 1
            total_actions = sum(counts.values())
            action_counts_by_level[lvl] = dict(counts)
            action_fractions_by_level[lvl] = (
                {a: c / total_actions for a, c in counts.items()}
                if total_actions > 0
                else {}
            )

    return EvalMetrics(
        accuracy=accuracy,
        total_problems=total,
        correct_count=correct_count,
        avg_tokens=avg_tokens,
        cost_per_correct=cost_per_correct,
        avg_steps=avg_steps,
        accuracy_by_level=accuracy_by_level,
        avg_tokens_by_level=avg_tokens_by_level,
        count_by_level=count_by_level,
        accuracy_by_subject=accuracy_by_subject,
        action_counts_by_level=action_counts_by_level,
        action_fractions_by_level=action_fractions_by_level,
    )


def compute_token_difficulty_correlation(results: list[RolloutResult]) -> float:
    """Spearman rank correlation between difficulty level and token count.

    Positive correlation means harder problems get more tokens (H2).
    """
    from scipy.stats import spearmanr

    levels = [r.level for r in results]
    tokens = [r.num_tokens for r in results]
    corr, _ = spearmanr(levels, tokens)
    return float(corr)


def print_eval_report(metrics: EvalMetrics) -> None:
    """Pretty-print evaluation results to stdout."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results  ({metrics.total_problems} problems)")
    print(f"{'='*60}")
    print(f"  Accuracy:           {metrics.accuracy:.4f} ({metrics.correct_count}/{metrics.total_problems})")
    print(f"  Avg tokens:         {metrics.avg_tokens:.1f}")
    print(f"  Cost per correct:   {metrics.cost_per_correct:.1f}")
    print(f"  Avg steps:          {metrics.avg_steps:.2f}")

    print(f"\n  {'Level':<8} {'Count':>6} {'Accuracy':>10} {'Avg Tokens':>12}")
    print(f"  {'-'*40}")
    for lvl in sorted(metrics.accuracy_by_level):
        print(
            f"  {lvl:<8} {metrics.count_by_level[lvl]:>6} "
            f"{metrics.accuracy_by_level[lvl]:>10.4f} "
            f"{metrics.avg_tokens_by_level[lvl]:>12.1f}"
        )

    print(f"\n  {'Subject':<30} {'Accuracy':>10}")
    print(f"  {'-'*42}")
    for subj in sorted(metrics.accuracy_by_subject):
        print(f"  {subj:<30} {metrics.accuracy_by_subject[subj]:>10.4f}")

    if metrics.action_counts_by_level:
        print(f"\n  Action distribution by level:")
        all_actions = sorted(
            {a for counts in metrics.action_counts_by_level.values() for a in counts}
        )
        header = f"  {'Level':<8}" + "".join(f"{a:>12}" for a in all_actions)
        print(header)
        print(f"  {'-'*(8 + 12 * len(all_actions))}")
        for lvl in sorted(metrics.action_fractions_by_level):
            fracs = metrics.action_fractions_by_level[lvl]
            row = f"  {lvl:<8}" + "".join(
                f"{fracs.get(a, 0.0):>11.1%} " for a in all_actions
            )
            print(row)

    print(f"{'='*60}\n")


def metrics_to_dict(metrics: EvalMetrics) -> dict:
    """Convert EvalMetrics to a JSON-serializable dict."""
    return {
        "accuracy": metrics.accuracy,
        "total_problems": metrics.total_problems,
        "correct_count": metrics.correct_count,
        "avg_tokens": metrics.avg_tokens,
        "cost_per_correct": metrics.cost_per_correct,
        "avg_steps": metrics.avg_steps,
        "accuracy_by_level": {str(k): v for k, v in metrics.accuracy_by_level.items()},
        "avg_tokens_by_level": {str(k): v for k, v in metrics.avg_tokens_by_level.items()},
        "count_by_level": {str(k): v for k, v in metrics.count_by_level.items()},
        "accuracy_by_subject": metrics.accuracy_by_subject,
        "action_counts_by_level": {
            str(k): v for k, v in metrics.action_counts_by_level.items()
        },
        "action_fractions_by_level": {
            str(k): v for k, v in metrics.action_fractions_by_level.items()
        },
    }
