"""Tests for evaluation harness."""

import math

import pytest

from src.pivot.eval_harness import evaluate, metrics_to_dict
from src.pivot.eval_types import RolloutResult


def _make_result(
    correct: bool,
    level: int = 1,
    subject: str = "algebra",
    num_tokens: int = 500,
    actions: list | None = None,
    prompt_id: str = "test",
) -> RolloutResult:
    return RolloutResult(
        prompt_id=prompt_id,
        question="What is 1+1?",
        gold_answer="2",
        level=level,
        subject=subject,
        generated_text="\\boxed{2}" if correct else "\\boxed{3}",
        predicted_answer="2" if correct else "3",
        correct=correct,
        num_tokens=num_tokens,
        actions=actions or [],
    )


class TestEvaluate:
    def test_perfect_accuracy(self):
        results = [_make_result(True, num_tokens=100) for _ in range(10)]
        m = evaluate(results)
        assert m.accuracy == 1.0
        assert m.correct_count == 10
        assert m.total_problems == 10
        assert m.avg_tokens == 100.0
        assert m.cost_per_correct == 100.0

    def test_zero_accuracy(self):
        results = [_make_result(False, num_tokens=200) for _ in range(5)]
        m = evaluate(results)
        assert m.accuracy == 0.0
        assert m.correct_count == 0
        assert m.cost_per_correct == float("inf")

    def test_mixed_accuracy(self):
        results = [
            _make_result(True, num_tokens=100),
            _make_result(False, num_tokens=300),
        ]
        m = evaluate(results)
        assert m.accuracy == 0.5
        assert m.avg_tokens == 200.0
        assert m.cost_per_correct == 400.0  # 400 total / 1 correct

    def test_by_level_breakdown(self):
        results = [
            _make_result(True, level=1),
            _make_result(True, level=1),
            _make_result(False, level=5),
            _make_result(False, level=5),
        ]
        m = evaluate(results)
        assert m.accuracy_by_level[1] == 1.0
        assert m.accuracy_by_level[5] == 0.0
        assert m.count_by_level[1] == 2
        assert m.count_by_level[5] == 2

    def test_by_subject_breakdown(self):
        results = [
            _make_result(True, subject="algebra"),
            _make_result(False, subject="geometry"),
        ]
        m = evaluate(results)
        assert m.accuracy_by_subject["algebra"] == 1.0
        assert m.accuracy_by_subject["geometry"] == 0.0

    def test_action_counts(self):
        results = [
            _make_result(True, level=1, actions=["terminate"]),
            _make_result(True, level=1, actions=["continue", "terminate"]),
            _make_result(False, level=5, actions=["continue", "refine", "continue", "terminate"]),
        ]
        m = evaluate(results)
        # Level 1: 1 continue, 2 terminate
        assert m.action_counts_by_level[1] == {"terminate": 2, "continue": 1}
        assert abs(m.action_fractions_by_level[1]["terminate"] - 2 / 3) < 1e-9
        # Level 5: 2 continue, 1 refine, 1 terminate
        assert m.action_counts_by_level[5]["continue"] == 2
        assert m.action_counts_by_level[5]["refine"] == 1
        assert m.action_counts_by_level[5]["terminate"] == 1

    def test_no_actions_skips_action_analysis(self):
        results = [_make_result(True), _make_result(False)]
        m = evaluate(results)
        assert m.action_counts_by_level == {}
        assert m.action_fractions_by_level == {}

    def test_avg_steps(self):
        results = [
            _make_result(True, actions=["continue", "terminate"]),
            _make_result(True, actions=["terminate"]),
        ]
        m = evaluate(results)
        assert m.avg_steps == 1.5  # (2+1)/2

    def test_single_problem(self):
        results = [_make_result(True, level=3, subject="geometry", num_tokens=42)]
        m = evaluate(results)
        assert m.accuracy == 1.0
        assert m.avg_tokens == 42.0
        assert m.count_by_level == {3: 1}

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            evaluate([])


class TestMetricsToDict:
    def test_serializable(self):
        import json

        results = [
            _make_result(True, level=1, actions=["terminate"]),
            _make_result(False, level=3, actions=["continue", "terminate"]),
        ]
        m = evaluate(results)
        d = metrics_to_dict(m)
        # Should not raise
        s = json.dumps(d)
        assert isinstance(s, str)
        # Keys are strings (JSON requirement for dict keys)
        assert "1" in d["accuracy_by_level"]
