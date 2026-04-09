"""Tests for ALP reward function."""

import pytest

from src.pivot.reward import (
    compute_alp_rewards,
    compute_alp_rewards_batch,
    compute_group_solve_rate,
)

BETA = 0.05
L_MAX = 2048


class TestGroupSolveRate:
    def test_all_correct(self):
        assert compute_group_solve_rate([True, True, True, True]) == 1.0

    def test_all_wrong(self):
        assert compute_group_solve_rate([False, False, False, False]) == 0.0

    def test_mixed(self):
        assert compute_group_solve_rate([True, False, True, False]) == 0.5

    def test_single_correct(self):
        assert compute_group_solve_rate([True]) == 1.0

    def test_empty(self):
        assert compute_group_solve_rate([]) == 0.0


class TestALPRewards:
    def test_all_correct_equal_length(self):
        n = 512
        rewards = compute_alp_rewards(
            [True, True, True, True], [n, n, n, n], beta=BETA, l_max=L_MAX
        )
        expected = 1.0 - BETA * 1.0 * n / L_MAX
        assert all(abs(r - expected) < 1e-9 for r in rewards)

    def test_all_wrong(self):
        # SR=0, so penalty=0, reward=0 for all
        rewards = compute_alp_rewards(
            [False, False, False], [500, 1000, 1500], beta=BETA, l_max=L_MAX
        )
        assert rewards == [0.0, 0.0, 0.0]

    def test_mixed_group(self):
        rewards = compute_alp_rewards(
            [True, False, True, False], [512, 1024, 256, 2048], beta=BETA, l_max=L_MAX
        )
        sr = 0.5
        assert len(rewards) == 4
        # Correct rollouts: 1.0 - penalty
        assert abs(rewards[0] - (1.0 - BETA * sr * 512 / L_MAX)) < 1e-9
        assert abs(rewards[2] - (1.0 - BETA * sr * 256 / L_MAX)) < 1e-9
        # Wrong rollouts: 0.0 - penalty (negative)
        assert abs(rewards[1] - (0.0 - BETA * sr * 1024 / L_MAX)) < 1e-9
        assert abs(rewards[3] - (0.0 - BETA * sr * 2048 / L_MAX)) < 1e-9

    def test_tokens_at_l_max(self):
        rewards = compute_alp_rewards([True, True], [L_MAX, L_MAX], beta=BETA, l_max=L_MAX)
        expected = 1.0 - BETA * 1.0  # penalty = beta * SR * 1.0
        assert all(abs(r - expected) < 1e-9 for r in rewards)

    def test_tokens_exceed_l_max(self):
        # penalty can exceed beta * SR (no clamping on ratio)
        rewards = compute_alp_rewards([True], [L_MAX * 2], beta=BETA, l_max=L_MAX)
        expected = 1.0 - BETA * 1.0 * 2.0
        assert abs(rewards[0] - expected) < 1e-9

    def test_single_rollout_correct(self):
        rewards = compute_alp_rewards([True], [100], beta=BETA, l_max=L_MAX)
        expected = 1.0 - BETA * 1.0 * 100 / L_MAX
        assert abs(rewards[0] - expected) < 1e-9

    def test_single_rollout_wrong(self):
        rewards = compute_alp_rewards([False], [100], beta=BETA, l_max=L_MAX)
        assert rewards == [0.0]  # SR=0, penalty=0

    def test_empty(self):
        assert compute_alp_rewards([], [], beta=BETA, l_max=L_MAX) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_alp_rewards([True, False], [100], beta=BETA, l_max=L_MAX)

    def test_negative_rewards_possible(self):
        # Wrong rollout on easy prompt -> negative reward
        rewards = compute_alp_rewards(
            [True, True, True, False], [100, 100, 100, 1500], beta=BETA, l_max=L_MAX
        )
        assert rewards[3] < 0.0


class TestALPRewardsBatch:
    def test_multiple_groups(self):
        groups = [
            [(True, 512), (False, 1024)],
            [(False, 256), (False, 512)],
        ]
        results = compute_alp_rewards_batch(groups, beta=BETA, l_max=L_MAX)
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2
        # Second group: all wrong, SR=0, all rewards = 0
        assert results[1] == [0.0, 0.0]

    def test_empty_batch(self):
        assert compute_alp_rewards_batch([], beta=BETA, l_max=L_MAX) == []
