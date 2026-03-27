"""GRPO training step with clipped surrogate loss and KL penalty.

Staged pipeline: generate-all → reward-all → advantages → batched
log-probs → single backward.  Uses batched operations from
logprobs.py and reward.py for ~5-7x speedup over the sequential
per-question, per-trajectory approach.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import BackoffConfig
from src.generation import TrajectoryRecord
from src.train.logprobs import compute_batch_logprobs
from src.train.reward import compute_batch_trajectory_reward
from src.train.rollout import generate_rollouts


@dataclass
class GRPOStepStats:
    """Statistics from one GRPO step."""

    loss: float
    avg_reward: float
    accuracy: float
    backoff_rate: float
    avg_progress: float
    avg_rho: float
    num_trajectories: int


def grpo_step(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: dict[str, int],
    questions: list[str],
    gold_answers: list[str],
    config: BackoffConfig,
    optimizer: torch.optim.Optimizer,
    step: int,
    max_grad_norm: float = 1.0,
) -> GRPOStepStats:
    """One GRPO training step over a batch of questions.

    Staged pipeline:
    1. Generate G rollouts per question (no grad, batched within question)
    2. Compute all rewards in batch (shared prompt cache)
    3. Compute group-relative advantages per question
    4. Batched log-prob computation for policy + ref models
    5. Compute total loss, single backward, optimizer step

    Args:
        model: trainable model (LoRA-wrapped).
        ref_model: frozen reference model (Phase 1 merged).
        tokenizer: tokenizer with special tokens.
        token_ids: token name -> ID mapping.
        questions: batch of question strings.
        gold_answers: corresponding gold answer strings.
        config: hyperparameters.
        optimizer: optimizer for model params.
        step: current training step.
        max_grad_norm: gradient clipping norm.
    """
    device = next(model.parameters()).device

    # ── Stage 1: Generate all rollouts ──
    model.eval()
    all_trajectories: list[TrajectoryRecord] = []
    all_question_indices: list[int] = []  # which question each traj belongs to

    for q_idx, question in enumerate(questions):
        trajectories = generate_rollouts(
            model, tokenizer, token_ids, question, config,
            num_rollouts=config.num_rollouts,
        )
        for traj in trajectories:
            all_trajectories.append(traj)
            all_question_indices.append(q_idx)

    # ── Stage 2: Compute all rewards in batch ──
    all_gold = [gold_answers[qi] for qi in all_question_indices]
    reward_results = compute_batch_trajectory_reward(
        model, tokenizer, all_trajectories, all_gold,
        alpha=config.alpha,
        num_probes=config.num_probes,
    )

    # Collect stats
    all_rewards = []
    all_correct = []
    all_backoffs = []
    all_progress = []
    for i, (reward, seg_progress) in enumerate(reward_results):
        all_rewards.append(reward)
        all_correct.append(
            1.0 if grade_answer_quick(all_trajectories[i], all_gold[i]) else 0.0
        )
        all_backoffs.append(1.0 if all_trajectories[i].backoff_count > 0 else 0.0)
        all_progress.append(sum(seg_progress))

    # ── Stage 3: Group-relative advantages per question ──
    # Map: trajectory index → advantage (None if skipped)
    advantages: list[float | None] = [None] * len(all_trajectories)

    for q_idx in range(len(questions)):
        # Indices of trajectories for this question
        group_indices = [
            i for i, qi in enumerate(all_question_indices) if qi == q_idx
        ]
        group_rewards = torch.tensor(
            [all_rewards[i] for i in group_indices], dtype=torch.float32
        )
        mean_r = group_rewards.mean()
        std_r = group_rewards.std()
        if std_r < 1e-8:
            continue  # no learning signal
        group_adv = (group_rewards - mean_r) / std_r
        for gi, adv in zip(group_indices, group_adv):
            if abs(adv.item()) >= 1e-6:
                advantages[gi] = adv.item()

    # Filter to trajectories with non-trivial advantages
    filtered_indices = [i for i, a in enumerate(advantages) if a is not None]
    filtered_trajs = [all_trajectories[i] for i in filtered_indices]
    filtered_advs = [advantages[i] for i in filtered_indices]

    if not filtered_trajs:
        # No learning signal — still step optimizer to stay in sync
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        n = len(all_rewards) if all_rewards else 1
        return GRPOStepStats(
            loss=0.0,
            avg_reward=sum(all_rewards) / n,
            accuracy=sum(all_correct) / n,
            backoff_rate=sum(all_backoffs) / n,
            avg_progress=sum(all_progress) / n,
            avg_rho=0.0,
            num_trajectories=len(all_rewards),
        )

    # ── Stage 4: Batched log-prob computation ──
    model.train()

    # Policy model log-probs (with grad)
    policy_lps = compute_batch_logprobs(
        model, filtered_trajs, token_ids, b_max=config.b_max
    )

    # Ref model log-probs (no grad)
    with torch.no_grad():
        ref_lps = compute_batch_logprobs(
            ref_model, filtered_trajs, token_ids, b_max=config.b_max
        )

    # ── Stage 5: Compute total loss, single backward ──
    optimizer.zero_grad()

    scale = 1.0 / (len(questions) * config.num_rollouts)
    total_loss = torch.zeros(1, device=device)
    all_losses = []
    all_rho = []

    for idx, (log_pi_theta, log_pi_ref) in enumerate(zip(policy_lps, ref_lps)):
        adv = torch.tensor(filtered_advs[idx], device=device)
        traj = filtered_trajs[idx]

        # Importance ratio
        rho = torch.exp(log_pi_theta - traj.stored_log_prob)
        all_rho.append(rho.item())

        # Clipped surrogate loss
        surr1 = rho * adv
        surr2 = torch.clamp(
            rho, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
        ) * adv
        policy_loss = -torch.min(surr1, surr2)

        # KL penalty
        kl = log_pi_theta - log_pi_ref
        loss = policy_loss + config.kl_coef * kl

        total_loss = total_loss + loss * scale
        all_losses.append(loss.item())

    total_loss.backward()

    # ── Gradient clip + optimizer step ──
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # ── Stats ──
    n = len(all_rewards) if all_rewards else 1
    return GRPOStepStats(
        loss=sum(all_losses) / max(len(all_losses), 1),
        avg_reward=sum(all_rewards) / n,
        accuracy=sum(all_correct) / n,
        backoff_rate=sum(all_backoffs) / n,
        avg_progress=sum(all_progress) / n,
        avg_rho=sum(all_rho) / max(len(all_rho), 1),
        num_trajectories=len(all_rewards),
    )


def grade_answer_quick(traj: TrajectoryRecord, gold: str) -> bool:
    """Quick correctness check without importing grade_answer again."""
    from src.data.gsm8k import grade_answer
    return grade_answer(traj.answer_number, gold)
