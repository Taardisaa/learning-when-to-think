"""GRPO training step with clipped surrogate loss and KL penalty."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import BackoffConfig
from src.generation import TrajectoryRecord
from src.train.logprobs import compute_trajectory_logprobs
from src.train.reward import compute_reward, anneal_lambda_explore
from src.train.rollout import generate_rollouts


@dataclass
class GRPOStepStats:
    """Statistics from one GRPO step."""

    loss: float
    avg_reward: float
    accuracy: float
    backoff_rate: float
    avg_t_net: float
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

    For each question:
    1. Generate G rollouts (no grad)
    2. Compute rewards, group-relative advantages
    3. For each rollout: clipped surrogate loss + KL penalty
    4. Accumulate gradients, clip, optimizer step

    Args:
        model: trainable model (LoRA-wrapped).
        ref_model: frozen reference model (Phase 1 merged).
        tokenizer: tokenizer with special tokens.
        token_ids: token name -> ID mapping.
        questions: batch of question strings.
        gold_answers: corresponding gold answer strings.
        config: hyperparameters.
        optimizer: optimizer for model params.
        step: current training step (for exploration annealing).
        max_grad_norm: gradient clipping norm.
    """
    device = next(model.parameters()).device
    lambda_explore = anneal_lambda_explore(
        config.lambda_explore, step, config.anneal_steps
    )

    all_losses = []
    all_rewards = []
    all_correct = []
    all_backoffs = []
    all_t_net = []
    all_rho = []

    optimizer.zero_grad()

    for question, gold in zip(questions, gold_answers):
        # ── Generate rollouts (no grad) ──
        model.eval()
        trajectories = generate_rollouts(
            model, tokenizer, token_ids, question, config,
            num_rollouts=config.num_rollouts,
        )
        model.train()

        # ── Compute rewards ──
        rewards = []
        for traj in trajectories:
            r = compute_reward(
                traj, gold,
                lambda_tok=config.lambda_tok,
                lambda_explore=lambda_explore,
            )
            rewards.append(r)
            all_rewards.append(r)
            all_correct.append(1.0 if traj.answer_number == gold else 0.0)
            all_backoffs.append(1.0 if traj.backoff_count > 0 else 0.0)
            all_t_net.append(traj.final_kv_length)

        # ── Group-relative advantages ──
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        if std_r < 1e-8:
            # All same reward — skip this question (no learning signal)
            continue
        advantages = (rewards_t - mean_r) / std_r

        # ── Compute loss for each rollout ──
        for traj, advantage in zip(trajectories, advantages):
            if abs(advantage.item()) < 1e-6:
                continue

            # New log-prob (with grad)
            log_pi_theta = compute_trajectory_logprobs(
                model, traj, token_ids, b_max=config.b_max
            )

            # Importance ratio
            rho = torch.exp(log_pi_theta - traj.stored_log_prob)
            all_rho.append(rho.item())

            # Clipped surrogate loss (maximize advantage → minimize negative)
            adv = advantage.to(device)
            surr1 = rho * adv
            surr2 = torch.clamp(rho, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2)

            # KL penalty: log π_θ - log π_ref
            with torch.no_grad():
                log_pi_ref = compute_trajectory_logprobs(
                    ref_model, traj, token_ids, b_max=config.b_max
                )
            kl = log_pi_theta - log_pi_ref

            # Total loss for this trajectory
            loss = policy_loss + config.kl_coef * kl

            # Scale by 1/(num_questions * num_rollouts) for averaging
            scale = 1.0 / (len(questions) * config.num_rollouts)
            (loss * scale).backward()

            all_losses.append(loss.item())

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
        avg_t_net=sum(all_t_net) / n,
        avg_rho=sum(all_rho) / max(len(all_rho), 1),
        num_trajectories=len(all_rewards),
    )
