"""Reward computation for GRPO training.

Combines binary outcome reward with a dense per-segment progress signal
inspired by MRT (Qu et al., 2025).  Progress measures the change in
meta-prover accuracy before vs. after each segment, rewarding segments
that move the model closer to the correct answer and penalising those
that move it further away.

Includes batched variants that share the prompt KV cache across probes
from the same question, avoiding redundant prompt computation.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.gsm8k import extract_predicted_number, grade_answer
from src.generation import TrajectoryRecord, _clone_cache


# ── Outcome reward ──────────────────────────────────────────────────


def compute_outcome_reward(
    trajectory: TrajectoryRecord,
    gold_answer: str,
) -> float:
    """Binary correctness reward: 1.0 if answer matches gold, else 0.0."""
    return 1.0 if grade_answer(trajectory.answer_number, gold_answer) else 0.0


# ── Combined reward ─────────────────────────────────────────────────


def compute_trajectory_reward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    trajectory: TrajectoryRecord,
    gold_answer: str,
    alpha: float = 0.5,
    num_probes: int = 1,
    max_answer_tokens: int = 200,
) -> tuple[float, list[float]]:
    """Compute total reward for a trajectory.

    R(tau) = correctness + alpha * sum_j r_prg(z_j)

    Args:
        model: current policy model (used as meta-prover).
        tokenizer: tokenizer with special tokens.
        trajectory: the generated trajectory.
        gold_answer: gold answer string.
        alpha: weight on dense progress signal.
        num_probes: answer samples per prefix (1 = greedy, cheap).
        max_answer_tokens: cap on answer generation length.

    Returns:
        (total_reward, per_segment_progress) tuple.
    """
    outcome = compute_outcome_reward(trajectory, gold_answer)
    if alpha == 0:
        # Skip expensive progress probing when not used
        return outcome, []
    progress = compute_segment_progress(
        model, tokenizer, trajectory, gold_answer,
        num_probes=num_probes,
        max_answer_tokens=max_answer_tokens,
    )
    total = outcome + alpha * sum(progress)
    return total, progress


# ── Per-segment progress reward ─────────────────────────────────────


@torch.no_grad()
def compute_segment_progress(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    trajectory: TrajectoryRecord,
    gold_answer: str,
    num_probes: int = 1,
    max_answer_tokens: int = 200,
) -> list[float]:
    """Compute per-segment progress rewards for a trajectory.

    For each segment boundary j, we force the model to terminate its
    thinking (append </think>) and greedily decode an answer.  Progress
    for segment j is:

        r_prg(z_j) = accuracy(mu | context_after_j)
                    - accuracy(mu | context_before_j)

    When num_probes == 1 (default), accuracy is binary {0, 1} and
    progress is in {-1, 0, +1}.  With num_probes > 1, accuracy is
    the fraction correct, giving a smoother but more expensive signal.

    Returns:
        List of floats, one per segment.
    """
    device = next(model.parameters()).device
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id

    # Reconstruct the prefix *as it existed in the KV cache* at each
    # segment boundary.  Walk through segments, applying backoff
    # truncations exactly as the generator did.
    prefixes: list[list[int]] = []
    active = list(trajectory.prompt_ids)

    # Before first segment: accuracy given just the prompt
    prefixes.append(list(active))

    for seg in trajectory.segments:
        # Append this segment's chunk tokens
        active = active + seg.chunk_ids

        if seg.depth is not None:
            # Backoff happened: truncate to rewind_pos, append directive
            active = active[:seg.rewind_pos] + seg.directive_ids
        # For continue/terminate: action token enters cache but we don't
        # include it in the prefix for probing — the meta-prover sees
        # the reasoning content, not the action markers.

        prefixes.append(list(active))

    # Evaluate meta-prover accuracy at each prefix
    accuracies = [
        _probe_accuracy(
            model, tokenizer, prefix, think_close_id, eos_id,
            gold_answer, num_probes, max_answer_tokens, device,
        )
        for prefix in prefixes
    ]

    # Progress = delta accuracy across each segment
    return [accuracies[j + 1] - accuracies[j]
            for j in range(len(trajectory.segments))]


def _probe_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix_ids: list[int],
    think_close_id: int,
    eos_id: int,
    gold_answer: str,
    num_probes: int,
    max_tokens: int,
    device: torch.device,
) -> float:
    """Force-terminate thinking at a prefix and check answer accuracy.

    Appends </think> to the prefix, greedily decodes an answer, and
    checks against the gold.  Returns fraction correct over num_probes
    attempts (for num_probes=1 this is just 0.0 or 1.0).
    """
    input_ids = prefix_ids + [think_close_id]

    correct = 0
    for _ in range(num_probes):
        # Prefill the prefix + </think> in one pass, then decode
        # autoregressively with KV cache.
        input_t = torch.tensor([input_ids], device=device)
        out = model(input_ids=input_t, use_cache=True)
        cache = out.past_key_values
        next_logits = out.logits[:, -1:, :]

        answer_ids: list[int] = []
        for _ in range(max_tokens):
            token = next_logits[0, -1, :].argmax().item()
            answer_ids.append(token)
            if token == eos_id:
                break
            token_t = torch.tensor([[token]], device=device)
            out = model(token_t, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_logits = out.logits[:, -1:, :]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        answer_number = extract_predicted_number(answer_text)

        if grade_answer(answer_number, gold_answer):
            correct += 1

    return correct / num_probes


# ── Batched reward computation ─────────────────────────────────────


def compute_batch_trajectory_reward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    trajectories: list[TrajectoryRecord],
    gold_answers: list[str],
    alpha: float = 0.5,
    num_probes: int = 1,
    max_answer_tokens: int = 200,
) -> list[tuple[float, list[float]]]:
    """Compute rewards for multiple trajectories with shared prompt caching.

    Groups trajectories by prompt (same question), prefills the shared
    prompt once per group, then extends each probe individually.

    Returns list of (total_reward, per_segment_progress) tuples.
    """
    outcomes = [
        compute_outcome_reward(traj, gold)
        for traj, gold in zip(trajectories, gold_answers)
    ]

    if alpha == 0:
        return [(o, []) for o in outcomes]

    all_progress = compute_batch_segment_progress(
        model, tokenizer, trajectories, gold_answers,
        num_probes=num_probes,
        max_answer_tokens=max_answer_tokens,
    )

    results = []
    for outcome, progress in zip(outcomes, all_progress):
        total = outcome + alpha * sum(progress)
        results.append((total, progress))
    return results


@torch.no_grad()
def compute_batch_segment_progress(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    trajectories: list[TrajectoryRecord],
    gold_answers: list[str],
    num_probes: int = 1,
    max_answer_tokens: int = 200,
) -> list[list[float]]:
    """Compute per-segment progress for multiple trajectories.

    Shares the prompt KV cache across all probes from trajectories with
    the same prompt (i.e., same question).  This avoids redundant prompt
    prefill — for Q questions × G rollouts × ~S segments, saves
    Q×G×S prompt prefills down to Q prefills.
    """
    device = next(model.parameters()).device
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id

    # ── Group trajectories by prompt ──
    prompt_groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, traj in enumerate(trajectories):
        key = tuple(traj.prompt_ids)
        prompt_groups[key].append(idx)

    # ── Results array ──
    all_progress: list[list[float] | None] = [None] * len(trajectories)

    for prompt_key, traj_indices in prompt_groups.items():
        prompt_ids = list(prompt_key)

        # Prefill the shared prompt once
        prompt_t = torch.tensor([prompt_ids], device=device)
        prompt_out = model(input_ids=prompt_t, use_cache=True)
        prompt_cache = prompt_out.past_key_values

        for traj_idx in traj_indices:
            traj = trajectories[traj_idx]
            gold = gold_answers[traj_idx]

            # Reconstruct prefixes for this trajectory
            prefixes = _build_probe_prefixes(traj)

            # Evaluate accuracy at each prefix by extending from prompt cache
            accuracies = []
            for prefix in prefixes:
                acc = _probe_accuracy_with_cache(
                    model, tokenizer, prefix, prompt_ids, prompt_cache,
                    think_close_id, eos_id, gold,
                    num_probes, max_answer_tokens, device,
                )
                accuracies.append(acc)

            # Progress = delta accuracy
            progress = [
                accuracies[j + 1] - accuracies[j]
                for j in range(len(traj.segments))
            ]
            all_progress[traj_idx] = progress

    return all_progress


def _build_probe_prefixes(trajectory: TrajectoryRecord) -> list[list[int]]:
    """Reconstruct prefix sequences at each segment boundary.

    Returns len(segments)+1 prefixes, identical to the logic in
    compute_segment_progress.
    """
    prefixes: list[list[int]] = []
    active = list(trajectory.prompt_ids)
    prefixes.append(list(active))

    for seg in trajectory.segments:
        active = active + seg.chunk_ids
        if seg.depth is not None:
            active = active[:seg.rewind_pos] + seg.directive_ids
        prefixes.append(list(active))

    return prefixes


def _probe_accuracy_with_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix_ids: list[int],
    prompt_ids: list[int],
    prompt_cache,
    think_close_id: int,
    eos_id: int,
    gold_answer: str,
    num_probes: int,
    max_tokens: int,
    device: torch.device,
) -> float:
    """Probe accuracy by extending from a shared prompt cache.

    Instead of prefilling the full prefix from scratch, clones the
    prompt cache and only processes the delta tokens (thinking content
    beyond the prompt + </think>).
    """
    # Delta = tokens after the shared prompt, plus </think>
    delta_ids = prefix_ids[len(prompt_ids):] + [think_close_id]

    correct = 0
    for _ in range(num_probes):
        cache = _clone_cache(prompt_cache)

        # Extend cache with delta tokens
        delta_t = torch.tensor([delta_ids], device=device)
        out = model(delta_t, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_logits = out.logits[:, -1:, :]

        # Greedy decode the answer
        answer_ids: list[int] = []
        for _ in range(max_tokens):
            token = next_logits[0, -1, :].argmax().item()
            answer_ids.append(token)
            if token == eos_id:
                break
            token_t = torch.tensor([[token]], device=device)
            out = model(token_t, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_logits = out.logits[:, -1:, :]

        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        answer_number = extract_predicted_number(answer_text)
        if grade_answer(answer_number, gold_answer):
            correct += 1

    return correct / num_probes
