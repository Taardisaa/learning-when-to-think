"""Segment-wise forward pass for trajectory log-probability computation.

For a trajectory with k backoffs, performs k+1 forward passes with the
correct KV cache context for each segment. Handles action/depth token
masking to match the distribution used during generation.

Includes a batched variant (`compute_batch_logprobs`) that processes
multiple trajectories in parallel for ~3-5x speedup.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.generation import TrajectoryRecord
from src.tokens import ACTION_TOKENS, DEPTH_TOKENS, TERMINATE_TOKEN


# ── Batched log-prob computation ────────────────────────────────────


def compute_batch_logprobs(
    model: PreTrainedModel,
    trajectories: list[TrajectoryRecord],
    token_ids: dict[str, int],
    b_max: int = 2,
    max_bucket_size: int = 16,
) -> list[torch.Tensor]:
    """Compute log-probabilities for multiple trajectories in batched forward passes.

    Collects all forward-pass sequences from all trajectories, groups them
    by similar length to minimise padding, and runs batched model calls.

    Args:
        model: the model to evaluate (can be LoRA-wrapped).
        trajectories: list of generated trajectories.
        token_ids: mapping from token name to ID.
        b_max: max backoffs — controls action masking per boundary.
        max_bucket_size: max sequences per batched forward pass.

    Returns:
        List of scalar tensors (one per trajectory), each the total
        log-probability of that trajectory under the model.  Gradients
        flow through if the model has grad enabled.
    """
    if not trajectories:
        return []

    device = next(model.parameters()).device
    depth_ids = [token_ids[t] for t in DEPTH_TOKENS]

    # ── Step 1: collect all passes, tagged with trajectory index ──
    tagged_passes: list[tuple[int, list[int], int, dict]] = []
    for traj_idx, traj in enumerate(trajectories):
        for seq_ids, score_start, token_meta in _build_passes(traj, token_ids, b_max):
            tagged_passes.append((traj_idx, seq_ids, score_start, token_meta))

    if not tagged_passes:
        return [torch.zeros(1, device=device).squeeze() for _ in trajectories]

    # ── Step 2: sort by length, bucket ──
    tagged_passes.sort(key=lambda x: len(x[1]))
    buckets: list[list[tuple[int, list[int], int, dict]]] = []
    for i in range(0, len(tagged_passes), max_bucket_size):
        buckets.append(tagged_passes[i : i + max_bucket_size])

    # ── Step 3: process each bucket as a batch ──
    # Accumulate per-trajectory log-probs
    traj_lps: list[torch.Tensor] = [
        torch.zeros(1, device=device, dtype=torch.float32) for _ in trajectories
    ]

    pad_id = 0  # padding token (value doesn't matter with attention_mask)

    for bucket in buckets:
        max_len = max(len(seq_ids) for _, seq_ids, _, _ in bucket)
        batch_size = len(bucket)

        # Build padded input_ids and attention_mask (left-unpadded, right-padded)
        input_ids_batch = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )

        for b, (_, seq_ids, _, _) in enumerate(bucket):
            seq_len = len(seq_ids)
            input_ids_batch[b, :seq_len] = torch.tensor(seq_ids, dtype=torch.long)
            attention_mask[b, :seq_len] = 1

        # Batched forward pass
        logits = model(
            input_ids=input_ids_batch, attention_mask=attention_mask
        ).logits  # [B, max_len, vocab]

        # ── Step 4: score each sequence ──
        for b, (traj_idx, seq_ids, score_start, token_meta) in enumerate(bucket):
            seq_logits = logits[b]  # [max_len, vocab]

            for pos in range(score_start, len(seq_ids)):
                meta = token_meta[pos]
                if meta == "skip":
                    continue

                target = seq_ids[pos]
                l = seq_logits[pos - 1]

                if meta == "action":
                    allowed = token_meta.get(f"allowed_{pos}")
                    lp = _masked_log_prob(l, target, allowed)
                elif meta == "depth":
                    lp = _masked_log_prob(l, target, depth_ids)
                else:
                    lp = F.log_softmax(l, dim=-1)[target]

                traj_lps[traj_idx] = traj_lps[traj_idx] + lp

    return [lp.squeeze() for lp in traj_lps]


# ── Single-trajectory log-prob computation ──────────────────────────


def compute_trajectory_logprobs(
    model: PreTrainedModel,
    trajectory: TrajectoryRecord,
    token_ids: dict[str, int],
    b_max: int = 2,
) -> torch.Tensor:
    """Compute total log-probability of a trajectory under the model.

    For trajectories with k backoffs, performs k+1 forward passes using
    the correct context for each segment. Action tokens get masked
    log-probs; depth tokens get masked log-probs; chunk/directive tokens
    get full-vocab log-probs.

    Args:
        model: the model to evaluate (can be LoRA-wrapped).
        trajectory: the generated trajectory.
        token_ids: mapping from token name to ID.
        b_max: max backoffs — controls action masking per boundary.

    Returns:
        Scalar tensor (with gradients) = sum of log-probs for all
        policy decisions in the trajectory.
    """
    device = next(model.parameters()).device

    action_ids = [token_ids[t] for t in ACTION_TOKENS]
    action_id_set = set(action_ids)
    depth_ids = [token_ids[t] for t in DEPTH_TOKENS]
    depth_id_set = set(depth_ids)
    backoff_id = token_ids["<backoff>"]
    continue_id = token_ids["<continue>"]
    terminate_id = token_ids[TERMINATE_TOKEN]

    passes = _build_passes(trajectory, token_ids, b_max)

    total_lp = torch.zeros(1, device=device, dtype=torch.float32)

    for seq_ids, score_start, token_meta in passes:
        input_ids = torch.tensor([seq_ids], device=device)
        logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]

        for pos in range(score_start, len(seq_ids)):
            meta = token_meta[pos]
            if meta == "skip":
                continue

            target = seq_ids[pos]
            # logits[pos-1] predicts token at position pos
            l = logits[pos - 1]

            if meta == "action":
                # Masked to allowed actions at this boundary
                allowed = token_meta.get(f"allowed_{pos}", action_ids)
                lp = _masked_log_prob(l, target, allowed)
            elif meta == "depth":
                lp = _masked_log_prob(l, target, depth_ids)
            else:
                # Full vocab (chunk or directive token)
                lp = F.log_softmax(l, dim=-1)[target]

            total_lp = total_lp + lp

    return total_lp.squeeze()


def _masked_log_prob(
    logits: torch.Tensor, target: int, allowed_ids: list[int]
) -> torch.Tensor:
    """Compute log-prob under a masked distribution."""
    mask = torch.full_like(logits, float("-inf"))
    mask[allowed_ids] = 0.0
    return F.log_softmax(logits + mask, dim=-1)[target]


def _build_passes(
    trajectory: TrajectoryRecord,
    token_ids: dict[str, int],
    b_max: int,
) -> list[tuple[list[int], int, dict]]:
    """Build forward pass info from trajectory segments.

    Returns list of (seq_ids, score_start, token_meta) where:
    - seq_ids: full token sequence for this forward pass
    - score_start: index of first position to score
    - token_meta: dict mapping position -> type string
      ("token", "action", "depth", "skip") plus "allowed_{pos}" for action masks
    """
    action_ids = [token_ids[t] for t in ACTION_TOKENS]
    backoff_id = token_ids["<backoff>"]
    continue_id = token_ids["<continue>"]
    terminate_id = token_ids[TERMINATE_TOKEN]

    active_prefix = list(trajectory.prompt_ids)
    current_tokens: list[int] = []
    score_start = len(active_prefix)
    meta: dict = {}
    backoff_count = 0

    passes = []

    for seg in trajectory.segments:
        base_pos = len(active_prefix) + len(current_tokens)

        # Chunk tokens — full vocab
        for i, tok in enumerate(seg.chunk_ids):
            meta[base_pos + i] = "token"
        current_tokens.extend(seg.chunk_ids)

        if seg.action == -1:
            # Partial segment (T_max reached) — no action to score
            continue

        # Action token — masked distribution
        action_pos = len(active_prefix) + len(current_tokens)
        meta[action_pos] = "action"

        # Determine which actions were allowed at this boundary
        if backoff_count < b_max:
            allowed = list(action_ids)
        else:
            allowed = [a for a in action_ids if a != backoff_id]
        meta[f"allowed_{action_pos}"] = allowed

        current_tokens.append(seg.action)

        if seg.depth is not None:
            # Depth token — masked distribution
            depth_pos = len(active_prefix) + len(current_tokens)
            meta[depth_pos] = "depth"
            current_tokens.append(seg.depth)

            backoff_count += 1

            # Directive tokens are generated BEFORE truncation, so they
            # belong in Pass 1 (with the wrong tokens still in context).
            # Score them here under the pre-truncation context.
            for i, tok in enumerate(seg.directive_ids):
                dir_pos = len(active_prefix) + len(current_tokens) + i
                if seg.force_continued and i == len(seg.directive_ids) - 1:
                    meta[dir_pos] = "skip"
                else:
                    meta[dir_pos] = "token"
            current_tokens.extend(seg.directive_ids)

            # Close this pass (includes chunk + action + depth + directive)
            passes.append((
                active_prefix + current_tokens,
                score_start,
                dict(meta),
            ))

            # Reset for next pass: after truncation, the directive is
            # re-injected into the clean cache as a prefix.
            full_seq = active_prefix + current_tokens
            active_prefix = full_seq[:seg.rewind_pos] + list(seg.directive_ids)
            score_start = len(active_prefix)
            current_tokens = []
            meta = {}

    # Final pass
    if current_tokens:
        passes.append((
            active_prefix + current_tokens,
            score_start,
            dict(meta),
        ))

    return passes
