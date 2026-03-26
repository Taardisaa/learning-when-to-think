"""Segment-wise forward pass for trajectory log-probability computation.

For a trajectory with k backoffs, performs k+1 forward passes with the
correct KV cache context for each segment. Handles action/depth token
masking to match the distribution used during generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.generation import TrajectoryRecord
from src.tokens import ACTION_TOKENS, DEPTH_TOKENS, TERMINATE_TOKEN


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

            # Close this pass
            passes.append((
                active_prefix + current_tokens,
                score_start,
                dict(meta),
            ))

            # Reset for next pass
            full_seq = active_prefix + current_tokens
            active_prefix = full_seq[:seg.rewind_pos]
            score_start = len(active_prefix)
            current_tokens = []
            meta = {}

            # Directive tokens — full vocab, except force-appended <continue>
            for i, tok in enumerate(seg.directive_ids):
                dir_pos = score_start + i
                if seg.force_continued and i == len(seg.directive_ids) - 1:
                    meta[dir_pos] = "skip"
                else:
                    meta[dir_pos] = "token"
            current_tokens.extend(seg.directive_ids)

    # Final pass
    if current_tokens:
        passes.append((
            active_prefix + current_tokens,
            score_start,
            dict(meta),
        ))

    return passes
