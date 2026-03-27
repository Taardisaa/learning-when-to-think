from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.boundary import BoundaryTracker
from src.config import BackoffConfig
from src.data.gsm8k import extract_predicted_number
from src.tokens import ACTION_TOKENS, DEPTH_TOKENS, TERMINATE_TOKEN


@dataclass
class Segment:
    """One chunk of generation between boundary decisions."""

    chunk_ids: list[int]  # tokens generated in the chunk
    action: int  # action token id chosen at boundary
    depth: int | None = None  # depth token id (backoff only)
    directive_ids: list[int] = field(default_factory=list)  # directive tokens (backoff only)
    force_continued: bool = False  # True if directive hit k_dir and <continue> was forced
    kv_start_pos: int = 0  # cache position at start of chunk
    kv_end_pos: int = 0  # cache position at end of chunk (before action)
    rewind_pos: int | None = None  # cache position after backoff crop (backoff only)


@dataclass
class TrajectoryRecord:
    """Everything needed for GRPO training from one generation."""

    prompt_ids: list[int]
    segments: list[Segment]
    answer_ids: list[int]  # tokens generated after </think>
    final_kv_length: int  # net tokens at termination (T_net for reward)
    answer_text: str | None
    answer_number: str | None
    backoff_count: int
    total_generated_tokens: int
    stored_log_prob: float  # sum of log-probs at T=1 (importance ratio denominator)
    terminated: bool = True  # False if T_max reached without </think>

    def build_forward_pass_sequences(self) -> list[list[int]]:
        """Reconstruct token sequences for segment-wise log-prob computation.

        For a trajectory with k backoffs, returns k+1 sequences. Each sequence
        is the full token IDs that were in the KV cache during that forward
        pass — from the prompt through to the next backoff or termination.

        Pass boundaries occur at backoff events: everything before the crop
        (including chunk + <backoff> + <depth>) forms one pass, and the
        post-crop state (surviving prefix + directive + new chunks) forms
        the next.
        """
        sequences: list[list[int]] = []
        # "active" = tokens currently surviving in the KV cache
        active = list(self.prompt_ids)
        pass_tokens: list[int] = []  # tokens added in the current pass

        for seg in self.segments:
            pass_tokens.extend(seg.chunk_ids)
            pass_tokens.append(seg.action)

            if seg.depth is not None:
                # Backoff: depth + directive are generated in the pre-truncation
                # context (model sees the wrong tokens when writing the directive).
                # All belong in this pass.
                pass_tokens.append(seg.depth)
                pass_tokens.extend(seg.directive_ids)
                sequences.append(active + pass_tokens)

                # After truncation: surviving prefix + directive re-injected
                full_before_crop = active + pass_tokens
                active = full_before_crop[:seg.rewind_pos] + list(seg.directive_ids)

                # Next pass starts fresh (directive is now part of the prefix)
                pass_tokens = []

        # Final pass (everything after last backoff, or the only pass)
        if pass_tokens:
            sequences.append(active + pass_tokens)

        return sequences


# ── Cache snapshot for Qwen3.5 hybrid architecture ──


@dataclass
class _CacheSnapshot:
    """Snapshot of Qwen3.5's hybrid cache at a boundary position.

    Stores deep copies of the linear attention recurrent/conv states,
    the KV cache seq_length for the full attention layers, and the
    logits at this position (for directive generation after restore).

    Memory: ~10.5 MB per snapshot (recurrent states + logits).
    """

    seq_length: int
    conv_states: dict[int, torch.Tensor]  # layer_idx -> conv_state clone
    recurrent_states: dict[int, torch.Tensor]  # layer_idx -> recurrent_state clone
    logits: torch.Tensor  # [1, 1, vocab] — logits at this position


def _save_snapshot(cache, logits: torch.Tensor) -> _CacheSnapshot:
    """Save a snapshot of the hybrid cache's recurrent state + logits."""
    conv = {}
    recur = {}
    for i, ltype in enumerate(cache.layer_types):
        if ltype == "linear_attention":
            conv[i] = cache.conv_states[i].clone()
            recur[i] = cache.recurrent_states[i].clone()
    return _CacheSnapshot(
        seq_length=cache.get_seq_length(),
        conv_states=conv,
        recurrent_states=recur,
        logits=logits.clone(),
    )


def _restore_snapshot(cache, snapshot: _CacheSnapshot) -> torch.Tensor:
    """Restore cache to a prior snapshot, in-place. Returns saved logits.

    - Full attention layers: slice KV tensors to saved length.
    - Linear attention layers: replace recurrent/conv states from snapshot.
    """
    target = snapshot.seq_length

    # Full attention: slice KV cache
    for i in cache.transformer_layers:
        cache.key_cache[i] = cache.key_cache[i][:, :, :target, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :target, :]

    # Linear attention: restore recurrent state
    for i, cs in snapshot.conv_states.items():
        cache.conv_states[i] = cs.clone()
    for i, rs in snapshot.recurrent_states.items():
        cache.recurrent_states[i] = rs.clone()

    return snapshot.logits


def _clone_cache(cache) -> object:
    """Deep-clone a Qwen3_5DynamicCache so each rollout gets an independent copy.

    Clones all mutable state (KV tensors, conv/recurrent states) while
    sharing the immutable metadata (layer_types, transformer_layers).
    """
    clone = object.__new__(type(cache))

    # Immutable metadata — share references
    clone.layer_types = cache.layer_types
    clone.transformer_layers = cache.transformer_layers
    clone.last_linear_layer = cache.last_linear_layer
    # has_previous_state is a read-only property derived from conv_states

    # Mutable state — deep clone
    clone.key_cache = [k.clone() if k is not None else None for k in cache.key_cache]
    clone.value_cache = [v.clone() if v is not None else None for v in cache.value_cache]
    clone.conv_states = [c.clone() if c is not None else None for c in cache.conv_states]
    clone.recurrent_states = [r.clone() if r is not None else None for r in cache.recurrent_states]

    return clone


# ── Generator ──


class BackoffGenerator:
    """Backoff-aware generation loop for Qwen3.5 models.

    Uses cache snapshots at semantic boundaries. On backoff, restores the
    recurrent state from a saved snapshot and slices the full-attention KV
    cache — ~800x faster than re-encoding from scratch.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        token_ids: dict[str, int],
        config: BackoffConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.config = config

        # Pre-compute ID lists for masked sampling
        self.action_ids = [token_ids[t] for t in ACTION_TOKENS]
        self.depth_ids = [token_ids[t] for t in DEPTH_TOKENS]
        self.continue_id = token_ids["<continue>"]
        self.backoff_id = token_ids["<backoff>"]
        self.terminate_id = token_ids[TERMINATE_TOKEN]

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor) -> tuple:
        """Prefill prompt and return a reusable (cache, logits) pair.

        Call once per prompt, then use generate(..., prefill=...) for each
        rollout. Each rollout deep-clones the cache, avoiding redundant
        prefill forward passes.

        Returns (cache, logits) — keep these and pass to generate().
        """
        out = self.model(prompt_ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1:, :]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        temperature: float | None = None,
        prefill: tuple | None = None,
    ) -> TrajectoryRecord:
        """Generate a trajectory with backoff-aware decisions.

        Args:
            prompt_ids: [1, seq_len] tensor of tokenized prompt.
            temperature: sampling temperature. None uses config default.
            prefill: (cache, logits) from prefill(). If provided, the cache
                is deep-cloned so each rollout is independent — saves one
                full forward pass per rollout.
        """
        if temperature is None:
            temperature = self.config.temperature
        device = prompt_ids.device
        prompt_list = prompt_ids[0].tolist()
        prompt_len = len(prompt_list)

        if prefill is not None:
            cache = _clone_cache(prefill[0])
            next_logits = prefill[1].clone()
        else:
            out = self.model(prompt_ids, use_cache=True)
            cache = out.past_key_values
            next_logits = out.logits[:, -1:, :]

        # State tracking
        full_sequence = list(prompt_list)  # all token IDs in the cache
        boundaries = [prompt_len]  # cache positions at semantic boundaries
        boundary_tracker = BoundaryTracker(self.config.k_min, self.config.k_max)

        # Snapshot management: boundary_pos → snapshot
        # Save initial snapshot at prompt end
        snapshots: dict[int, _CacheSnapshot] = {
            prompt_len: _save_snapshot(cache, next_logits),
        }

        backoff_count = 0
        segments: list[Segment] = []
        total_log_prob = 0.0
        total_generated = 0

        while total_generated < self.config.t_max:
            # ── GENERATE CHUNK ──
            chunk_start_pos = len(full_sequence)
            chunk_ids: list[int] = []
            chunk_log_prob = 0.0

            for _ in range(self.config.k_max):
                token, lp = self._sample(next_logits, temperature, full_sequence)
                chunk_ids.append(token)
                chunk_log_prob += lp
                total_generated += 1
                full_sequence.append(token)

                # Forward pass: feed token, get logits for next position
                token_t = torch.tensor([[token]], device=device)
                out = self.model(token_t, past_key_values=cache, use_cache=True)
                next_logits = out.logits[:, -1:, :]

                # Check boundary
                tok_text = self.tokenizer.decode([token])
                if boundary_tracker.step(tok_text):
                    break

                if total_generated >= self.config.t_max:
                    break

            boundary_pos = len(full_sequence)
            boundaries.append(boundary_pos)

            # Save snapshot at this boundary (before action token)
            snapshots[boundary_pos] = _save_snapshot(cache, next_logits)
            # Prune old snapshots: only keep those still in the boundaries stack
            self._prune_snapshots(snapshots, boundaries)

            if total_generated >= self.config.t_max:
                # Record partial segment and bail
                segments.append(Segment(
                    chunk_ids=chunk_ids, action=-1,
                    kv_start_pos=chunk_start_pos, kv_end_pos=boundary_pos,
                ))
                total_log_prob += chunk_log_prob
                break

            # ── ACTION DECISION ──
            # Logits from last chunk token predict the action
            allowed = list(self.action_ids)
            if backoff_count >= self.config.b_max:
                allowed = [a for a in allowed if a != self.backoff_id]

            action, action_lp = self._sample_masked(next_logits, allowed, temperature, full_sequence)
            total_generated += 1
            full_sequence.append(action)

            # Feed action token to model
            action_t = torch.tensor([[action]], device=device)
            out = self.model(action_t, past_key_values=cache, use_cache=True)
            next_logits = out.logits[:, -1:, :]

            # ── EXECUTE ACTION ──

            if action == self.continue_id:
                segments.append(Segment(
                    chunk_ids=chunk_ids, action=action,
                    kv_start_pos=chunk_start_pos, kv_end_pos=boundary_pos,
                ))
                total_log_prob += chunk_log_prob + action_lp

            elif action == self.backoff_id:
                backoff_count += 1

                # Sample depth token (model sees the wrong tokens)
                depth, depth_lp = self._sample_masked(
                    next_logits, self.depth_ids, temperature, full_sequence
                )
                total_generated += 1
                full_sequence.append(depth)

                # Feed depth token to model — still has wrong tokens in cache
                depth_t = torch.tensor([[depth]], device=device)
                out = self.model(depth_t, past_key_values=cache, use_cache=True)
                next_logits = out.logits[:, -1:, :]

                # ── GENERATE DIRECTIVE (before truncation) ──
                # The directive is conditioned on the wrong tokens' hidden
                # states, so the model can "see" what went wrong and write
                # a specific correction (e.g., "26.53 is wrong, should be 27").
                directive_ids: list[int] = []
                directive_lp = 0.0
                force_continued = False

                for _ in range(self.config.k_dir):
                    token, lp = self._sample(next_logits, temperature, full_sequence)
                    directive_ids.append(token)
                    directive_lp += lp
                    total_generated += 1
                    full_sequence.append(token)

                    token_t = torch.tensor([[token]], device=device)
                    out = self.model(token_t, past_key_values=cache, use_cache=True)
                    next_logits = out.logits[:, -1:, :]

                    if token == self.continue_id:
                        break

                # Force-append <continue> if directive didn't end with one
                if not directive_ids or directive_ids[-1] != self.continue_id:
                    force_continued = True
                    directive_ids.append(self.continue_id)
                    full_sequence.append(self.continue_id)
                    total_generated += 1

                # Compute rewind depth (1, 2, or 3)
                depth_val = self.depth_ids.index(depth) + 1
                depth_val = min(depth_val, len(boundaries) - 1)

                # Pop boundaries to find target
                for _ in range(depth_val):
                    popped = boundaries.pop()
                    snapshots.pop(popped, None)
                target_pos = boundaries[-1]

                # Record segment
                seg = Segment(
                    chunk_ids=chunk_ids, action=action, depth=depth,
                    directive_ids=directive_ids,
                    force_continued=force_continued,
                    kv_start_pos=chunk_start_pos, kv_end_pos=boundary_pos,
                    rewind_pos=target_pos,
                )
                total_log_prob += chunk_log_prob + action_lp + depth_lp + directive_lp

                # ── NOW TRUNCATE: restore snapshot to rewind point ──
                _restore_snapshot(cache, snapshots[target_pos])
                full_sequence = full_sequence[:target_pos]

                # ── RE-INJECT directive into the clean cache ──
                # Feed directive tokens into the rewound cache so the
                # continuation is conditioned on the correction.
                directive_tensor = torch.tensor(
                    [directive_ids], device=device
                )
                out = self.model(
                    directive_tensor, past_key_values=cache, use_cache=True
                )
                next_logits = out.logits[:, -1:, :]
                full_sequence.extend(directive_ids)

                segments.append(seg)
                boundary_tracker.reset()

            elif action == self.terminate_id:
                # Record final segment
                segments.append(Segment(
                    chunk_ids=chunk_ids, action=action,
                    kv_start_pos=chunk_start_pos, kv_end_pos=boundary_pos,
                ))
                total_log_prob += chunk_log_prob + action_lp

                # Generate answer freely (greedy, no boundary tracking)
                answer_ids = self._generate_answer(cache, next_logits, device)
                answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
                full_sequence.extend(answer_ids)

                return TrajectoryRecord(
                    prompt_ids=prompt_list,
                    segments=segments,
                    answer_ids=answer_ids,
                    final_kv_length=cache.get_seq_length(),
                    answer_text=answer_text,
                    answer_number=extract_predicted_number(answer_text),
                    backoff_count=backoff_count,
                    total_generated_tokens=total_generated + len(answer_ids),
                    stored_log_prob=total_log_prob,
                    terminated=True,
                )

        # T_max reached without termination
        gen_text = self.tokenizer.decode(
            full_sequence[prompt_len:], skip_special_tokens=True
        )
        return TrajectoryRecord(
            prompt_ids=prompt_list,
            segments=segments,
            answer_ids=[],
            final_kv_length=cache.get_seq_length(),
            answer_text=gen_text,
            answer_number=extract_predicted_number(gen_text),
            backoff_count=backoff_count,
            total_generated_tokens=total_generated,
            stored_log_prob=total_log_prob,
            terminated=False,
        )

    # ── Helpers ──

    @staticmethod
    def _prune_snapshots(
        snapshots: dict[int, _CacheSnapshot],
        boundaries: list[int],
    ) -> None:
        """Remove snapshots for positions no longer in the boundaries stack."""
        live = set(boundaries)
        dead = [pos for pos in snapshots if pos not in live]
        for pos in dead:
            del snapshots[pos]

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, generated_ids: list[int]
    ) -> torch.Tensor:
        """Apply repetition penalty to logits for previously generated tokens."""
        penalty = self.config.repetition_penalty
        if penalty == 1.0 or not generated_ids:
            return logits
        unique_ids = list(set(generated_ids))
        token_logits = logits[unique_ids]
        # Divide positive logits, multiply negative logits (standard HF convention)
        logits[unique_ids] = torch.where(
            token_logits > 0, token_logits / penalty, token_logits * penalty
        )
        return logits

    def _sample(
        self, logits: torch.Tensor, temperature: float,
        generated_ids: list[int] | None = None,
    ) -> tuple[int, float]:
        """Sample from full vocab. Returns (token_id, log_prob_at_T1).

        Log-prob is always at temperature=1 (raw policy) for GRPO,
        regardless of the sampling temperature.
        """
        logits = logits[0, -1, :]  # [vocab]
        # Log-prob at T=1 for storage (before repetition penalty)
        log_probs_t1 = F.log_softmax(logits, dim=-1)

        # Apply repetition penalty for sampling only
        if generated_ids is not None:
            logits = self._apply_repetition_penalty(logits.clone(), generated_ids)

        if temperature <= 0:
            token = logits.argmax().item()
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).item()

        return token, log_probs_t1[token].item()

    def _sample_masked(
        self, logits: torch.Tensor, allowed_ids: list[int], temperature: float,
        generated_ids: list[int] | None = None,
    ) -> tuple[int, float]:
        """Sample from logits masked to allowed_ids only.

        Log-prob is computed under the masked distribution at T=1.
        """
        logits = logits[0, -1, :]  # [vocab]
        # Mask: -inf for everything except allowed
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed_ids] = 0.0
        masked_logits = logits + mask

        # Log-prob at T=1 under masked distribution
        log_probs_t1 = F.log_softmax(masked_logits, dim=-1)

        # Apply repetition penalty for sampling only
        if generated_ids is not None:
            masked_logits = self._apply_repetition_penalty(masked_logits.clone(), generated_ids)

        if temperature <= 0:
            token = masked_logits.argmax().item()
        else:
            probs = F.softmax(masked_logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).item()

        return token, log_probs_t1[token].item()

    def _generate_answer(
        self,
        cache,
        next_logits: torch.Tensor,
        device: torch.device,
        max_answer_tokens: int = 200,
    ) -> list[int]:
        """Generate answer tokens after </think>. Greedy, no actions."""
        answer_ids: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_answer_tokens):
            logits = next_logits[0, -1, :]
            token = logits.argmax().item()
            answer_ids.append(token)

            if token == eos_id:
                break

            token_t = torch.tensor([[token]], device=device)
            out = self.model(token_t, past_key_values=cache, use_cache=True)
            next_logits = out.logits[:, -1:, :]

        return answer_ids
