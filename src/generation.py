from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.boundary import BoundaryTracker
from src.config import BackoffConfig
from src.data.gsm8k import extract_predicted_number
from src.tokens import ACTION_TOKENS, BACKOFF_TOKENS, TERMINATE_TOKEN


@dataclass
class Segment:
    """One chunk of generation between boundary decisions."""

    chunk_ids: list[int]  # tokens generated in the chunk
    action: int | None  # backoff/terminate token id, or None for auto-boundary
    directive_ids: list[int] = field(default_factory=list)  # directive tokens (backoff only)
    kv_start_pos: int = 0  # cache position at start of chunk
    kv_end_pos: int = 0  # cache position at end of chunk
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
        (including chunk + <backoff_N>) forms one pass, and the post-crop
        state (surviving prefix + directive + new chunks) forms the next.
        """
        sequences: list[list[int]] = []
        # "active" = tokens currently surviving in the KV cache
        active = list(self.prompt_ids)
        pass_tokens: list[int] = []  # tokens added in the current pass

        for seg in self.segments:
            pass_tokens.extend(seg.chunk_ids)
            if seg.action is not None:
                pass_tokens.append(seg.action)

            if seg.directive_ids:
                # Backoff: directive is generated in the pre-truncation
                # context (model sees the wrong tokens when writing the
                # directive). All belong in this pass.
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


# ── Cache utilities for standard DynamicCache (Qwen3) ──


def _truncate_cache(cache, target_pos: int) -> None:
    """Truncate a standard DynamicCache to target_pos by slicing KV tensors."""
    for layer_idx in range(len(cache.key_cache)):
        cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :target_pos, :]
        cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :target_pos, :]


def _clone_cache(cache) -> object:
    """Deep-clone a DynamicCache so each rollout gets an independent copy."""
    from transformers import DynamicCache

    clone = DynamicCache()
    for layer_idx in range(len(cache.key_cache)):
        clone.key_cache.append(cache.key_cache[layer_idx].clone())
        clone.value_cache.append(cache.value_cache[layer_idx].clone())
    return clone


# ── Generator ──


class BackoffGenerator:
    """Backoff-aware generation loop for Qwen3 models.

    Boundaries are detected heuristically (sentence endings, logical
    connectives, etc.) using BoundaryTracker. At each boundary, the
    cache position is recorded. On backoff, the KV cache is simply
    sliced to a prior boundary position.
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

        # Pre-compute ID lists
        self.action_ids = [token_ids[t] for t in ACTION_TOKENS]
        self.backoff_ids = [token_ids[t] for t in BACKOFF_TOKENS]
        self.backoff_id_set = set(self.backoff_ids)
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
        """Generate a trajectory with heuristic boundary detection.

        The model generates freely. Boundaries are detected by
        BoundaryTracker (sentence endings, logical connectives, etc.)
        and cache positions are silently recorded. When the model emits
        a special token, we react:

        - <backoff_N>: collect directive tokens, then rewind the KV
          cache to a prior boundary and re-inject the directive.
        - </think>: generate the answer and return.

        Args:
            prompt_ids: [1, seq_len] tensor of tokenized prompt.
            temperature: sampling temperature. None uses config default.
            prefill: (cache, logits) from prefill(). If provided, the cache
                is deep-cloned so each rollout is independent.
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
        full_sequence = list(prompt_list)
        boundaries = [prompt_len]  # cache positions at detected boundaries
        boundary_tracker = BoundaryTracker(self.config.k_min, self.config.k_max)

        backoff_count = 0
        segments: list[Segment] = []
        total_log_prob = 0.0
        total_generated = 0

        # Current chunk accumulator (tokens between boundary decisions)
        chunk_ids: list[int] = []
        chunk_log_prob = 0.0
        chunk_start_pos = prompt_len

        while total_generated < self.config.t_max:
            token, lp = self._sample(next_logits, temperature, full_sequence)
            total_generated += 1

            if token in self.backoff_id_set:
                # ── BACKOFF: record segment, collect directive, rewind ──
                backoff_count += 1
                backoff_lp = lp
                backoff_action = token
                full_sequence.append(token)
                token_t = torch.tensor([[token]], device=device)
                out = self.model(token_t, past_key_values=cache, use_cache=True)
                next_logits = out.logits[:, -1:, :]

                # Collect directive tokens until newline or k_dir limit
                directive_ids: list[int] = []
                directive_lp = 0.0

                for _ in range(self.config.k_dir):
                    d_token, d_lp = self._sample(
                        next_logits, temperature, full_sequence
                    )
                    total_generated += 1
                    full_sequence.append(d_token)

                    # Stop collecting directive at newline
                    d_text = self.tokenizer.decode([d_token])
                    if "\n" in d_text and directive_ids:
                        directive_lp += d_lp
                        directive_ids.append(d_token)
                        token_t = torch.tensor([[d_token]], device=device)
                        out = self.model(token_t, past_key_values=cache, use_cache=True)
                        next_logits = out.logits[:, -1:, :]
                        break

                    directive_ids.append(d_token)
                    directive_lp += d_lp

                    token_t = torch.tensor([[d_token]], device=device)
                    out = self.model(token_t, past_key_values=cache, use_cache=True)
                    next_logits = out.logits[:, -1:, :]

                # Depth is encoded in the backoff token itself
                depth_val = self.backoff_ids.index(backoff_action) + 1
                depth_val = min(depth_val, len(boundaries) - 1)

                for _ in range(depth_val):
                    boundaries.pop()
                target_pos = boundaries[-1]

                seg = Segment(
                    chunk_ids=chunk_ids, action=backoff_action,
                    directive_ids=directive_ids,
                    kv_start_pos=chunk_start_pos,
                    kv_end_pos=chunk_start_pos + len(chunk_ids),
                    rewind_pos=target_pos,
                )
                total_log_prob += (
                    chunk_log_prob + backoff_lp + directive_lp
                )

                # Rewind cache to target boundary
                _truncate_cache(cache, target_pos)
                full_sequence = full_sequence[:target_pos]

                # Re-inject directive into the clean cache
                directive_tensor = torch.tensor([directive_ids], device=device)
                out = self.model(
                    directive_tensor, past_key_values=cache, use_cache=True
                )
                next_logits = out.logits[:, -1:, :]
                full_sequence.extend(directive_ids)

                segments.append(seg)

                # Reset chunk and boundary tracker
                chunk_ids = []
                chunk_log_prob = 0.0
                chunk_start_pos = len(full_sequence)
                boundary_tracker.reset()

            elif token == self.terminate_id:
                # ── TERMINATE: generate answer ──
                full_sequence.append(token)
                token_t = torch.tensor([[token]], device=device)
                out = self.model(token_t, past_key_values=cache, use_cache=True)
                next_logits = out.logits[:, -1:, :]

                segments.append(Segment(
                    chunk_ids=chunk_ids, action=self.terminate_id,
                    kv_start_pos=chunk_start_pos,
                    kv_end_pos=len(full_sequence),
                ))
                total_log_prob += chunk_log_prob + lp

                answer_ids = self._generate_answer(cache, next_logits, device)
                answer_text = self.tokenizer.decode(
                    answer_ids, skip_special_tokens=True
                )
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

            else:
                # ── REGULAR TOKEN: accumulate into chunk ──
                chunk_ids.append(token)
                chunk_log_prob += lp
                full_sequence.append(token)

                token_t = torch.tensor([[token]], device=device)
                out = self.model(token_t, past_key_values=cache, use_cache=True)
                next_logits = out.logits[:, -1:, :]

                # Check for heuristic boundary
                token_text = self.tokenizer.decode([token])
                if boundary_tracker.step(token_text):
                    boundary_pos = len(full_sequence)
                    boundaries.append(boundary_pos)
                    # Prune old boundaries beyond rewind depth
                    if len(boundaries) > self.config.d_max + 1:
                        boundaries = boundaries[-(self.config.d_max + 1):]

                    # Record segment (auto-boundary, no action token)
                    segments.append(Segment(
                        chunk_ids=chunk_ids, action=None,
                        kv_start_pos=chunk_start_pos,
                        kv_end_pos=boundary_pos,
                    ))
                    total_log_prob += chunk_log_prob

                    # Reset chunk
                    chunk_ids = []
                    chunk_log_prob = 0.0
                    chunk_start_pos = boundary_pos

        # T_max reached without termination
        if chunk_ids:
            segments.append(Segment(
                chunk_ids=chunk_ids, action=-1,
                kv_start_pos=chunk_start_pos,
                kv_end_pos=len(full_sequence),
            ))
            total_log_prob += chunk_log_prob

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
