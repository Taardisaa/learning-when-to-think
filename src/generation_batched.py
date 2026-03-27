"""Batched rollout generation for GRPO training.

Runs G rollouts simultaneously in one batch, sharing the model forward
pass. ~5-10x faster than sequential generation for G=16.

Rollouts that back off are removed from the batch and finished
individually (backoff is rare, so this rarely triggers).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import BackoffConfig
from src.data.gsm8k import extract_predicted_number
from src.generation import (
    Segment,
    TrajectoryRecord,
    BackoffGenerator,
    _save_snapshot,
    _restore_snapshot,
    _CacheSnapshot,
)
from src.tokens import ACTION_TOKENS, BACKOFF_TOKENS, TERMINATE_TOKEN


def _expand_cache_batch(cache, G: int):
    """Expand a batch=1 cache to batch=G by repeating along dim 0."""
    for i, ltype in enumerate(cache.layer_types):
        if ltype == "full_attention":
            cache.key_cache[i] = cache.key_cache[i].expand(G, -1, -1, -1).contiguous()
            cache.value_cache[i] = cache.value_cache[i].expand(G, -1, -1, -1).contiguous()
        else:
            cache.conv_states[i] = cache.conv_states[i].expand(G, -1, -1).contiguous()
            cache.recurrent_states[i] = cache.recurrent_states[i].expand(G, -1, -1, -1).contiguous()


def _extract_single_cache(cache, idx: int):
    """Extract a single-batch cache for rollout idx (for backoff fallback)."""
    from src.generation import _clone_cache
    clone = _clone_cache(cache)
    for i, ltype in enumerate(clone.layer_types):
        if ltype == "full_attention":
            clone.key_cache[i] = clone.key_cache[i][idx:idx+1].clone()
            clone.value_cache[i] = clone.value_cache[i][idx:idx+1].clone()
        else:
            clone.conv_states[i] = clone.conv_states[i][idx:idx+1].clone()
            clone.recurrent_states[i] = clone.recurrent_states[i][idx:idx+1].clone()
    return clone


class BatchedBackoffGenerator:
    """Batched generation: G rollouts per forward pass.

    Free token-by-token generation — the model decides when to emit
    <continue>, <backoff_N>, or </think>. Special tokens are detected
    in the stream and handled reactively.

    Rollouts that terminate get their answers generated individually.
    Rollouts that back off fall back to the sequential BackoffGenerator.
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

        self.action_ids = [token_ids[t] for t in ACTION_TOKENS]
        self.backoff_ids = [token_ids[t] for t in BACKOFF_TOKENS]
        self.backoff_id_set = set(self.backoff_ids)
        self.continue_id = token_ids["<continue>"]
        self.terminate_id = token_ids[TERMINATE_TOKEN]

        # Sequential fallback for backoff/answer
        self._seq_gen = BackoffGenerator(model, tokenizer, token_ids, config)

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids: torch.Tensor,
        G: int,
        temperature: float | None = None,
    ) -> list[TrajectoryRecord]:
        """Generate G rollouts in a batched fashion.

        Args:
            prompt_ids: [1, seq_len] tokenized prompt.
            G: number of rollouts.
            temperature: sampling temperature.

        Returns list of G TrajectoryRecords.
        """
        if temperature is None:
            temperature = self.config.temperature
        device = prompt_ids.device
        prompt_list = prompt_ids[0].tolist()
        prompt_len = len(prompt_list)

        # ── Prefill (batch=1, then expand) ──
        out = self.model(prompt_ids, use_cache=True)
        cache = out.past_key_values
        _expand_cache_batch(cache, G)
        # Logits: [1, 1, vocab] → [G, 1, vocab]
        next_logits = out.logits[:, -1:, :].expand(G, -1, -1).contiguous()

        # ── Per-rollout state ──
        states = [_RolloutState(prompt_list) for _ in range(G)]
        active = [True] * G  # still generating
        results: list[TrajectoryRecord | None] = [None] * G

        total_steps = 0

        while any(active) and total_steps < self.config.t_max:
            total_steps += 1

            # ── Sample tokens for all rollouts ──
            tokens = []
            for i in range(G):
                if not active[i]:
                    tokens.append(self.tokenizer.pad_token_id or 0)
                    continue

                s = states[i]
                logits_i = next_logits[i, -1, :]  # [vocab]

                token, lp = self._sample_full(logits_i, temperature)
                s.total_generated += 1
                s.full_sequence.append(token)

                if token == self.continue_id:
                    # Boundary — finish segment, start new chunk
                    s.finish_segment(token, lp)

                elif token in self.backoff_id_set:
                    # Pull out of batch, handle individually
                    s.action_lp = lp
                    s.current_action = token
                    active[i] = False
                    results[i] = self._finish_with_backoff(
                        cache, i, s, prompt_ids, prompt_list, temperature
                    )
                    tokens.append(self.tokenizer.pad_token_id or 0)
                    continue

                elif token == self.terminate_id:
                    s.finish_segment(token, lp)
                    active[i] = False
                    # Generate answer individually
                    results[i] = self._finish_terminated(
                        cache, i, s, next_logits[i:i+1, :, :], prompt_list
                    )
                    tokens.append(self.tokenizer.pad_token_id or 0)
                    continue

                else:
                    # Regular token — accumulate into chunk
                    s.chunk_ids.append(token)
                    s.chunk_log_prob += lp

                tokens.append(token)

            # ── Batched forward pass ──
            if not any(active):
                break
            token_batch = torch.tensor(tokens, device=device).unsqueeze(1)  # [G, 1]
            out = self.model(token_batch, past_key_values=cache, use_cache=True)
            next_logits = out.logits[:, -1:, :]  # [G, 1, vocab]

        # ── Handle any still-active rollouts (hit t_max) ──
        for i in range(G):
            if results[i] is None:
                s = states[i]
                gen_text = self.tokenizer.decode(
                    s.full_sequence[prompt_len:], skip_special_tokens=True
                )
                results[i] = TrajectoryRecord(
                    prompt_ids=prompt_list,
                    segments=s.segments,
                    answer_ids=[],
                    final_kv_length=len(s.full_sequence),
                    answer_text=gen_text,
                    answer_number=extract_predicted_number(gen_text),
                    backoff_count=s.backoff_count,
                    total_generated_tokens=s.total_generated,
                    stored_log_prob=s.total_log_prob,
                    terminated=False,
                )

        return results

    # ── Helpers ──

    def _sample_full(self, logits: torch.Tensor, temperature: float):
        """Sample from full vocab, return (token, log_prob_at_T1)."""
        log_probs = F.log_softmax(logits, dim=-1)
        if temperature <= 0:
            token = logits.argmax().item()
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).item()
        return token, log_probs[token].item()

    def _finish_terminated(self, cache, idx, state, logits_i, prompt_list):
        """Generate answer for a terminated rollout (sequential)."""
        single_cache = _extract_single_cache(cache, idx)
        answer_ids = self._seq_gen._generate_answer(
            single_cache, logits_i, logits_i.device
        )
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        state.full_sequence.extend(answer_ids)

        return TrajectoryRecord(
            prompt_ids=prompt_list,
            segments=state.segments,
            answer_ids=answer_ids,
            final_kv_length=single_cache.get_seq_length() + len(answer_ids),
            answer_text=answer_text,
            answer_number=extract_predicted_number(answer_text),
            backoff_count=state.backoff_count,
            total_generated_tokens=state.total_generated + len(answer_ids),
            stored_log_prob=state.total_log_prob,
            terminated=True,
        )

    def _finish_with_backoff(self, cache, idx, state, prompt_ids, prompt_list, temperature):
        """Handle a rollout that emitted <backoff_N>.

        Extracts this rollout's cache, completes the backoff sequence
        (directive, cache rewind), then hands off to the sequential
        generator for the remainder of the trajectory.
        """
        from dataclasses import replace as dc_replace

        device = prompt_ids.device
        single_cache = _extract_single_cache(cache, idx)
        backoff_action = state.current_action

        # ── 1. Feed <backoff_N> token → get logits for directive ──
        action_t = torch.tensor([[backoff_action]], device=device)
        out = self.model(action_t, past_key_values=single_cache, use_cache=True)
        dir_logits = out.logits

        # ── 2. Generate directive (up to k_dir tokens) ──
        directive_ids: list[int] = []
        directive_lp = 0.0
        force_continued = False

        for _ in range(self.config.k_dir):
            token, lp = self._seq_gen._sample(dir_logits, temperature, state.full_sequence)
            directive_ids.append(token)
            directive_lp += lp
            state.total_generated += 1
            state.full_sequence.append(token)

            token_t = torch.tensor([[token]], device=device)
            out = self.model(token_t, past_key_values=single_cache, use_cache=True)
            dir_logits = out.logits

            if token == self.continue_id:
                break

        if not directive_ids or directive_ids[-1] != self.continue_id:
            force_continued = True
            directive_ids.append(self.continue_id)
            state.full_sequence.append(self.continue_id)
            state.total_generated += 1

        # ── 3. Compute rewind target ──
        boundaries = [state.prompt_len]
        for seg in state.segments:
            boundaries.append(seg.kv_end_pos)
        current_boundary = state.chunk_start_pos + len(state.chunk_ids)
        boundaries.append(current_boundary)

        # Depth is encoded in the backoff token
        depth_val = self.backoff_ids.index(backoff_action) + 1
        depth_val = min(depth_val, len(boundaries) - 1)

        for _ in range(depth_val):
            boundaries.pop()
        target_pos = boundaries[-1]

        # ── 4. Record backoff segment ──
        seg = Segment(
            chunk_ids=list(state.chunk_ids),
            action=backoff_action,
            directive_ids=directive_ids,
            force_continued=force_continued,
            kv_start_pos=state.chunk_start_pos,
            kv_end_pos=current_boundary,
            rewind_pos=target_pos,
        )
        state.segments.append(seg)
        state.total_log_prob += state.chunk_log_prob + state.action_lp + directive_lp
        state.backoff_count += 1

        # ── 5. Re-prefill to rewind point + inject directive ──
        rewound_seq = state.full_sequence[:target_pos] + directive_ids
        rewound_t = torch.tensor([rewound_seq], device=device)
        out = self.model(rewound_t, use_cache=True)
        rewound_cache = out.past_key_values
        rewound_logits = out.logits[:, -1:, :]

        # ── 6. Continue with sequential generator ──
        remaining_t_max = max(self.config.t_max - state.total_generated, 1)
        cont_config = dc_replace(self.config, t_max=remaining_t_max)
        cont_gen = BackoffGenerator(self.model, self.tokenizer, self.token_ids, cont_config)

        cont_prompt = torch.tensor([rewound_seq], device=device)
        cont_traj = cont_gen.generate(
            cont_prompt, temperature=temperature,
            prefill=(rewound_cache, rewound_logits),
        )

        # ── 7. Combine pre-backoff + backoff + continuation ──
        return TrajectoryRecord(
            prompt_ids=prompt_list,
            segments=state.segments + cont_traj.segments,
            answer_ids=cont_traj.answer_ids,
            final_kv_length=cont_traj.final_kv_length,
            answer_text=cont_traj.answer_text,
            answer_number=cont_traj.answer_number,
            backoff_count=state.backoff_count + cont_traj.backoff_count,
            total_generated_tokens=state.total_generated + cont_traj.total_generated_tokens,
            stored_log_prob=state.total_log_prob + cont_traj.stored_log_prob,
            terminated=cont_traj.terminated,
        )


class _RolloutState:
    """Mutable per-rollout state for batched generation."""

    def __init__(self, prompt_list: list[int]):
        self.full_sequence = list(prompt_list)
        self.prompt_len = len(prompt_list)
        self.segments: list[Segment] = []
        self.backoff_count = 0
        self.total_generated = 0
        self.total_log_prob = 0.0

        # Current chunk being built
        self.chunk_ids: list[int] = []
        self.chunk_log_prob = 0.0
        self.chunk_start_pos = len(prompt_list)

        # Temp storage for action decision
        self.action_lp = 0.0
        self.current_action = -1

    def finish_segment(self, action: int, action_lp: float):
        """Close the current chunk as a segment."""
        boundary_pos = self.chunk_start_pos + len(self.chunk_ids)
        self.segments.append(Segment(
            chunk_ids=list(self.chunk_ids),
            action=action,
            kv_start_pos=self.chunk_start_pos,
            kv_end_pos=boundary_pos,
        ))
        self.total_log_prob += self.chunk_log_prob + action_lp

        # Reset for next chunk
        self.chunk_ids = []
        self.chunk_log_prob = 0.0
        self.chunk_start_pos = len(self.full_sequence)
        self.action_lp = 0.0
