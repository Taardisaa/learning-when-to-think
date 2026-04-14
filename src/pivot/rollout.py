"""Structured rollout for RL training with pluggable action-injection strategies.

This module provides a `rollout_func` compatible with trl 0.29.1's GRPOTrainer.
It generates K completions per prompt using HF generate + a custom logits
processor (from `action_strategies`), and returns the token ids plus per-token
log-probs expected by trl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from src.pivot.action_strategies import ActionStrategy
from src.pivot.tokens import ACTION_TOKENS

if TYPE_CHECKING:
    from trl import GRPOTrainer


def make_rollout_func(strategy: ActionStrategy) -> Any:
    """Construct a rollout_func closure with the given action strategy.

    The returned callable has the signature expected by trl's GRPOTrainer:
        f(prompts: list, trainer: GRPOTrainer) -> dict with keys
            prompt_ids, completion_ids, logprobs (and optional extras)
    """

    def rollout(prompts: list, trainer: "GRPOTrainer") -> dict:
        tokenizer = trainer.processing_class
        model = trainer.model_wrapped
        device = trainer.accelerator.device

        # Resolve action token ids once
        action_token_ids = [
            tokenizer.convert_tokens_to_ids(tok) for tok in ACTION_TOKENS
        ]
        logits_processor = strategy.logits_processor(action_token_ids)

        # Apply chat template if conversational; otherwise assume plain strings
        def to_prompt_string(p):
            if isinstance(p, list):  # list of message dicts
                return tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
            return p

        prompt_strings = [to_prompt_string(p) for p in prompts]

        # Tokenize each prompt separately (different lengths); batch-generate
        # with left-padding so completions align on the right.
        tokenizer.padding_side = "left"
        encoding = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        num_gen = trainer.num_generations
        gen_kwargs = dict(trainer.generation_kwargs)
        # Strip kwargs that aren't compatible with generate()
        for k in ("cache_implementation",):
            gen_kwargs.pop(k, None)
        gen_kwargs["num_return_sequences"] = num_gen
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = [logits_processor]
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True

        with torch.no_grad():
            out = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                **gen_kwargs,
            )

        sequences = out.sequences  # (batch*num_gen, prompt_len + comp_len)
        prompt_len = encoding.input_ids.shape[1]
        completion_ids_batched = sequences[:, prompt_len:]

        # Stack scores: tuple of (batch*num_gen, vocab_size), length comp_len
        scores = torch.stack(out.scores, dim=1)  # (B, comp_len, vocab)
        # log-softmax over full vocab (note: if logits_processor masked to -inf,
        # those positions stay -inf, and log_softmax distributes mass over the
        # unmasked tokens correctly).
        log_probs_all = F.log_softmax(scores.float(), dim=-1)

        # Gather log-prob of the sampled token at each step
        gathered = log_probs_all.gather(
            -1, completion_ids_batched.unsqueeze(-1)
        ).squeeze(-1)  # (B, comp_len)

        # Trim each row at first EOS (pad thereafter doesn't matter for trl,
        # but we return ragged lists)
        eos_id = tokenizer.eos_token_id
        prompt_ids_list = []
        completion_ids_list = []
        logprobs_list = []

        # Expand prompt ids to one per rollout
        # encoding.input_ids has shape (num_prompts, prompt_len) — we need num_prompts * num_gen rows.
        expanded_prompts = encoding.input_ids.repeat_interleave(num_gen, dim=0)

        for i in range(sequences.shape[0]):
            # Strip left-padding from the prompt
            raw_prompt = expanded_prompts[i].tolist()
            pad_id = tokenizer.pad_token_id or 0
            # Find first non-pad position
            j = 0
            while j < len(raw_prompt) and raw_prompt[j] == pad_id:
                j += 1
            prompt_ids_list.append(raw_prompt[j:])

            # Find completion length: up to first EOS (inclusive)
            comp = completion_ids_batched[i].tolist()
            eos_pos = len(comp)
            for k, t in enumerate(comp):
                if t == eos_id:
                    eos_pos = k + 1
                    break
            completion_ids_list.append(comp[:eos_pos])
            logprobs_list.append(gathered[i, :eos_pos].tolist())

        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "logprobs": logprobs_list,
        }

    return rollout
