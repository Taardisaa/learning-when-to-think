"""Action token setup for the 3-action MDP pivot.

Adds <continue>, <refine>, <terminate> as new special tokens to the
Qwen2.5-Math-7B-Instruct vocabulary. Qwen2.5-Math has no <think>/</think>
tokens — action tokens appear inline in the natural CoT.

Usage:
    from src.pivot.tokens import setup_tokenizer_and_model, enable_new_token_grad

    token_ids = setup_tokenizer_and_model(tokenizer, model)
    # After wrapping with LoRA:
    hooks = enable_new_token_grad(model, token_ids)
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

ACTION_TOKENS = ["<continue>", "<refine>", "<terminate>"]


def setup_tokenizer_and_model(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> dict[str, int]:
    """Add 3 action tokens, resize embeddings, initialize with mean+noise.

    Returns dict mapping token string -> token id.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ACTION_TOKENS}
    )
    assert num_added == len(ACTION_TOKENS), (
        f"Expected to add {len(ACTION_TOKENS)} tokens, got {num_added}. "
        "Tokens may already exist in the vocabulary."
    )

    # Resize model embeddings
    old_num_embeddings = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new embeddings: mean of existing + small noise
    with torch.no_grad():
        input_emb = model.get_input_embeddings().weight
        mean_emb = input_emb[:old_num_embeddings].mean(dim=0)
        for i in range(old_num_embeddings, len(tokenizer)):
            noise = torch.randn_like(mean_emb) * 0.01
            input_emb[i] = mean_emb + noise

        # Same for output (lm_head) embeddings if separate
        output_emb = model.get_output_embeddings()
        if output_emb is not None and output_emb.weight is not input_emb:
            out_w = output_emb.weight
            mean_out = out_w[:old_num_embeddings].mean(dim=0)
            for i in range(old_num_embeddings, len(tokenizer)):
                noise = torch.randn_like(mean_out) * 0.01
                out_w[i] = mean_out + noise

    # Build token_ids dict
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in ACTION_TOKENS}

    # Verify new tokens got valid (non-UNK) IDs
    unk_id = tokenizer.unk_token_id
    for tok in ACTION_TOKENS:
        assert token_ids[tok] != unk_id, f"Token {tok} mapped to UNK"

    return token_ids


def get_action_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """Get token IDs for action tokens (assumes setup_tokenizer_and_model was called)."""
    return {tok: tokenizer.convert_tokens_to_ids(tok) for tok in ACTION_TOKENS}


def enable_new_token_grad(model: PreTrainedModel, token_ids: dict[str, int]) -> list:
    """Register gradient hooks so only new token embedding rows are trained.

    Call AFTER get_peft_model(). Returns hook handles (keep a reference).
    """
    new_ids = list(token_ids.values())
    handles = []

    for emb_module in [model.get_input_embeddings(), model.get_output_embeddings()]:
        if emb_module is None:
            continue
        w = emb_module.weight
        w.requires_grad_(True)

        def _mask_grad(grad, _ids=new_ids, _size=w.shape[0]):
            mask = torch.zeros(_size, 1, device=grad.device, dtype=grad.dtype)
            for idx in _ids:
                mask[idx] = 1.0
            return grad * mask

        handles.append(w.register_hook(_mask_grad))

    return handles
