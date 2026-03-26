import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

# The 5 new special tokens we add to the vocabulary.
# <terminate> is NOT added — it reuses the existing </think> (id 248069 in Qwen3.5).
NEW_SPECIAL_TOKENS = [
    "<continue>",
    "<backoff>",
    "<depth_1>",
    "<depth_2>",
    "<depth_3>",
]

TERMINATE_TOKEN = "</think>"  # already in Qwen3.5 vocab
TERMINATE_TOKEN_ID = 248069

# Convenience groupings for masked sampling
ACTION_TOKENS = ["<continue>", "<backoff>", TERMINATE_TOKEN]
DEPTH_TOKENS = ["<depth_1>", "<depth_2>", "<depth_3>"]


def setup_tokenizer_and_model(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> dict[str, int]:
    """Add special tokens, resize embeddings, initialize new embeddings.

    Returns a dict mapping token string -> token id for all action/depth tokens
    plus <terminate>.
    """
    # Add new tokens
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": NEW_SPECIAL_TOKENS}
    )
    assert num_added == len(NEW_SPECIAL_TOKENS), (
        f"Expected to add {len(NEW_SPECIAL_TOKENS)} tokens, got {num_added}. "
        "Tokens may already exist in the vocabulary."
    )

    # Resize model embeddings
    old_num_embeddings = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new embeddings: mean of existing + small noise.
    # Puts them in a typical region of embedding space so early forward
    # passes don't produce out-of-distribution activations.
    with torch.no_grad():
        input_emb = model.get_input_embeddings().weight
        mean_emb = input_emb[:old_num_embeddings].mean(dim=0)
        for i in range(old_num_embeddings, len(tokenizer)):
            noise = torch.randn_like(mean_emb) * 0.01
            input_emb[i] = mean_emb + noise

        # Do the same for the output (lm_head) embeddings if they're separate
        output_emb = model.get_output_embeddings()
        if output_emb is not None and output_emb.weight is not input_emb:
            out_w = output_emb.weight
            mean_out = out_w[:old_num_embeddings].mean(dim=0)
            for i in range(old_num_embeddings, len(tokenizer)):
                noise = torch.randn_like(mean_out) * 0.01
                out_w[i] = mean_out + noise

    # Build token_ids dict
    all_special = NEW_SPECIAL_TOKENS + [TERMINATE_TOKEN]
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in all_special}

    # Verify </think> is at the expected ID
    assert token_ids[TERMINATE_TOKEN] == TERMINATE_TOKEN_ID, (
        f"Expected </think> at id {TERMINATE_TOKEN_ID}, "
        f"got {token_ids[TERMINATE_TOKEN]}"
    )

    # Verify new tokens got valid (non-UNK) IDs
    unk_id = tokenizer.unk_token_id
    for tok in NEW_SPECIAL_TOKENS:
        assert token_ids[tok] != unk_id, f"Token {tok} mapped to UNK"

    return token_ids


def enable_new_token_grad(model: PreTrainedModel, token_ids: dict[str, int]) -> list:
    """Register gradient hooks so only new token embedding rows are trained.

    When using LoRA, the full embedding is frozen. This selectively enables
    gradients for just the new token rows without unfreezing the entire matrix
    or breaking weight tying.

    Call AFTER get_peft_model(). Returns hook handles (keep a reference to
    prevent garbage collection).

    Usage:
        model = get_peft_model(base_model, lora_config)
        hooks = enable_new_token_grad(model, token_ids)
    """
    new_ids = [token_ids[tok] for tok in NEW_SPECIAL_TOKENS]
    handles = []

    for emb_module in [model.get_input_embeddings(), model.get_output_embeddings()]:
        if emb_module is None:
            continue
        w = emb_module.weight
        # Enable grad on the full parameter (required for backward to flow)
        w.requires_grad_(True)

        # Mask: zero out gradients for all rows except new tokens
        def _mask_grad(grad, _ids=new_ids, _size=w.shape[0]):
            mask = torch.zeros(_size, 1, device=grad.device, dtype=grad.dtype)
            for idx in _ids:
                mask[idx] = 1.0
            return grad * mask

        handles.append(w.register_hook(_mask_grad))

    return handles
