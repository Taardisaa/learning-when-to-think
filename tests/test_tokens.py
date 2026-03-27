import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model

from src.tokens import (
    NEW_SPECIAL_TOKENS,
    TERMINATE_TOKEN,
    ACTION_TOKENS,
    BACKOFF_TOKENS,
    setup_tokenizer_and_model,
    enable_new_token_grad,
)

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"


def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cpu"
    )
    return tokenizer, model


def test_setup_returns_all_token_ids():
    tokenizer, model = _load()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    # Should have entries for all new tokens + </think>
    expected = set(NEW_SPECIAL_TOKENS + [TERMINATE_TOKEN])
    assert set(token_ids.keys()) == expected


def test_terminate_token_not_unk():
    tokenizer, model = _load()
    token_ids = setup_tokenizer_and_model(tokenizer, model)
    assert token_ids[TERMINATE_TOKEN] != tokenizer.unk_token_id


def test_new_tokens_get_unique_ids():
    tokenizer, model = _load()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    new_ids = [token_ids[t] for t in NEW_SPECIAL_TOKENS]
    # All unique
    assert len(set(new_ids)) == len(new_ids)
    # None equal to </think>
    assert token_ids[TERMINATE_TOKEN] not in new_ids


def test_embeddings_resized():
    tokenizer, model = _load()
    old_vocab = len(tokenizer)
    setup_tokenizer_and_model(tokenizer, model)
    new_vocab = len(tokenizer)

    assert new_vocab == old_vocab + len(NEW_SPECIAL_TOKENS)
    assert model.get_input_embeddings().weight.shape[0] == new_vocab


def test_new_embeddings_not_zero():
    tokenizer, model = _load()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    emb = model.get_input_embeddings().weight
    for tok in NEW_SPECIAL_TOKENS:
        tid = token_ids[tok]
        assert emb[tid].abs().sum() > 0, f"{tok} embedding is all zeros"


def test_action_and_backoff_groups():
    assert "<backoff_1>" in ACTION_TOKENS
    assert "<backoff_2>" in ACTION_TOKENS
    assert "<backoff_3>" in ACTION_TOKENS
    assert TERMINATE_TOKEN in ACTION_TOKENS
    assert "<continue>" not in ACTION_TOKENS
    assert len(BACKOFF_TOKENS) == 3


def test_roundtrip_encode_decode():
    tokenizer, model = _load()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    for tok in NEW_SPECIAL_TOKENS:
        tid = token_ids[tok]
        decoded = tokenizer.decode([tid])
        assert decoded == tok, f"Roundtrip failed: {tok} -> {tid} -> {decoded}"


def test_enable_new_token_grad_masks_pretrained():
    tokenizer, model = _load()
    token_ids = setup_tokenizer_and_model(tokenizer, model)

    lora = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    hooks = enable_new_token_grad(model, token_ids)

    # Forward + backward with a new token in input
    new_tok_id = token_ids["<backoff_1>"]
    input_ids = tokenizer("test", return_tensors="pt").input_ids
    input_ids = torch.cat([input_ids, torch.tensor([[new_tok_id]])], dim=1)
    out = model(input_ids=input_ids, labels=input_ids.clone())
    out.loss.backward()

    grad = model.get_input_embeddings().weight.grad
    nonzero_rows = set((grad.abs().sum(dim=1) > 0).nonzero().squeeze().tolist())
    new_token_ids = set(token_ids[t] for t in NEW_SPECIAL_TOKENS)

    # Only new token rows should have gradients
    assert nonzero_rows.issubset(new_token_ids), (
        f"Pretrained rows got grad: {nonzero_rows - new_token_ids}"
    )
    # At least the token we used should have grad
    assert new_tok_id in nonzero_rows

    for h in hooks:
        h.remove()
