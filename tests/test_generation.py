import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.tokens import setup_tokenizer_and_model, TERMINATE_TOKEN
from src.generation import (
    BackoffGenerator, Segment, TrajectoryRecord,
    _truncate_cache,
)

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"


@pytest.fixture(scope="module")
def setup():
    """Load model/tokenizer once for all tests in this module."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    token_ids = setup_tokenizer_and_model(tokenizer, model)
    return model, tokenizer, token_ids


def _make_prompt(tokenizer, question: str) -> torch.Tensor:
    messages = [{"role": "user", "content": f"Solve the following math problem. Give the final answer after ####.\n\n{question}"}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return tokenizer.encode(text, return_tensors="pt").to("cuda")


class ForcedActionGenerator(BackoffGenerator):
    """Generator that forces specific action choices for testing.

    Overrides _sample: when the model naturally emits an action token
    (backoff or terminate), replaces it with the next forced action.
    """

    def __init__(self, *args, forced_actions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced = forced_actions or []
        self._idx = 0

    def _sample(self, logits, temperature, generated_ids=None):
        token_id, lp = super()._sample(logits, temperature, generated_ids)
        # If the naturally sampled token is an action token, override it
        if token_id in self.action_ids:
            if self._idx < len(self._forced):
                forced = self._forced[self._idx]
                self._idx += 1
                logits_flat = logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
                return forced, log_probs[forced].item()
        return token_id, lp


# --- Cache mechanics ---

def test_truncate_cache_reduces_length(setup):
    model, tokenizer, token_ids = setup

    prompt_ids = _make_prompt(tokenizer, "What is 2+3?")
    prompt_len = prompt_ids.shape[1]

    # Prefill
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1:, :]

    # Generate 20 tokens
    for _ in range(20):
        token = logits[0, -1].argmax().item()
        out = model(torch.tensor([[token]], device="cuda"), past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1:, :]

    assert cache.get_seq_length() == prompt_len + 20

    # Truncate cache back to prompt_len
    _truncate_cache(cache, prompt_len)
    assert cache.get_seq_length() == prompt_len


def test_generation_continues_after_truncate(setup):
    model, tokenizer, token_ids = setup

    prompt_ids = _make_prompt(tokenizer, "What is 1+1?")

    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1:, :]
    initial_len = cache.get_seq_length()

    # Generate some tokens, then truncate
    for _ in range(10):
        token = logits[0, -1].argmax().item()
        out = model(torch.tensor([[token]], device="cuda"), past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1:, :]

    _truncate_cache(cache, initial_len)
    assert cache.get_seq_length() == initial_len

    # Continue generating 5 tokens from truncated state
    # Re-inject a token to get new logits after truncation
    token = torch.tensor([[logits[0, -1].argmax().item()]], device="cuda")
    out = model(token, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1:, :]

    for _ in range(4):
        token = logits[0, -1].argmax().item()
        out = model(torch.tensor([[token]], device="cuda"), past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1:, :]

    assert cache.get_seq_length() == initial_len + 5


# --- Basic generation ---

def test_generate_produces_trajectory(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=30, temperature=0.7)
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

    prompt_ids = _make_prompt(tokenizer, "What is 5 * 6?")
    traj = gen.generate(prompt_ids)

    assert isinstance(traj, TrajectoryRecord)
    assert len(traj.segments) >= 1
    assert traj.total_generated_tokens > 0
    assert traj.stored_log_prob < 0  # log-probs are negative
    assert traj.final_kv_length > 0


def test_greedy_generation_is_deterministic(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=30, temperature=0.0)
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

    prompt_ids = _make_prompt(tokenizer, "What is 3+4?")
    traj1 = gen.generate(prompt_ids)
    traj2 = gen.generate(prompt_ids)

    # Greedy should produce identical outputs
    assert traj1.segments[0].chunk_ids == traj2.segments[0].chunk_ids
    assert traj1.stored_log_prob == pytest.approx(traj2.stored_log_prob, abs=1e-3)


# --- Forced backoff ---

def test_forced_backoff_creates_backoff_segment(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=512, k_min=10, k_max=30, temperature=0.7, k_dir=10)

    gen = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<backoff_1>"],
            token_ids[TERMINATE_TOKEN],
        ],
    )

    prompt_ids = _make_prompt(tokenizer, "What is 15 + 27?")
    traj = gen.generate(prompt_ids)

    assert traj.backoff_count == 1
    assert traj.terminated

    backoff_segs = [s for s in traj.segments if s.rewind_pos is not None]
    assert len(backoff_segs) == 1
    assert len(backoff_segs[0].directive_ids) > 0
    assert backoff_segs[0].rewind_pos is not None


def test_build_forward_pass_sequences(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=512, k_min=10, k_max=30, temperature=0.7, k_dir=10)

    gen = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<backoff_1>"],
            token_ids[TERMINATE_TOKEN],
        ],
    )

    prompt_ids = _make_prompt(tokenizer, "What is 15 + 27?")
    traj = gen.generate(prompt_ids)

    seqs = traj.build_forward_pass_sequences()

    # 1 backoff → 2 forward passes
    assert len(seqs) == 2, f"Expected 2 passes, got {len(seqs)}"

    # Both start with the prompt
    for s in seqs:
        assert s[:len(traj.prompt_ids)] == traj.prompt_ids

    # Pass 1 should end with the last directive token
    backoff_seg = [s for s in traj.segments if s.rewind_pos is not None][0]
    assert seqs[0][-1] == backoff_seg.directive_ids[-1]
    # <backoff_N> should be in Pass 1
    assert backoff_seg.action in seqs[0]

    # Pass 2 should end with </think>
    assert seqs[1][-1] == token_ids[TERMINATE_TOKEN]


def test_forced_double_backoff(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=512, k_min=10, k_max=30, temperature=0.7, k_dir=10, b_max=2)

    gen = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<backoff_1>"],
            token_ids["<backoff_1>"],
            token_ids[TERMINATE_TOKEN],
        ],
    )

    prompt_ids = _make_prompt(tokenizer, "What is 100 - 37?")
    traj = gen.generate(prompt_ids)

    assert traj.backoff_count == 2
    assert traj.terminated

    backoff_segs = [s for s in traj.segments if s.rewind_pos is not None]
    assert len(backoff_segs) == 2


# --- Log-prob properties ---

def test_log_prob_is_finite(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=30, temperature=0.7)
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

    prompt_ids = _make_prompt(tokenizer, "What is 10 + 20?")
    traj = gen.generate(prompt_ids)

    assert not torch.isnan(torch.tensor(traj.stored_log_prob))
    assert not torch.isinf(torch.tensor(traj.stored_log_prob))
    assert traj.stored_log_prob < 0


# --- Auto-boundary segments ---

def test_auto_boundary_segments_have_none_action(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=30, temperature=0.7)
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

    prompt_ids = _make_prompt(tokenizer, "What is 5 * 6?")
    traj = gen.generate(prompt_ids)

    # Should have at least some auto-boundary segments (action=None)
    auto_segs = [s for s in traj.segments if s.action is None]
    if len(traj.segments) > 1:
        assert len(auto_segs) > 0, "Expected auto-boundary segments with action=None"
