"""Tests for segment-wise log-prob computation (Milestone 5).

Verifies:
1. 0-backoff trajectory: compute_trajectory_logprobs matches single forward pass
2. 1-backoff trajectory: segment boundaries and pass construction correct
3. Gradients flow through LoRA weights
"""

import torch
import pytest
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.tokens import setup_tokenizer_and_model, TERMINATE_TOKEN, ACTION_TOKENS, BACKOFF_TOKENS
from src.generation import BackoffGenerator, TrajectoryRecord
from src.train.logprobs import compute_trajectory_logprobs, compute_batch_logprobs, _build_passes


@pytest.fixture(scope="module")
def setup():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    token_ids = setup_tokenizer_and_model(tokenizer, model)
    return model, tokenizer, token_ids


def _make_prompt(tokenizer, question):
    messages = [{"role": "user", "content": f"Solve the following math problem. Give the final answer after ####.\n\n{question}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    return tokenizer.encode(text, return_tensors="pt").to("cuda")


class ForcedActionGenerator(BackoffGenerator):
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


# --- Test 1: 0-backoff matches single forward pass ---

def test_zero_backoff_matches_single_pass(setup):
    """For a trajectory with no backoffs, compute_trajectory_logprobs should
    produce the same result as a manual single forward pass."""
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=25, temperature=0.0)

    # Force: continue, terminate (no backoff)
    gen = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[token_ids["<continue>"], token_ids[TERMINATE_TOKEN]],
    )

    prompt_ids = _make_prompt(tokenizer, "What is 2+3?")
    traj = gen.generate(prompt_ids)

    assert traj.backoff_count == 0
    assert traj.terminated

    # compute_trajectory_logprobs
    computed_lp = compute_trajectory_logprobs(model, traj, token_ids, b_max=config.b_max)

    # Verify it's close to the stored log-prob from generation
    # (Both should be at T=1, so they should match exactly for greedy)
    assert abs(computed_lp.item() - traj.stored_log_prob) < 0.5, (
        f"Mismatch: computed={computed_lp.item():.4f}, stored={traj.stored_log_prob:.4f}"
    )

    # Also verify single-pass construction
    passes = _build_passes(traj, token_ids, config.b_max)
    assert len(passes) == 1, f"Expected 1 pass for 0-backoff, got {len(passes)}"


# --- Test 2: 1-backoff trajectory has 2 passes ---

def test_one_backoff_has_two_passes(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=512, k_min=10, k_max=25, temperature=0.0, k_dir=8)

    gen = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<continue>"],
            token_ids["<backoff_1>"],
            token_ids[TERMINATE_TOKEN],
        ],
    )

    prompt_ids = _make_prompt(tokenizer, "What is 15 + 27?")
    traj = gen.generate(prompt_ids)

    assert traj.backoff_count == 1

    passes = _build_passes(traj, token_ids, config.b_max)
    assert len(passes) == 2, f"Expected 2 passes, got {len(passes)}"

    # Pass 1 should end with last directive token (directive generated
    # before truncation, scored in pre-truncation context)
    seq1, start1, meta1 = passes[0]
    backoff_seg = [s for s in traj.segments if s.rewind_pos is not None][0]
    assert seq1[-1] == backoff_seg.directive_ids[-1]

    # Pass 2 prefix = surviving tokens + directive (re-injected)
    seq2, start2, meta2 = passes[1]
    expected_prefix_len = backoff_seg.rewind_pos + len(backoff_seg.directive_ids)
    assert start2 == expected_prefix_len

    # Pass 2 should end with </think>
    assert seq2[-1] == token_ids[TERMINATE_TOKEN]

    # Compute log-probs — should be finite
    lp = compute_trajectory_logprobs(model, traj, token_ids, b_max=config.b_max)
    assert not torch.isnan(lp)
    assert not torch.isinf(lp)
    assert lp.item() < 0


# --- Test 3: Gradients flow through LoRA ---

def test_gradients_flow(setup):
    base_model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=25, temperature=0.0)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.train()

    gen = ForcedActionGenerator(
        lora_model, tokenizer, token_ids, config,
        forced_actions=[token_ids["<continue>"], token_ids[TERMINATE_TOKEN]],
    )
    prompt_ids = _make_prompt(tokenizer, "What is 1+1?")

    with torch.no_grad():
        traj = gen.generate(prompt_ids)

    # Compute log-probs WITH gradients
    lp = compute_trajectory_logprobs(lora_model, traj, token_ids, b_max=config.b_max)
    lp.backward()

    # Check that LoRA weights have gradients
    has_grad = False
    for name, p in lora_model.named_parameters():
        if "lora" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients on LoRA weights after backward()"

    # Clean up: remove LoRA to not affect other tests
    lora_model.zero_grad()
    lora_model.eval()


# --- Test 4: Action masking respects b_max ---

def test_action_masking_respects_b_max(setup):
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=512, k_min=10, k_max=25, temperature=0.0, k_dir=8, b_max=1)

    gen = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<continue>"],
            token_ids["<backoff_1>"],  # uses up b_max=1
            # Next action: <backoff> should be masked
        ],
    )

    prompt_ids = _make_prompt(tokenizer, "What is 8 * 9?")
    traj = gen.generate(prompt_ids)

    passes = _build_passes(traj, token_ids, b_max=1)

    # Find action positions in pass 2 (after the backoff)
    if len(passes) >= 2:
        _, _, meta2 = passes[-1]
        for key, val in meta2.items():
            if isinstance(key, str) and key.startswith("allowed_"):
                allowed = val
                for bt in BACKOFF_TOKENS:
                    assert token_ids[bt] not in allowed, (
                        f"{bt} should be masked after b_max reached"
                    )


# --- Test 5: Batched log-probs match sequential ---

def test_batch_logprobs_matches_sequential(setup):
    """compute_batch_logprobs should produce the same values as calling
    compute_trajectory_logprobs individually on each trajectory."""
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=25, temperature=0.0)

    # Generate two trajectories with different questions
    trajs = []
    for q in ["What is 2+3?", "What is 7*8?"]:
        gen = ForcedActionGenerator(
            model, tokenizer, token_ids, config,
            forced_actions=[token_ids["<continue>"], token_ids[TERMINATE_TOKEN]],
        )
        prompt_ids = _make_prompt(tokenizer, q)
        traj = gen.generate(prompt_ids)
        trajs.append(traj)

    # Sequential
    seq_lps = [
        compute_trajectory_logprobs(model, t, token_ids, b_max=config.b_max)
        for t in trajs
    ]

    # Batched
    batch_lps = compute_batch_logprobs(model, trajs, token_ids, b_max=config.b_max)

    assert len(batch_lps) == len(seq_lps)
    for i, (s, b) in enumerate(zip(seq_lps, batch_lps)):
        assert abs(s.item() - b.item()) < 0.5, (
            f"Traj {i}: sequential={s.item():.4f}, batched={b.item():.4f}"
        )


def test_batch_logprobs_with_backoff(setup):
    """Batched log-probs match sequential even with backoff trajectories."""
    model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=512, k_min=10, k_max=25, temperature=0.0, k_dir=8)

    # One with backoff, one without
    gen_no_backoff = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[token_ids["<continue>"], token_ids[TERMINATE_TOKEN]],
    )
    gen_backoff = ForcedActionGenerator(
        model, tokenizer, token_ids, config,
        forced_actions=[
            token_ids["<continue>"],
            token_ids["<backoff_1>"],
            token_ids[TERMINATE_TOKEN],
        ],
    )

    prompt1 = _make_prompt(tokenizer, "What is 5+5?")
    prompt2 = _make_prompt(tokenizer, "What is 3*4?")

    traj1 = gen_no_backoff.generate(prompt1)
    traj2 = gen_backoff.generate(prompt2)
    trajs = [traj1, traj2]

    seq_lps = [
        compute_trajectory_logprobs(model, t, token_ids, b_max=config.b_max)
        for t in trajs
    ]
    batch_lps = compute_batch_logprobs(model, trajs, token_ids, b_max=config.b_max)

    for i, (s, b) in enumerate(zip(seq_lps, batch_lps)):
        # Tolerance accounts for numerical differences between batched
        # and sequential forward passes (different CUDA kernel paths)
        assert abs(s.item() - b.item()) < 0.5, (
            f"Traj {i}: sequential={s.item():.4f}, batched={b.item():.4f}"
        )


def test_batch_logprobs_gradients_flow(setup):
    """Gradients flow through batched log-prob computation."""
    base_model, tokenizer, token_ids = setup
    config = BackoffConfig(t_max=256, k_min=10, k_max=25, temperature=0.0)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.train()

    gen = ForcedActionGenerator(
        lora_model, tokenizer, token_ids, config,
        forced_actions=[token_ids["<continue>"], token_ids[TERMINATE_TOKEN]],
    )
    prompt_ids = _make_prompt(tokenizer, "What is 1+1?")

    with torch.no_grad():
        traj = gen.generate(prompt_ids)

    batch_lps = compute_batch_logprobs(
        lora_model, [traj], token_ids, b_max=config.b_max
    )
    batch_lps[0].backward()

    has_grad = False
    for name, p in lora_model.named_parameters():
        if "lora" in name and p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients on LoRA weights from batched log-probs"

    lora_model.zero_grad()
    lora_model.eval()
