# Implementation Plan: Learning When to Think

## Context

Train Qwen3.5 models (0.8B/2B/4B/9B) to learn a `<backoff>` action that truncates the KV cache mid-generation, injects a corrective directive, and resumes reasoning from a cleaner state. The method spec is fully documented in `project_docs/02_method.md` and `03_training.md`. Baselines (forward-only GSM8K eval) are complete. Zero training code exists yet.

**Decision**: Upgrade `transformers` to 5.x for Qwen3.5 support. Install `peft` and `trl`.

---

## Project Structure

```
src/
  __init__.py
  config.py              # Central config dataclass
  tokens.py              # Special token setup
  boundary.py            # Semantic boundary detection
  generation.py          # Backoff-aware generation loop
  data/
    __init__.py
    gsm8k.py             # Data loading + answer extraction (adapt from Ivan's)
    synthetic.py          # SFT synthetic data generation
  train/
    __init__.py
    sft.py               # Phase 1 SFT wrapper
    logprobs.py           # Segment-wise forward passes
    reward.py             # Reward computation
    rollout.py            # Rollout generation for GRPO
    grpo.py               # GRPO training loop
  eval/
    __init__.py
    evaluate.py           # Evaluation pipeline
scripts/
  generate_sft_data.py    # CLI: generate synthetic SFT data
  train_phase1.py         # CLI: Phase 1 SFT
  train_phase2.py         # CLI: Phase 2 GRPO + exploration bonus
  train_phase3.py         # CLI: Phase 3 pure GRPO
  evaluate.py             # CLI: evaluation
  compare_results.py      # CLI: comparison tables + plots
  run_experiment.py       # CLI: orchestrate full pipeline
  ablation_no_backoff.py
  ablation_mask_vs_truncate.py
  ablation_no_directive.py
configs/
  default.yaml
tests/
  test_tokens.py
  test_boundary.py
  test_generation.py
  test_logprobs.py
```

---

Important NOTES: use venv's python!

## Milestone 0: Environment Setup DONE

**Do**: Upgrade transformers, install peft + trl, verify Qwen3.5 loads.

```bash
pip install --upgrade transformers peft trl accelerate
```

**Verify**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import LoraConfig, get_peft_model
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype="bfloat16", device_map="auto")
t = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
print(t.convert_tokens_to_ids("</think>"))  # should be 248069
```

If vLLM breaks after upgrade, it's OK -- baselines are already computed, and training eval uses transformers directly.

---

## Milestone 1: Core Utilities (tokens, boundary, data, config)

### `src/config.py`
Central `BackoffConfig` dataclass. All hyperparameters:
- Model: `model_name`, LoRA params (r=16, alpha=32, targets=q/k/v/o)
- Boundary: `k_min=15`, `k_max=80`
- Backoff: `d_max=3`, `k_dir=20`, `b_max=2`, `t_max=2048`
- Reward: `lambda_tok=0.0001`, `lambda_explore=0.1`, `anneal_steps=500`
- GRPO: `num_rollouts=4`, `clip_epsilon=0.2`, `kl_coef=0.01`, `lr=1e-4`
- SFT: `epochs=3`, `lr=2e-5`, `max_seq_len=2048`

### `src/tokens.py`
- Add 5 new tokens: `<continue>`, `<backoff>`, `<depth_1>`, `<depth_2>`, `<depth_3>`
- `<terminate>` = existing `</think>` (already at id 248069 in Qwen3.5) -- do NOT re-add
- `setup_tokenizer_and_model(tokenizer, model)` -> adds tokens, resizes embeddings, returns `token_ids` dict
- Initialize new embeddings from mean of existing + small noise (better than pure random)

### `src/boundary.py`
- `is_boundary(token_text)` -- checks against boundary set: punctuation (`. ; ? ! \n\n`), logic connectives (`therefore, however, so, but, thus, hence, because`), structural (`Step, Answer:, Therefore,`)
- `BoundaryTracker` class: tracks chunk length, fires when `chunk_len >= k_min AND boundary hit` or `chunk_len >= k_max`

### `src/data/gsm8k.py`
Adapt from Ivan's `marie/ivan:src/data/gsm8k.py`:
- `load_gsm8k(split, subset_size)` -> `[{question, answer_number, full_answer}]`
- `extract_answer_number(text)` -- parse `####` format
- `extract_predicted_number(text)` -- last number heuristic
- `grade_answer(predicted, gold, tol=1e-3)` -- float comparison

**Verify**: Unit tests for token IDs, boundary detection, answer extraction.

---

## Milestone 2: Backoff-Aware Generation Loop DONE

**File**: `src/generation.py`

This is the core inference engine. Implements the generation loop from `02_method.md` lines 92-148.

### `TrajectoryRecord` dataclass
Stores everything needed for GRPO:
- `segments`: list of `Segment` (each has: `chunk_ids`, `action`, `depth`, `directive_ids`, `kv_start_pos`, `kv_end_pos`)
- `final_kv_length`: net tokens at termination (for reward)
- `answer_text`, `answer_number`, `backoff_count`, `total_generated_tokens`
- `stored_log_prob`: accumulated during generation (for importance ratio denominator)
- `build_forward_pass_sequences()`: reconstructs the k+1 token sequences for segment-wise log-prob computation

### `BackoffGenerator` class
```
__init__(model, tokenizer, token_ids, config)
generate(prompt_ids) -> TrajectoryRecord
```

**Generation loop**:
1. Encode prompt into KV cache (`DynamicCache`)
2. Loop until terminate or T_max:
   a. **Generate chunk**: token-by-token via `model(input_ids, past_key_values, use_cache=True)`. Sample from full vocab. Track boundary detection.
   b. **Action decision**: one forward pass with logits masked to action tokens only (mask `<backoff>` if `backoff_count >= b_max`). Sample action.
   c. **Execute**:
      - `<continue>`: action token enters KV cache, next chunk
      - `<backoff>`: sample depth token (logits masked to depth tokens), call `kv_cache.crop(target_pos)`, generate directive (up to k_dir tokens, stop on `<continue>`), resume
      - `<terminate>` (`</think>`): let model generate answer freely, extract number
3. Return `TrajectoryRecord`

**Critical details**:
- After `kv_cache.crop(target_pos)`, next forward pass must use `cache_position` starting from `target_pos`
- Accumulate log-prob of each sampled token during generation (store as `stored_log_prob`)
- For action/depth tokens, compute log-prob under the masked distribution (not full vocab)

**Verify**:
- Load base Qwen3.5-0.8B (no training), add tokens, run on 5 GSM8K problems
- Check `kv_cache.get_seq_length()` decreases after `crop()`
- Print trajectories, verify format is coherent (action decisions at boundaries)

---

## Milestone 3: Synthetic SFT Data Generation

**File**: `src/data/synthetic.py`, `scripts/generate_sft_data.py`

### Data construction (no LLM needed -- rule-based)
For each GSM8K train problem:
- Parse gold solution into sentences/steps
- **60% clean examples**: insert `<continue>` at sentence boundaries, end with `</think>` + answer
- **40% backoff examples**: inject a wrong step (arithmetic error, copy error, logic skip), follow with `<backoff> <depth_d> [directive] <continue>`, then correct continuation

### Wrong step generation (rule-based heuristics)
- Arithmetic: swap a number in a calculation (e.g., 48/2=12 instead of 24)
- Copy: use wrong number from prior step
- Directives: "the calculation was wrong, redo carefully", "used wrong number, check again", etc.

### Format (Qwen3.5 chat template)
```
<|im_start|>user
Solve the following math problem...
[question]<|im_end|>
<|im_start|>assistant
<think>
[step 1] <continue>
[wrong step] <backoff> <depth_1> [directive] <continue>
[corrected step] <continue>
[final step] </think>
#### [answer]<|im_end|>
```

**Output**: `data/sft_train.jsonl` (~500-2000 examples)

**Verify**: Inspect 20 random examples. Check `<continue>` at every boundary, `<backoff>` in ~40%, correct answer always present.

---

## Milestone 4: Phase 1 -- SFT Warm-Up

**Files**: `src/train/sft.py`, `scripts/train_phase1.py`

### Implementation
- Load Qwen3.5-0.8B, add special tokens via `tokens.setup_tokenizer_and_model()`
- Apply LoRA (r=16, alpha=32, targets=q/k/v/o) via peft
- Load SFT data from JSONL
- Train with TRL `SFTTrainer`: 3 epochs, lr=2e-5, batch=1, grad_accum=4, bf16, max_seq_len=2048
- **Critical**: ensure loss includes new special tokens (not masked by data collator)
- Save LoRA adapter + modified tokenizer to `checkpoints/phase1/final`

**Verify**:
- Loss drops significantly in epoch 1 (learning new token embeddings)
- Generate on 20 test problems: output should contain `<continue>` at boundaries
- Model occasionally produces `<backoff>` (even if placement is bad -- format is learned)

---

## Milestone 5: Segment-Wise Forward Pass (hardest component)

**File**: `src/train/logprobs.py`

### Core function
```python
def compute_trajectory_logprobs(model, trajectory, token_ids) -> torch.Tensor
```

For a trajectory with k backoffs, perform k+1 forward passes:
- **Pass 1**: `prompt + chunk1 + <continue> + chunk2 + <backoff> + <depth_d>` -- score all tokens
- **Pass 2**: `prompt + chunk1 + directive + <continue> + chunk3 + ...` -- score directive onward

Each pass: full forward, compute log_softmax, gather log-probs for target tokens.

### Action token masking in log-prob computation
For action tokens (`<continue>`, `<backoff>`, `<terminate>`): compute log_softmax only over action token logits, not full vocab. Same for depth tokens. This ensures the log-prob matches what was used during sampling.

```python
if is_action_token(token_id):
    masked_logits = logits[action_ids]
    log_prob = F.log_softmax(masked_logits, dim=-1)[action_index]
else:
    log_prob = F.log_softmax(logits, dim=-1)[token_id]
```

### Reference model log-probs
Same function, but called with `torch.no_grad()` on the frozen reference model.

**Verify** (critical -- get this right before GRPO):
1. Generate a 0-backoff trajectory. Verify `compute_trajectory_logprobs` matches a single forward pass over the full sequence.
2. Generate a 1-backoff trajectory. Manually construct the two pass sequences. Verify segment boundaries are correct.
3. Verify gradients flow: `total_log_prob.backward()` produces non-zero gradients on LoRA weights.

---

## Milestone 6: Phase 2 -- GRPO + Exploration Bonus

**Files**: `src/train/reward.py`, `src/train/rollout.py`, `src/train/grpo.py`, `scripts/train_phase2.py`

### `reward.py`
```python
R(tau) = correctness - lambda_tok * T_net + lambda_explore * has_backoff
```
- `correctness`: 1.0 if exact match, else 0.0
- `T_net`: `trajectory.final_kv_length` (net tokens, not gross)
- Exploration bonus anneals: `lambda_explore(step) = lambda_explore * max(0, 1 - step/anneal_steps)`

### `rollout.py`
- `generate_rollouts(model, tokenizer, token_ids, question, config, G)` -> list of `TrajectoryRecord`
- Uses `BackoffGenerator` in inference mode, temperature > 0 for diverse rollouts

### `grpo.py` -- `grpo_step()`
For each question in batch:
1. Generate G=4 rollouts (no grad)
2. Compute rewards, group-relative advantages: `A_i = (R_i - mean) / std`
3. For each rollout with |A_i| > eps:
   a. `log_pi_theta = compute_trajectory_logprobs(model, traj)` (with grad)
   b. `rho = exp(log_pi_theta - traj.stored_log_prob)` (importance ratio)
   c. Clipped surrogate: `loss = max(-rho * A, -clip(rho, 1-eps, 1+eps) * A)`
   d. KL penalty: `kl = log_pi_theta - log_pi_ref` (ref model, no grad)
   e. `(loss + kl_coef * kl).backward()`
4. Gradient clip (max_norm=1.0), optimizer step

### Reference model
- Load base Qwen3.5-0.8B + merge Phase 1 LoRA into base weights (`model.merge_and_unload()`)
- Freeze all params
- This is the "policy that knows the tokens but hasn't learned strategy"

### Training script
- Load Phase 1 checkpoint, apply new LoRA for Phase 2
- AdamW optimizer, lr=1e-4
- Train on GSM8K train set (500 problems)
- Log per step: loss, avg_reward, accuracy, backoff_rate, avg_T_net
- Save checkpoints every 100 steps
- Transition to Phase 3 when backoff_rate stabilizes at 10-50% and lambda_explore ~ 0

**Verify**:
- Run 10 steps on 10 problems: loss is finite, gradients non-zero, backoff_rate > 0%
- Memory fits in 96GB (Qwen3.5-0.8B: ~1.6GB bf16, x2 for train+ref, + activations)

---

## Milestone 7: Phase 3 -- Pure GRPO

**File**: `scripts/train_phase3.py`

Identical to Phase 2 but `lambda_explore = 0` throughout. Loads Phase 2 checkpoint.

**Monitor**:
- If backoff_rate drops to <5%: backoff isn't helping (H1 may fail for this model size)
- If backoff_rate stays 10-30%: model has learned when backoff is useful
- If backoff_rate >50%: over-reliance, may need backoff penalty

---

## Milestone 8: Evaluation Pipeline

**Files**: `src/eval/evaluate.py`, `scripts/evaluate.py`, `scripts/compare_results.py`

### Evaluation modes
1. **Backoff-enabled**: uses `BackoffGenerator` on test set (transformers, not vLLM)
2. **Forward-only**: same trained model but mask `<backoff>` in action decisions (only `<continue>` + `<terminate>`)
3. **Base model**: existing baseline numbers from `baselines/`

### Metrics
- Accuracy, avg T_net, backoff_rate, avg depth, cost_per_correct
- Wilson score 95% CI (from Ivan's `wilson_ci()`)
- Token savings: `(T_forward - T_net_backoff) / T_forward`

### Comparison output
- JSON results file per evaluation
- Comparison table (stdout + JSON)
- Plots: accuracy vs token cost scatter, accuracy bar chart, backoff_rate over training

---

## Milestone 9: Ablations

Three ablation scripts, each trains a variant and evaluates:

1. **No backoff** (`ablation_no_backoff.py`): action space = `{<continue>, <terminate>}` only. Same GRPO setup. This is the key comparison -- does `<backoff>` actually help?

2. **Mask vs truncate** (`ablation_mask_vs_truncate.py`): on backoff, instead of `kv_cache.crop()`, apply attention mask to zero out deleted tokens. Tests whether context reclamation matters.

3. **No directive** (`ablation_no_directive.py`): on backoff, truncate KV cache but skip directive generation. Resume immediately. Tests whether directives add value beyond simple retry.

---

## Milestone 10: Orchestration + Scale Up

**File**: `scripts/run_experiment.py`

Orchestrates full pipeline for a given model size:
1. Generate SFT data (if not exists)
2. Phase 1 SFT -> Phase 2 GRPO+explore -> Phase 3 pure GRPO
3. Evaluate (backoff, forward-only, ablations)
4. Generate comparison plots

Each step checks if output exists (resume-friendly).

**Scale plan**: Develop on 0.8B, then run on 2B/4B/9B. Larger models may need:
- Reduced `num_rollouts` (G=2 instead of 4)
- Gradient checkpointing
- Multi-GPU with accelerate

---

## Implementation Order

```
M0: Environment Setup                    (no deps)
M1: Config + Tokens + Boundary + Data    (depends on M0)
M2: Generation Loop                      (depends on M1)  <-- can parallel with M3
M3: Synthetic SFT Data                   (depends on M1)  <-- can parallel with M2
M4: Phase 1 SFT                          (depends on M1, M3)
M5: Segment-wise Log-probs               (depends on M2)
M6: Phase 2 GRPO                         (depends on M2, M4, M5)
M7: Phase 3 GRPO                         (depends on M6)
M8: Evaluation                           (depends on M2, can start early)
M9: Ablations                            (depends on M6, M8)
M10: Orchestration                       (depends on all)
```

## Key Risks

1. **Segment-wise forward pass correctness** (M5) -- if log-probs are wrong, GRPO won't converge. Extensive unit testing required.
2. **KV cache position IDs after crop** -- must use correct `cache_position` parameter. Test thoroughly.
3. **Action token log-prob masking** -- must match the masked distribution used during sampling, not full vocab.
4. **Exploration failure** -- if backoff_rate stays at 0% in Phase 2, increase lambda_explore or temperature for action decisions.
5. **transformers 5.x upgrade** -- may break vLLM. Acceptable since baselines are computed; training eval uses transformers directly.
