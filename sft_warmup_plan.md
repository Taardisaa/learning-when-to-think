# Plan: Stage 2 — SFT Warmup Dataset for 3-Action Tokens

## Context
GRPO needs the model to emit `<continue>`, `<refine>`, `<terminate>` tokens, but Qwen2.5-Math-7B-Instruct has never seen them. Without SFT warmup, the model assigns near-zero probability to these tokens and RL can't learn. We need an SFT dataset that teaches the token format before GRPO training begins.

**Key facts about Qwen2.5-Math-7B-Instruct:**
- Does CoT natively as plain text (no `<think>`/`</think>`)
- Uses ChatML format (`<|im_start|>`, `<|im_end|>`)
- Baseline: 79% on MATH-500, avg 643 tokens

## Action Token Format

No `<think>` wrapper — tokens appear inline in the natural CoT:
```
[step 1] <continue>
[step 2] <continue>
[step 3 — wrong] <refine> Wait, I made an error. [corrective reasoning] <continue>
[final step] <terminate>
\boxed{answer}
```

Rules:
- `<continue>` after each intermediate reasoning step (at semantic boundaries)
- `<refine>` after a wrong step, followed by corrective directive + corrected reasoning
- `<terminate>` after the final step, immediately before `\boxed{}`
- Every example has exactly one `<terminate>`

## Pipeline

### Step 1: Generate rollouts from Qwen2.5-Math-7B-Instruct
`scripts/generate_rollouts_pivot.py` — generate K=8 rollouts on ~1000 MATH train problems via vLLM.

Output: `data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl`
```json
{"question": "...", "answer": "...", "rollouts": [{"text": "...", "predicted": "...", "correct": bool, "num_tokens": int}], "pass_rate": float}
```

Reuse: `src/data/math.py:load_math_train()`, `extract_boxed_answer()`, `grade_math_answer()`

### Step 2: Build SFT dataset via LLM rewriting
`scripts/generate_sft_3action.py` — use a stronger model (Qwen3-32B) to rewrite rollouts with action tokens.

**Why LLM-based, not script-based:** Regex-based splitting puts tokens at wrong boundaries (mid-calculation, mid-argument). Naive stitching of wrong+correct rollouts produces incoherent reasoning. An LLM handles all the judgment calls — boundary detection, error identification, directive writing — in one pass.

**Approach:** For each rollout, prompt Qwen3-32B via vLLM with:
- Token definitions and placement rules
- 2-3 few-shot examples (clean + refine)
- The original question + rollout text + gold answer
- Instruction to rewrite the trajectory with properly placed action tokens

**Clean examples (~60%)** — feed a correct rollout, ask the model to insert `<continue>` at semantic boundaries and `<terminate>` before the answer.

**Refine examples (~40%)** — feed a wrong rollout + the gold answer, ask the model to:
1. Identify where the reasoning went wrong
2. Insert `<continue>` tokens at proper boundaries up to the error
3. Insert `<refine>` at the error point with a specific directive
4. Write corrective reasoning leading to the correct answer
5. End with `<terminate>` + `\boxed{gold_answer}`

**Quality filter:** After generation, validate each example:
- Exactly one `<terminate>`, at least one `<continue>`
- `\boxed{}` present and matches gold answer
- `<terminate>` appears before `\boxed{}`
- For refine examples: `<refine>` token is present
- Reject malformed outputs

Target: ~1500–2000 examples after filtering.

Output: `data/sft_3action_math_train.jsonl`
```json
{"question": "...", "answer": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...with <continue>/<refine>/<terminate> tokens..."}], "has_refine": bool}
```

### Step 3: Token setup
`src/pivot/tokens.py` — add 3 new special tokens, initialize embeddings.
```python
ACTION_TOKENS = ["<continue>", "<refine>", "<terminate>"]
```
- `setup_tokenizer_and_model()`: add tokens, resize embeddings, init with mean+noise
- `get_action_token_ids()`: return token IDs for the 3 actions

### Step 4: SFT training
Reuse `scripts/train_phase1.py` pattern with updated defaults:
- Model: `Qwen/Qwen2.5-Math-7B-Instruct`
- Data: `data/sft_3action_math_train.jsonl`
- LoRA: r=16, alpha=32, on q/k/v/o + gate/up/down projections
- Freeze base embeddings, only train new token embeddings + LoRA

## Files to create
- `src/pivot/tokens.py` — action token setup
- `scripts/generate_rollouts_pivot.py` — rollout generation
- `scripts/generate_sft_3action.py` — SFT dataset construction
- `configs/generate_rollouts_qwen25.yaml` — rollout gen config
- `configs/sft_3action.yaml` — SFT training config

## Files to modify
- `proposal.tex` — add SFT warmup phase to Method section
- `WORK_SPLIT.md` — add SFT data prep to Member 1's tasks, update timeline
- `TODO.md` — add SFT warmup tasks

## Existing code to reuse
- `src/data/math.py`: `load_math_train()`, `extract_boxed_answer()`, `grade_math_answer()`
- `src/tokens.py`: reference for `setup_tokenizer_and_model()` pattern (mean+noise init)
- `scripts/generate_sft_real_backoff.py`: reference for rollout generation + stitching pattern
- `scripts/train_phase1.py`: reference for SFT training loop

## Implementation order
1. ✅ `src/pivot/tokens.py` — token setup
2. ✅ `scripts/generate_rollouts_pivot.py` + config — generate rollouts (27 min on 1 GPU)
3. ✅ `scripts/generate_sft_3action.py` — LLM-based rewriting via Qwen3-14B
4. ✅ Update `proposal.tex`, `WORK_SPLIT.md`, `TODO.md`
5. ✅ Quality check

## Completed Results

**Rollout generation:**
```
python -m scripts.generate_rollouts_pivot --config configs/generate_rollouts_qwen25.yaml
→ 1000 problems, K=8, pass rate 74.8%, 631 trivial / 200 mixed / 169 impossible
```

**SFT dataset generation:**
```
python -m scripts.generate_sft_3action \
    --rollouts data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl \
    --rewriter Qwen/Qwen3-14B --gpus 1 --tp 1
→ 816 examples (449 clean, 367 refine = 55%/45%)
```

**Quality checks (all pass):**
- Every example: exactly 1 `<terminate>`, ≥1 `<continue>`, correct `\boxed{}`
- `<terminate>` always before `\boxed{}`
- All answers match gold
- Refine directives are error-specific (not generic templates)
- Avg 6.6 `<continue>` per example, median token length 456

## Remaining
- SFT warmup training (Member 2's responsibility, Stage 2)
