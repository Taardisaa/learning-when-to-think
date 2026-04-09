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

### Step 2: Build SFT dataset
`scripts/generate_sft_3action.py` — construct training examples from rollouts.

**Clean examples (~60%)** — from correct rollouts:
1. Split CoT into steps at semantic boundaries (paragraph breaks, sentence-ending + logical connectives)
2. Append `<continue>` after each intermediate step
3. Append `<terminate>` before `\boxed{}`

**Refine examples (~40%)** — from problems with both wrong and correct rollouts:
1. Take wrong rollout, split into steps, add `<continue>` after each
2. At end of wrong reasoning, insert `<refine>` + corrective directive
3. Append correct reasoning with `<continue>` tokens
4. End with `<terminate>` + `\boxed{}`

Directive format (same as existing backoff): "Wait, I got X but that's wrong. [describe error]. Let me [corrective direction]."

Target: ~1500–2000 examples total.

Output: `data/sft_3action_math_train.jsonl`
```json
{"question": "...", "answer": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...with <continue>/<refine>/<terminate> tokens..."}], "has_refine": bool}
```

Step segmentation is deterministic (script-appropriate per CLAUDE.md). Refine directives for complex cases use subagents per CLAUDE.md rules.

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
1. `src/pivot/tokens.py` — token setup (unblocks everything)
2. `scripts/generate_rollouts_pivot.py` + config — generate rollouts (needs GPU, ~30-60 min)
3. `scripts/generate_sft_3action.py` — build dataset from rollouts
4. Update `proposal.tex`, `WORK_SPLIT.md`, `TODO.md`
5. Quality check: inspect 30-50 examples manually

## Verification
1. Rollouts generated: `wc -l data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl` → ~1000 lines
2. SFT data built: `wc -l data/sft_3action_math_train.jsonl` → ~1500-2000 lines
3. Spot-check: every example has exactly one `<terminate>`, correct `\boxed{}`, coherent reasoning
4. Token setup: `python -c "from src.pivot.tokens import ACTION_TOKENS; print(ACTION_TOKENS)"` works
5. Train smoke test: 1 epoch on 50 examples, model generates action tokens in output
