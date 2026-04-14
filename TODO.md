# TODO — Learning When to Think

## Stage 1: Data Preparation & Evaluation Infrastructure ✅

- [x] `src/pivot/eval_types.py` — shared types (RolloutResult, EvalMetrics)
- [x] `src/pivot/reward.py` — ALP reward function
- [x] `src/pivot/eval_harness.py` — evaluation metrics (accuracy, tokens, cost, action usage)
- [x] `scripts/eval_pivot.py` — CLI eval script with YAML config support
- [x] `configs/eval_qwen_math_7b.yaml` — eval config for base model
- [x] Fix `\dfrac` grading bug in `src/data/math.py`
- [x] Tests for reward + eval (28 passing)
- [x] Run baseline eval on MATH-500 (79.0% accuracy, 643 avg tokens)

## Stage 2: SFT Warmup Dataset ✅

- [x] `src/pivot/tokens.py` — action token setup (continue/refine/terminate)
- [x] `scripts/generate_rollouts_pivot.py` — K=8 rollouts from Qwen2.5-Math-7B on ~1000 MATH train
- [x] `scripts/generate_sft_3action.py` — LLM-based rewriting via Qwen3-14B
- [x] Generate dataset → 816 examples (449 clean, 367 refine), all quality checks pass
- [x] Move old backoff code to `deprecated/`

## Stage 3: SFT Warmup Training

- [x] SFT training script for Qwen2.5-Math-7B-Instruct with LoRA (r=16, α=32)
- [x] Train on `data/sft_3action_math_train.jsonl` (freeze base, train new token embeddings + LoRA)
- [ ] Verify model emits `<continue>`, `<refine>`, `<terminate>` tokens in output

## Stage 4: RL Training (GRPO + DeGRPO)

- [ ] K-rollout sampling loop per prompt
- [ ] 3-action policy with control tokens
- [ ] GRPO advantage computation and policy update
- [ ] DeGRPO gradient weighting (w_ctrl=2.0, w_resp=1.0)
- [ ] Wire ALP reward into training loop
- [ ] `allow_refine` and `degrpo` ablation switches
- [ ] End-to-end smoke test: rollout → reward → GRPO update → eval

## Stage 5: Experiments & Ablations

**Comparison structure:**

| Method | SFT Warmup? | Action Tokens? | What it tests |
|--------|-------------|----------------|---------------|
| CoT baseline | No | No | Starting point (79%) |
| Vanilla GRPO | No | No | Does standard RL on CoT help? |
| Ours (full) | Yes | Yes | Action tokens + ALP + DeGRPO |
| degrpo=false | Yes | Yes | Is control-token upweighting needed? |
| allow_refine=false | Yes | Yes | Is the refine action needed? |
| Different β | Yes | Yes | Sensitivity to length penalty |

- [ ] Train vanilla GRPO (no action tokens, standard GRPO on CoT)
- [ ] Train main method (ALP + DeGRPO, β=0.05)
- [ ] Evaluate all methods on MATH-500
- [ ] Ablation: degrpo=false
- [ ] Ablation: allow_refine=false
- [ ] Ablation: different β values
- [ ] H1: Pareto frontier plot (accuracy vs avg tokens)
- [ ] H2: Token-difficulty correlation (Spearman)
- [ ] H3: Action distribution by difficulty (bar charts)
- [ ] Results tables for writeup

## Stage 6: Presentation (Apr 20)

- [ ] Finalize slides with results
- [ ] Rehearse
- [ ] Present
