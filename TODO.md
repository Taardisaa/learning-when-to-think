# TODO — Learning When to Think

## Stage 1: Parallel Development (Apr 7–11)

### Member 1 — Data Pipeline, Evaluation & Reward
- [x] `src/pivot/eval_types.py` — shared types (RolloutResult, EvalMetrics)
- [x] `src/pivot/reward.py` — ALP reward function
- [x] `src/pivot/eval_harness.py` — evaluation metrics (accuracy, tokens, cost, action usage)
- [x] `scripts/eval_pivot.py` — CLI eval script with YAML config support
- [x] `configs/eval_qwen_math_7b.yaml` — eval config for base model
- [x] Fix `\dfrac` grading bug in `src/data/math.py`
- [x] Tests for reward + eval (28 passing)
- [x] Run baseline eval on MATH-500 with Qwen2.5-Math-7B-Instruct (CoT single-pass)

#### SFT Warmup Dataset (new — see `sft_warmup_plan.md`)
- [x] `src/pivot/tokens.py` — action token setup (continue/refine/terminate)
- [x] `scripts/generate_rollouts_pivot.py` — K=8 rollouts from Qwen2.5-Math-7B on ~1000 MATH train problems
- [x] `scripts/generate_sft_3action.py` — LLM-based rewriting (replaces regex script)
- [x] Run: `python -m scripts.generate_sft_3action --rollouts data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl --rewriter Qwen/Qwen3-14B --gpus 1 --tp 1` → 816 examples (449 clean, 367 refine)
- [x] Quality check: all structural checks pass, directives are error-specific, token placement is natural

### Member 2 — Core RL Training
- [ ] LoRA setup on Qwen2.5-Math-7B-Instruct (attention + MLP projections, r=16, α=32)
- [ ] 3-action policy with special control tokens (continue/refine/terminate)
- [ ] GRPO advantage computation and policy update step
- [ ] DeGRPO gradient weighting — split tokens into control/response slices, apply weighted loss
- [ ] `allow_refine` and `degrpo` ablation switches

### Member 3 — Generation Infrastructure
- [ ] K-rollout sampling loop per prompt (shared by baselines and GRPO)
- [ ] Baseline: CoT single-pass decoding
- [ ] Baseline: direct-answer (no reasoning)

---

## Stage 2: Integration + SFT Warmup (Apr 12–14)
- [ ] Member 1 delivers SFT warmup dataset + token setup to Member 2
- [ ] Member 2 runs SFT warmup training on Qwen2.5-Math-7B-Instruct
- [ ] Wire `compute_alp_rewards` into Member 2's GRPO trainer
- [ ] Wire `RolloutResult` + `evaluate()` into Member 3's rollout loop
- [ ] End-to-end smoke test: rollout → reward → GRPO update → eval
- [ ] Collect baseline results (CoT + direct-answer) on full MATH-500

---

## Stage 3: Experiments (Apr 15–18)
- [ ] Train main method (ALP + DeGRPO, β=0.05, w_ctrl=2.0, w_resp=1.0)
- [ ] Evaluate trained model on MATH-500
- [ ] Ablation: vanilla GRPO (degrpo=false)
- [ ] Ablation: no refine (allow_refine=false)
- [ ] Ablation: different β values
- [ ] H1: Pareto frontier plot (accuracy vs avg tokens across methods)
- [ ] H2: Token-difficulty correlation (Spearman: level vs tokens)
- [ ] H3: Action distribution by difficulty (bar charts)
- [ ] Results tables for writeup

---

## Stage 4: Presentation (Apr 19–20)
- [ ] Finalize slides with results
- [ ] Rehearse
- [ ] Present Apr 20
