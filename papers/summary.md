
## Verify

### Lightman et al. 2023 — Let's Verify Step by Step

**Paper:** papers/process_reward/lightman2023_lets_verify_step_by_step.pdf
**Summary:** papers/read/lightman2023_lets_verify_step_by_step.md
**Dataset:** PRM800K — 800K human step-level labels across 75K solutions to 12K MATH problems (https://github.com/openai/prm800k)

**Key findings:**
- PRM (step-level verification) beats ORM (outcome-only) by +5.8pp on MATH best-of-1860
- Only need to label up to the first error — error *location* is the critical signal
- Active learning on "convincing wrong-answer" solutions gives 2.6x data efficiency
- Generalizes OOD to AP exams and AMC competitions

**What we can use:**
- PRM800K as training data for a step-level verifier (or as supervision for teaching the model to self-verify)
- The "label up to first error" scheme maps directly to our semantic boundary framework — detect *where* the first wrong chunk is
- Replace `<backoff_N>` (undo + rewrite, cold-start problem) with `<verify>` (detect error, then self-correct via appending) — simpler action, proven effective
- Our existing meta-prover (force-answer-at-boundary) is already a verification mechanism — the pivot is making the model learn to invoke it explicitly

### Luo et al. 2024 — OmegaPRM (Google DeepMind)

**Paper:** papers/process_reward/luo2024_omegaprm.pdf
**Summary:** papers/read/luo2024_omegaprm.md

**Key findings:**
- Fully automated process supervision — no human labels, beats PRM800K
- MCTS + binary search for first error → 75x more efficient than brute-force MC rollouts
- 1.5M step-level annotations from 12K MATH problems
- Soft MC labels outperform hard binary labels (70.1% vs 63.3% PRM accuracy)
- Gemini Pro: 51% → 69.4% on MATH500 with PRM-weighted majority voting

**How it actually works:**
- "Correct step" = from this prefix, at least 1 of k rollouts reaches the gold answer. "Wrong step" = none do. No reasoning analysis — purely outcome-based.
- Binary search narrows down the first error: split solution at midpoint, roll out, check which half fails. O(k log M) vs O(kM).
- MCTS reuses rollouts across searches + prioritizes "high MC but wrong answer" rollouts (most informative).
- Noisy but works at scale — k=8 rollouts, filter too-hard/too-easy questions, 1.5M labels overwhelm noise.

**What we can use:**
- Run OmegaPRM on Qwen3-1.7B rollouts to generate our own process supervision data — on-policy, no distribution shift
- Binary search for first error is cheap enough for 2×A6000 — just needs rollouts + answer checking
- Soft MC labels can train a small PRM as the `<verify>` oracle
- Step boundaries don't need to be semantically precise — relaxes our boundary detection requirements

### Uesato et al. 2022 — Process vs Outcome Feedback (DeepMind)

**Paper:** papers/process_reward/uesato2022_process_outcome_feedback.pdf
**Summary:** papers/read/uesato2022_process_outcome_feedback.md

**Key findings:**
- On GSM8K, ORM and PRM achieve similar final-answer error (~13-14%). Later overturned on harder MATH by Lightman.
- But trace error (wrong reasoning, right answer) differs drastically: outcome-based 12-20%, process-based 3-4%
- ORMs implicitly learn step-level verification — ORM predictions agree more with human step labels (85%) than with ORM's own training labels (77%)
- RL against a reward model (ORM-RL, PRM-RL) beats RL against final-answer correctness across all settings

**What we can use:**
- ORM-approximates-PRM: if we can't afford step labels, an ORM trained on final-answer correctness may still be a decent step verifier — cheap bootstrap for `<verify>`
- Expert iteration (filter-and-distill) is simpler than GRPO and still effective — fallback if GRPO is too unstable
- The trace-error metric is exactly what we care about: models that get right answers via wrong reasoning are dangerous, and only process supervision fixes this

### Zhang et al. 2025 — Lessons Developing PRMs (Qwen Team, Alibaba)

**Paper:** papers/process_reward/zhang2025_lessons_developing_prms.pdf
**Summary:** papers/read/zhang2025_lessons_developing_prms.md
**Released models:** [Qwen2.5-Math-PRM-7B](https://hf.co/Qwen/Qwen2.5-Math-PRM-7B), [Qwen2.5-Math-PRM-72B](https://hf.co/Qwen/Qwen2.5-Math-PRM-72B)

**Key findings (reality check on MC-based PRMs):**
- MC estimation trains **value models** (future potential), not PRMs (step correctness) — wrong step can get positive label if compensating errors downstream lead to correct answer
- MC-trained PRMs look good on BoN but fail on ProcessBench (actual step error detection): 40.1 F1 vs human annotation's 56.5 F1 with 3× less data
- Most open-source PRMs are secretly ORMs — >40% concentrate min-score on the final answer step
- Process error rate scales with difficulty: GSM8K 3.1%, MATH 11.9%, OlympiadBench 27.4%, Omni-MATH 43.4%
- Hard labels > soft labels (after filtering) — opposite of OmegaPRM's finding
- Consensus filtering (keep only when MC ∩ LLM-as-a-judge agree on error location) → 40% of data, matches full LLM-judge quality

**LLM-as-a-judge (how it works):**
- Post-hoc annotation, not online: generate full rollout → split into steps → feed (problem + full solution) to a strong LLM (Qwen2.5-72B-Instruct) → judge reads step-by-step, stops at first error
- Pipeline: `[Generate rollouts] → [LLM judge labels steps] → [Train PRM on labels] → [Use PRM at inference]`
- Better than MC because it actually reads the math ("is 5×7=35?") rather than asking "can we still reach the right answer from here?"
- Tradeoff: needs a strong judge model (72B), MC only needs the policy model itself for rollouts

**What we can use:**
- **Use Qwen2.5-Math-PRM-7B directly as `<verify>` oracle** — open-source, SOTA among 7B PRMs (73.5 avg F1 on ProcessBench), same model family we use. Saves building our own PRM.
- If generating our own data via OmegaPRM, must add consensus filtering with LLM-as-a-judge or accept we're training a value model, not a verifier
- Evaluate with ProcessBench, not just BoN — BoN alone is misleading for process verification quality

## Backtracking

### Yang et al. 2025 — Self-Backtracking (Nanjing University)

**Paper:** papers/backtracking/yang2025_self_backtracking.pdf
**Summary:** papers/read/yang2025_self_backtracking.md
**Code:** https://github.com/LAMDASZ-ML/Self-Backtracking

**Key findings:**
- SFT-only method: trains the model to emit a `<backtrack>` token when it detects a dead-end reasoning path
- `<backtrack>` always rolls back exactly 1 step; multi-step rollback via chaining (repeat up to depth budget $b$)
- +40pp over SFT+Greedy on Countdown task (Llama3.2-1B: 28.6% → 73.5%)
- Even b=0 (no actual backtracking, just sampling + perplexity selection) gives massive gains — the `<backtrack>` token implicitly acts as a state evaluator
- b=1 is sufficient; b>1 has diminishing returns (alternatives after deeper rollback lack diversity)
- Error action $a_{err}$ is **masked** in training loss — model learns to detect errors, not generate them
- Optimal data ratio: $\mathcal{D}_{op} : \mathcal{D}_{back} \geq$ 1:0.5 (clean examples should outnumber backtrack examples)
- Self-improvement (expert iteration): after 3 rounds, greedy single-pass surpasses original search performance

**How step boundaries work:**
- Only tested on Countdown — each step = one arithmetic operation on one line (rigid format)
- The inference algorithm just rolls back to the previous newline — no ambiguity
- Does NOT address step boundaries in free-form CoT (MATH, open-ended reasoning)

**Comparison with our `<backoff_N>`:**

| Aspect | Self-Backtracking | Our `<backoff_N>` |
|---|---|---|
| Token | Single `<backtrack>` (always 1 step) | `<backoff_1/2/3>` (variable depth) |
| Multi-step rollback | Chained single-step backtracks via inference algorithm | Single token encodes full rewind depth |
| Step boundaries | Trivial (newline-delimited task format) | Learned semantic boundaries in free-form CoT |
| Error source | Synthetically injected (3 types) | Real errors from model's own wrong rollouts |
| Error masking | $a_{err}$ masked in loss | Error chunks kept verbatim |
| Directive | None — just backtracks | Explicit corrective directive after token |
| Inference | Tree search (sample N, expand, select by perplexity) | Standard autoregressive (inline self-correction) |
| Task domain | Countdown only (combinatorial search) | MATH (open-ended reasoning) |

**What we can use:**
- b=1 sufficiency supports our 60% `<backoff_1>` weighting — most value comes from single-step correction
- Data ratio finding (clean ≥ backtrack) informs our mix of clean vs. perturbed trajectories
- Self-improvement loop is directly applicable: train with backoff → generate → filter correct → retrain for fast inference
- Their error masking is the opposite of our approach — worth ablating whether masking error chunks helps in our setting
- The `<backtrack>` token as implicit state evaluator suggests our `<backoff_N>` tokens may teach the model self-evaluation as a side effect
