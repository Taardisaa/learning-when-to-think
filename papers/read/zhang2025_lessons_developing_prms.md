# The Lessons of Developing Process Reward Models in Mathematical Reasoning

**Authors:** Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba)

**Date:** 2025 | **Venue:** arXiv:2501.07301

## Problem

Building PRMs that actually work in practice is harder than the literature suggests. The Qwen team found three critical problems when following the standard recipe (MC estimation for data, Best-of-N for evaluation):

| Problem | What goes wrong |
|---|---|
| MC estimation produces bad training data | MC conflates "can a correct answer be reached?" with "is this step correct?" — correct answers from wrong steps and wrong answers from correct steps both introduce noise |
| BoN evaluation is misleading | Policy models generate right-answer-wrong-reasoning solutions; PRMs that tolerate these score well on BoN but fail at actual process verification |
| PRMs degrade into ORMs | Optimizing for BoN pushes PRMs to concentrate their minimum scores on the final answer step, effectively becoming outcome reward models |

The paper asks: **what actually works for building PRMs that genuinely verify reasoning steps, not just predict final answer correctness?**

## Key Concepts

### PRM vs Value Model

A critical distinction the paper highlights:
- **PRM**: Evaluates whether the current step is *correct* — a deterministic judgment about what has happened so far
- **Value model**: Estimates the *potential* of reaching the correct answer from the current state — a forward-looking prediction

MC estimation trains value models, not PRMs. When $c_t = \text{num(correct rollouts)} / \text{num(total rollouts)}$, this estimates future potential, not current-step correctness. A step can be wrong but still lead to correct completions (the model recovers), or correct but lead to no correct completions (the problem is just hard from this point).

**Example:** Step 3 contains an arithmetic error ($5 \times 7 = 32$), but 3 out of 8 completions from step 3 happen to reach the correct final answer anyway (the model makes a compensating error later or restates the computation). MC gives $c_3 = 3/8 = 0.375$ — the step gets a positive label despite being wrong.

### Process Error Rate in Correct-Answer Solutions

The fraction of solutions with correct final answers that still contain reasoning errors. Measured by human annotation. Key finding: this rate increases with problem difficulty.

| Benchmark | Process Error Rate (among correct-answer solutions) |
|---|---|
| GSM8K | 3.1% |
| MATH | 11.9% |
| OlympiadBench | 27.4% |
| Omni-MATH | 43.4% |

On hard benchmarks, nearly half of "correct" solutions have flawed reasoning. A PRM that can't detect these is useless — it just becomes an ORM.

### Consensus Filtering

The paper's proposed data construction method. Keep a training example only when **both** MC estimation and LLM-as-a-judge agree on the error location. This filters out ~60% of the data but dramatically improves quality.

**Example:** For a 6-step solution, MC estimation says steps 1-4 correct, step 5 wrong. LLM-as-a-judge says steps 1-3 correct, step 4 wrong. They disagree on the error location → discard this example. Only when both point to the same first-error step is the example retained.

### ProcessBench

A step-level benchmark (Zheng et al., 2024) that directly measures a PRM's ability to identify the first erroneous step in a reasoning trace (or correctly conclude all steps are correct). This is the ground-truth evaluation for whether a PRM is actually doing process verification, unlike BoN which only measures downstream answer selection.

## Core Design / Method

### Data Construction Pipeline

**Phase 1: MC Estimation (expansion)**
- ~500K queries with gold answers
- 6-8 responses per query from Qwen2/Qwen2.5-Math-Instruct (7B and 72B)
- Split responses into steps at `\n\n` delimiters
- 8 completions per step to estimate correctness
- Hard labels: step correct if any of 8 completions reaches correct answer ($c_t > 0$), otherwise negative
- Hard labels chosen over soft — after filtering, hard labels substantially outperform soft

**Phase 2: Consensus Filtering**
- Use Qwen2.5-Instruct-72B as LLM-as-a-judge to verify each step
- Retain only examples where MC estimation and LLM-as-a-judge agree on error location
- ~40% of data survives filtering
- Result: comparable to LLM-as-a-judge quality with only 40% of the data

### Why Hard Labels Beat Soft Labels (After Filtering)

Before consensus filtering: soft ≈ hard (noise masks the difference).
After consensus filtering: hard >> soft.

Reasoning: (1) Step correctness should be deterministic (correct or not), so continuous scores add noise. (2) With only 8 completions, the MC estimates are crude anyway — 0.125, 0.25, 0.375, etc. Many correct steps get scores like 0.5 or 0.625, diluting the positive signal.

### Optimal MC Threshold

Tested thresholds from 1/8 to 7/8 for binarizing MC estimates. Best results: threshold = 0 (i.e., a step is negative only if $c_t = 0$, meaning all 8 completions fail). Any higher threshold degrades performance on both BoN and ProcessBench.

### PRM Architecture

- Initialized from Qwen2.5-Math-7B/72B-Instruct
- Replace LM head with scalar value head (two linear layers)
- Training losses: cross-entropy (hard labels) or MSE (soft labels) on last tokens of each step
- Remove all steps after the first error during training to prevent confusion

### Three Biases in BoN Evaluation

**Bias 1: Policy model generates right-answer-wrong-reasoning solutions.**
On Omni-MATH, 43.4% of correct-answer solutions contain process errors. BoN rewards selecting these, so a PRM that tolerates bad reasoning gets inflated BoN scores.

**Bias 2: Limited process verification inflates BoN scores.**
MC-trained PRMs show opposite trends on BoN vs ProcessBench (Figure 7): high BoN, low ProcessBench. They score well by selecting correct-answer solutions regardless of reasoning quality.

**Bias 3: PRMs drift toward outcome scoring.**
Many existing PRMs concentrate their minimum step-score on the final answer step (Figure 8). EurusPRM-Stage1/2, Math-Shepherd-PRM-7B, Skywork-PRM-7B all exceed 40% of responses with min-score at the last step. This means the solution score is determined by the final step, making the PRM behave like an ORM.

### Scoring Strategy Depends on Training Data

| Training Data | Best Scoring Strategy |
|---|---|
| MC estimation | **Last** step score |
| LLM-as-a-judge | Product or minimum |
| Human annotation | Product or minimum |

MC-trained PRMs work best with last-step scoring because they've learned value estimation (future potential), not step correctness. Genuine PRMs (LLM/human-trained) work best with product/minimum because each step score is an independent correctness judgment.

## Results

### Best-of-8 on Qwen2.5-Math-7B-Instruct (Table 6)

| Model | GSM8K | MATH | Minerva | GaoKao | Olympiad | College | MMLU STEM | Avg. |
|---|---|---|---|---|---|---|---|---|
| pass@8 (upper bound) | 98.1 | 92 | 49.3 | 80.5 | 59.6 | 52.6 | 90.5 | 74.7 |
| maj@8 | 96.7 | 87.1 | 41.2 | 72.5 | 44.4 | 47.8 | 73.8 | 66.2 |
| Math-Shepherd-PRM-7B | 97.3 | 85.4 | 37.9 | 70.6 | 40.4 | 47.2 | 70.5 | 64.2 |
| Skywork-PRM-7B | 97.3 | 87.3 | 38.2 | 71.9 | 43.7 | 47.8 | 67.7 | 64.8 |
| Qwen2.5-Math-7B-PRM800K | 96.9 | 86.9 | 37.1 | 71.2 | 44.0 | 47.6 | 70.9 | 64.9 |
| **Qwen2.5-Math-PRM-7B** | **97.1** | **88.0** | **42.6** | **74.5** | **47.6** | **48.7** | **74.5** | **67.6** |
| Qwen2.5-Math-PRM-72B | 97.6 | 88.7 | 46.0 | 77.4 | 52.9 | 50.1 | 82.3 | 70.7 |

Qwen2.5-Math-PRM-7B outperforms all 7B PRMs by +2.7% average, beating even models trained on more data.

### ProcessBench — Step Error Identification (Table 7)

| Model | GSM8K F1 | MATH F1 | OlympiadBench F1 | Omni-MATH F1 | Avg. F1 |
|---|---|---|---|---|---|
| Math-Shepherd-PRM-7B | 13.7 | 15.0 | 71.0 | 24.8 | 31.5 |
| Qwen2.5-Math-7B-PRM800K | 31.6 | 62.6 | 35.7 | 50.7 | 56.5 |
| Qwen2.5-Math-7B-Math-Shepherd | 31.6 | 7.4 | 93.8 | 13.7 | 28.9 |
| **Qwen2.5-Math-PRM-7B** | **55.7** | **77.6** | **85.5** | **67.5** | **73.5** |
| Qwen2.5-Math-PRM-72B | 67.9 | 80.6 | 82.0 | 74.3 | 78.3 |
| *LLM-as-judge: o1-mini* | *88.9* | *95.1* | *87.2* | *84.2* | *87.9* |

Qwen2.5-Math-PRM-7B achieves 73.5 avg F1 — vastly better than Math-Shepherd (28.9) or PRM800K-trained (56.5) at the same model size. But still behind o1-mini as judge (87.9).

### Data Construction Method Comparison (Tables 3-4)

**Best-of-8 performance (higher = better for BoN):**

| Method | # samples | Avg. BoN |
|---|---|---|
| MC Estimation (Math-Shepherd) | 440k | 64.3 |
| MC Estimation (their data) | 860k | 65.9 |
| LLM-as-a-judge | 860k | 65.9 |
| Human Annotation (PRM800K) | 264k | 64.9 |

**ProcessBench F1 (higher = better step verification):**

| Method | # samples | Avg. F1 |
|---|---|---|
| MC Estimation (Math-Shepherd) | 440k | 28.9 |
| MC Estimation (their data) | 860k | 40.1 |
| LLM-as-a-judge | 860k | 46.5 |
| Human Annotation (PRM800K) | 264k | 56.5 |

BoN scores are similar across methods, but ProcessBench reveals huge quality differences. Human annotation wins on step verification with 3× less data.

## Key Takeaways

1. **MC estimation trains value models, not PRMs.** The fundamental issue: MC estimates future potential ("can we still get the right answer from here?"), not current-step correctness ("is this step right?"). A wrong step with compensating errors downstream gets a positive MC label. This conflation is why MC-trained PRMs fail at step-level error identification.

2. **BoN evaluation alone is misleading for PRMs.** A PRM can score well on BoN by selecting correct-answer solutions regardless of reasoning quality. ProcessBench (or similar step-level benchmarks) is necessary to evaluate whether the PRM actually verifies process. MC-trained PRMs show *inverse* trends on BoN vs ProcessBench.

3. **Consensus filtering (MC ∩ LLM-as-a-judge) is the practical sweet spot.** Keeping only examples where both methods agree on the error location retains ~40% of data but achieves quality comparable to LLM-as-a-judge alone. More data-efficient than either method alone.

4. **Hard labels beat soft labels after filtering.** Step correctness is binary — a step is either right or wrong. Soft MC scores (0.375, 0.625, etc.) with only 8 completions are too noisy to be useful as continuous labels. After filtering cleans the data, hard labels clearly win.

5. **MC threshold should be 0.** A step should be labeled negative only when all completions fail ($c_t = 0$). Any positive threshold (e.g., $c_t < 0.5$) hurts performance. This is the most permissive criterion — "guilty only if no path to success exists."

6. **Most existing PRMs are secretly ORMs.** When optimized purely for BoN, PRMs learn to concentrate their discriminative signal on the final answer step. Many open-source PRMs have >40% of their minimum scores at the last step — they're just checking the answer, not the reasoning.

7. **Process error rate scales with difficulty.** On GSM8K, only 3.1% of correct-answer solutions have reasoning errors. On Omni-MATH, it's 43.4%. PRMs only matter when problems are hard enough that right-answer-wrong-reasoning is common. This validates focusing on MATH over GSM8K for our work.

8. **Human annotation remains most data-efficient.** PRM800K (264K samples) achieves 56.5 avg F1 on ProcessBench — beating MC estimation with 860K samples (40.1 F1). Quality > quantity for step-level verification.

9. **Released models: Qwen2.5-Math-PRM-7B and 72B** are SOTA among open-source PRMs. Available at https://hf.co/Qwen/Qwen2.5-Math-PRM-7B and https://hf.co/Qwen/Qwen2.5-Math-PRM-72B.

## Connection to Our Work

This paper is a critical reality check for our `<verify>` pivot:

- **Don't blindly use MC estimation (OmegaPRM) for training data.** The Qwen team found MC-trained PRMs are weak at actual step verification despite looking good on BoN. If we generate process supervision data via OmegaPRM, we should either add consensus filtering with an LLM judge, or accept that we're training a value model, not a verifier.
- **We can use Qwen2.5-Math-PRM-7B directly** as our `<verify>` oracle instead of training our own PRM. It's open-source, 7B parameters, SOTA on ProcessBench, and trained on the same Qwen2.5 family we're using. This saves the entire PRM training pipeline.
- **Evaluate with ProcessBench, not just BoN.** When we build our `<verify>` mechanism, we need to measure whether verification actually catches step errors, not just whether it improves final-answer accuracy.
- **The process error rate data (Table in Figure 6) validates our approach.** On hard problems (OlympiadBench, Omni-MATH), 27-43% of correct-answer solutions have reasoning errors. This is exactly the gap that `<verify>` should exploit — catch these errors before they compound.
