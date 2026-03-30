# Solving Math Word Problems with Process- and Outcome-Based Feedback

**Authors:** Jonathan Uesato*, Nate Kushman*, Ramana Kumar*, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, Irina Higgins (DeepMind)

**Date:** 2022 | **Venue:** arXiv:2211.14275

## Problem

When training LLMs to generate step-by-step reasoning, two supervision paradigms exist:

| Supervision | What it labels | Signal source |
|---|---|---|
| Outcome-based | Final answer only | Automatic (answer checking) |
| Process-based | Each reasoning step | Human annotators |

Prior work assumed these would differ significantly in final-answer accuracy, but no comprehensive comparison existed on a natural language reasoning task. Additionally, a model can get the right answer via wrong reasoning (false positives), which outcome supervision rewards. The paper asks: **on GSM8K, does the type of supervision matter for (a) final-answer accuracy and (b) reasoning trace correctness?**

## Key Concepts

### Trace Error Rate

The fraction of problems with correct final answers where the model's reasoning trace contains at least one incorrect step (as judged by human annotators). This is the "hidden danger" metric — the model looks right but reasoned wrong.

**Example:** Model solves "Tyrion changes face masks 2 times every time he goes out, 3 times a day, for 2 days" by computing 3×2=6 masks/day, then 6×2=12. The final answer 12 is correct, but intermediate step "Tyrion goes out 6 face masks every day" is nonsensical. Trace error = 1 for this problem despite correct final answer.

### Final-Answer Error Rate

The fraction of problems where the model's final answer is wrong. Easy to compute automatically via string matching (GSM8K answers are always integers).

### Outcome-Supervised Reward Model (ORM)

A reward model where every step in a solution receives the same binary label: whether the full solution's final answer is correct. Trained to predict a "correct/incorrect" token after each step. At test time, the prediction at the final step serves as the solution score.

**Example:** A 5-step solution reaching the correct answer → all 5 steps labeled "correct." A 5-step solution reaching a wrong answer → all 5 steps labeled "incorrect." The ORM doesn't know which specific steps are right or wrong.

### Process-Supervised Reward Model (PRM)

A reward model where each step receives its own binary label from human annotators: are the steps so far correct? A step is "incorrect" if "it would no longer be possible to reach the correct solution without undoing that step."

**Example:** In a 5-step solution, steps 1-3 are correct, step 4 introduces an error. Labels: steps 1-3 = correct, steps 4-5 = incorrect. Human annotator only needs to identify the first error.

### Expert Iteration (RL via Expert Iteration)

An RL meta-algorithm alternating two phases:
1. **Policy improvement:** Generate K solutions per problem, filter/rank them using some criterion (final-answer correctness, ORM score, or PRM score)
2. **Distillation:** Train the base policy on the filtered "expert" samples via SFT

Three variants tested:
- **Final-Answer RL:** Filter by final-answer correctness only
- **ORM-RL:** Rank by ORM score, select best
- **PRM-RL:** At each step, generate K candidates, select by PRM score, continue (stepwise beam search)

### RM-Weighted Decoding

Generate K=96 solutions, then select the best one. First pick the final answer with the largest total RM-weighted probability mass, then pick the highest-RM-scored solution with that answer. Slightly better than pure best-of-K (about 1% on final-answer error).

## Core Design / Method

### Training Pipeline

Three components combined in different configurations:

1. **Base policy:** Either SFT (finetuned on GSM8K reasoning traces) or few-shot (5-shot prompted base LM, Chinchilla 70B)
2. **Reward model:** ORM or PRM, trained on the base policy's samples
3. **RL:** Expert iteration using Final-Answer, ORM, or PRM feedback

All models based on Chinchilla 70B (Hoffmann et al., 2022).

### ORM Training

Standard binary classification. For each solution sample:
- If final answer correct → all steps labeled "correct"
- If final answer wrong → all steps labeled "incorrect"

Trained on K=96 samples per problem from the policy, temperature 1.0. Initialized from SFT model (for SFT-based) or base LM (for few-shot-based).

### PRM Training

Binary classification per step, using human annotations. Annotators identify the first major mistake; all steps before it are "correct," all after are "incorrect."

Dataset: 1,560 model solutions across 530 training problems, yielding 9,856 step-level binary labels. Small dataset — initialized from ORM parameters and trained with lower learning rate ($1 \times 10^{-7}$).

### PRM-RL (Stepwise Expert Iteration)

At each step, generate K=96 candidate next-steps, select the one with highest PRM score, continue until the final answer or max 15 steps. This creates a stepwise beam search guided by the PRM.

### The Surprising ORM-PRM Agreement Finding

Despite being trained only on final-answer labels, ORM predictions agree more with PRM (human step-level) labels than with ORM labels themselves:

| | ORM | Few-shot ORM | ORM labels | PRM | PRM labels |
|---|---|---|---|---|---|
| ORM agreement | 1.0 | 0.88 | 0.77 | 0.87 | 0.85 |
| PRM labels agreement | 0.85 | 0.84 | 0.81 | 0.87 | 1.0 |

ORM agrees with PRM labels at 0.85, but with its own ORM labels at only 0.77. The ORM has learned to check intermediate steps rather than just predict the final answer — because recognizing step correctness is easier than internally computing the answer.

## Results

### Main Results (Table 1)

| Approach | Trace Error (%) | Final-Answer Error (%) |
|---|---|---|
| Few-shot, Majority Voting | — | 41.5 |
| Few-shot + Final-Answer RL, Majority Voting | 19.8 (7.9-31.7) | 23.5 |
| SFT, Majority Voting | 11.4 (4.8-18.0) | 22.3 |
| SFT + Final-Answer RL, Majority Voting | 12.1 (4.6-19.6) | 20.2 |
| Few-shot, ORM reranking | — | 27.8 |
| Few-shot + Final-Answer RL, ORM reranking | 12.4 (2.1-22.8) | 16.6 |
| SFT + Final-Answer RL, ORM reranking | 3.7 (0.5-6.9) | 14.2 |
| SFT, ORM reranking | 4.4 (0.6-8.3) | 14.8 |
| SFT, PRM reranking | 3.5 (0.5-6.5) | 14.1 |
| Few-shot + ORM-RL, ORM reranking | 5.5 (2.6-8.4) | 13.8 |
| **SFT + ORM-RL, ORM reranking** | **3.4 (0.0-6.8)** | **12.7** |
| SFT + PRM-RL, PRM reranking | 3.8 (0.5-7.1) | 12.9 |

### Key Comparisons

**Final-answer error — process vs outcome supervision comparable:**

| Setting | Process-based | Outcome-based |
|---|---|---|
| Without RM | SFT: 22.3% | Few-shot+FA-RL: 23.5% |
| With RM reranking | SFT+PRM: 14.1% | SFT+ORM: 14.8% |
| With RM+RL | SFT+PRM-RL: 12.9% | SFT+ORM-RL: 12.7% |

**Trace error — process-based wins decisively:**

| Setting | SFT (process-based) | Few-shot+FA-RL (outcome-based) |
|---|---|---|
| Majority voting | 11.4% | 19.8% |
| + ORM reranking | 4.4% | 12.4% |

### Selective Prediction (Abstaining on 30% of inputs)

| Model | Final-Answer Error (all) | Final-Answer Error (abstain 30%) |
|---|---|---|
| SFT + ORM-RL | 12.7% | 2.7% |
| SFT + PRM-RL | 12.9% | ~3% |

### Decoding Strategy Comparison (Table 3)

| Model | Greedy | Majority | RM-weighted |
|---|---|---|---|
| Few-shot | 54.3 | 41.4 | 27.8 |
| Few-shot + Final-Answer RL | 36.4 | 23.8 | 16.6 |
| Few-shot + ORM-RL | 31.5 | 18.8 | 13.8 |
| SFT | 41.1 | 22.3 | 14.1 |
| SFT + ORM-RL | 31.2 | 17.8 | 12.7 |
| SFT + PRM-RL | 34.4 | 17.6 | 12.9 |

ORM-RL and PRM-RL consistently beat Final-Answer RL across all decoding strategies.

## Key Takeaways

1. **For final-answer accuracy, outcome and process supervision perform comparably on GSM8K.** This was surprising — both with and without reward models, the gap is ≤1%. This finding was later overturned by Lightman et al. (2023) on the harder MATH dataset, where PRM wins by +5.8pp.

2. **For trace correctness, process supervision is essential.** Outcome-based RL (Final-Answer RL) produces high trace error (12.4-19.8%) even when final answers are correct. Process-based approaches (SFT or RM reranking) achieve 3.4-4.4% trace error. Models optimized for outcomes learn to produce correct answers via incorrect reasoning.

3. **ORMs implicitly learn step-level verification.** ORM predictions agree more with human step-level (PRM) labels (85%) than with the ORM's own training labels (77%). The ORM finds it easier to check step correctness than to internally compute final answers. This is a key insight — even cheap outcome labels can produce a model that approximates process feedback.

4. **RL against a reward model beats RL against final-answer correctness.** ORM-RL and PRM-RL outperform Final-Answer RL on both metrics and all decoding strategies. The reward model acts as a smoother, more informative training signal than binary answer checking.

5. **GSM8K may be too easy to separate ORM from PRM.** The paper's finding that ORM ≈ PRM on final-answer error is likely dataset-specific. On GSM8K, incorrect reasoning steps rarely lead to correct final answers (low spurious correlation), so outcome labels are relatively clean. On harder benchmarks (MATH), false positives are more common and PRM's advantage becomes clear.

6. **Selective prediction with RMs is highly effective.** Abstaining on 30% of inputs (based on RM confidence) reduces final-answer error from ~13% to ~2.7% — a 5× reduction. Models with lower trace error benefit more from selective prediction.

7. **SFT on human traces is a strong process-based baseline.** Plain SFT (no RL, no RM) already achieves 11.4% trace error and 22.3% final-answer error. Adding RM reranking brings trace error to 3.5-4.4%. The human reasoning traces provide strong process supervision even without explicit step labels.

## Connection to Our Work

This paper is the direct predecessor to Lightman et al. (2023). Its core finding — ORM ≈ PRM on GSM8K — motivated Lightman to test on harder problems (MATH), where PRM decisively wins. For our project:

- **The ORM-approximates-PRM finding is practically useful.** If we can't afford human step labels, training an ORM on final-answer correctness may still yield a decent step-level verifier. This could bootstrap our `<verify>` mechanism cheaply.
- **Trace error vs final-answer error is exactly our distinction.** We care about reasoning quality (trace), not just getting the right answer. This paper shows that without process supervision, models learn to hack the outcome — precisely why we need `<verify>`.
- **Expert iteration is a simpler alternative to GRPO.** Their RL approach (filter-and-distill) is much simpler than PPO/GRPO and still effective. Could be a fallback if GRPO is too unstable for our setup.
