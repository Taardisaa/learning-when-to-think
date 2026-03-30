# Let's Verify Step by Step

**Authors:** Hunter Lightman*, Vineet Kosaraju*, Yura Burda*, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe* (OpenAI)

**Date:** 2023 | **Venue:** arXiv:2305.20050

## Problem

LLMs performing multi-step math reasoning regularly produce hallucinations. A single logical error in a chain-of-thought (CoT) derails the entire solution. Two paradigms exist for training reward models to detect these errors:

| Supervision Type | Signal | Granularity | Label Source |
|---|---|---|---|
| Outcome supervision (ORM) | Final answer correct/incorrect | Solution-level | Automatic (answer checking) |
| Process supervision (PRM) | Each step correct/incorrect | Step-level | Human labelers |

Prior work (Uesato et al. 2022) found these comparable on GSM8K. This paper asks: **does process supervision outperform outcome supervision on harder math (MATH dataset), at scale, with sufficient data?**

## Key Concepts

### Outcome-Supervised Reward Model (ORM)
A reward model trained on binary labels: did the solution reach the correct final answer? At test time, the ORM's prediction at the **final token** serves as the solution score. Key weakness: false positives — solutions that reach the right answer via incorrect reasoning get positive labels, corrupting the training signal.

**Example:** A solution that makes two canceling errors (e.g., wrong sign twice) and arrives at the correct answer would receive a positive ORM label despite flawed reasoning.

### Process-Supervised Reward Model (PRM)
A reward model that receives a label (positive/negative/neutral) for **each step** in the solution. At test time, step-level scores are aggregated into a solution score. Key advantage: precise credit assignment — the PRM knows exactly which steps are correct and where the first error occurs.

**Example:** In a 6-step algebra solution where step 5 introduces an error, the PRM labels steps 1-4 as positive and step 5 as negative. The ORM only sees that the final answer is wrong.

### Convincing Wrong-Answer Solutions
Solutions rated highly by the current best PRM but that reach an incorrect final answer. These are the hardest cases for the reward model — they "look correct" step-by-step but aren't. The active learning strategy prioritizes labeling these to maximally improve the PRM.

**Example:** A solution that sets up equations correctly, solves them cleanly, but silently drops a constraint from the problem statement. Each step looks locally valid, but the overall approach is wrong.

### Best-of-N Evaluation
The protocol used to compare reward models. Given a test problem:
1. Sample N solutions from the generator
2. Score each with the reward model
3. Select the highest-scored solution
4. Check if its final answer is correct

A better reward model selects correct solutions more often. This decouples the reward model evaluation from the generator quality.

### PRM800K
The released dataset: ~800K step-level human labels across 75K solutions to 12K MATH problems. Collected in two phases — phase 1 (uniform sampling, 5% of labels) and phase 2 (active learning on convincing wrong-answer solutions, 95% of labels). Available at https://github.com/openai/prm800k.

## Core Design / Method

### Architecture and Training

Both ORMs and PRMs are finetuned from the base GPT-4 model (pre-RLHF). All models first undergo an additional pretraining stage on **MathMix**, a ~1.5B token dataset of math-relevant content.

The generator is trained to produce solutions in a **newline-delimited step-by-step format** (to make step parsing easy), by finetuning on few-shot generated correct solutions for one epoch.

### PRM Scoring: Solution-Level Aggregation

To compare multiple solutions, step-level scores must be reduced to a single solution score:

$$S(\text{solution}) = \prod_{i=1}^{K} P(\text{step}_i \text{ is correct})$$

| Term | Meaning |
|---|---|
| $S(\text{solution})$ | Overall solution score used for best-of-N ranking |
| $K$ | Number of steps in the solution |
| $P(\text{step}_i \text{ is correct})$ | PRM's predicted probability that step $i$ is positive |

**Concrete example:** A 3-step solution with step scores [0.95, 0.90, 0.80]:
- Product scoring: $0.95 \times 0.90 \times 0.80 = 0.684$
- Alternative (minimum): $\min(0.95, 0.90, 0.80) = 0.80$

The product performs slightly better (78.2% vs 77.6% on best-of-1860). The product penalizes longer solutions slightly, but this bias is minor.

### Neutral Label Handling

Each step gets one of three labels: **positive** (correct and progresses toward solution), **negative** (incorrect/unreasonable), **neutral** (ambiguous, technically valid but misleading). At scoring time, treating neutrals as positive performs best (78.2% vs 77.4%).

| Neutral Treatment | Product | Minimum |
|---|---|---|
| neutral = positive | **78.2%** | 77.6% |
| neutral = negative | 77.4% | 77.8% |

### Labeling Strategy: Supervise Up to First Error Only

For incorrect solutions, human labelers annotate steps only **up to and including the first negative step**, then stop. This design choice:
1. Makes the comparison with outcome supervision fairer (for correct solutions, both provide the same info; for incorrect, process adds only the error location)
2. Keeps labeling cost comparable
3. Still provides the key additional signal: *where* the error is

### Active Learning Strategy

The data collection uses an iterative active learning loop:

1. Train a small PRM_selector on one sample per problem
2. Generate N=1000 solutions per problem, score with PRM_selector
3. Select samples: 80% are the most convincing wrong-answer solutions, 20% are most convincing right-or-wrong solutions
4. Label selected samples with PRM_large (the large-scale PRM used as oracle)
5. Retrain PRM_selector and repeat

This yields ~**2.6x data efficiency** over uniform sampling (measured by comparing slopes of performance vs. data curves).

### Small-Scale Synthetic Supervision Setup

To conduct controlled ablations without expensive human labeling:
- Use the large-scale PRM (PRM_large) as a labeling oracle for small-scale models
- Train small-scale generators and reward models (~200x less pretraining compute than GPT-4)
- This allows apples-to-apples comparison: same training data, only supervision type differs

## Results

### Large-Scale: Best-of-1860 on MATH (500-problem test subset)

| Method | % Solved |
|---|---|
| Majority Voting | 69.6 |
| ORM (best-of-1860) | 72.4 |
| **PRM (best-of-1860)** | **78.2** |

The PRM outperforms the ORM by **5.8 percentage points**. The gap widens as N increases (Figure 3), indicating PRMs are better at searching over large solution sets.

### Small-Scale: Process vs Outcome (Controlled Comparison)

With identical training data (200 samples/problem, PRM_large as oracle for both):

| Supervision | Best-of-500 (%) |
|---|---|
| ORM (final-answer supervised) | ~38 |
| ORM (PRM_large supervised) | ~45 |
| **PRM (PRM_large supervised)** | **~56** |

Process supervision outperforms both forms of outcome supervision at all data scales and test-time compute budgets.

### OOD Generalization (Best-of-100, STEM tests)

| Domain | ORM | PRM | Majority Vote | # Problems |
|---|---|---|---|---|
| AP Calculus | 68.9% | **86.7%** | 80.0% | 45 |
| AP Chemistry | 68.9% | **80.0%** | 71.7% | 60 |
| AP Physics | 77.8% | **86.7%** | 82.2% | 45 |
| AMC10/12 | 49.1% | **53.2%** | 32.8% | 84 |
| **Aggregate** | 63.8% | **72.9%** | 61.3% | 234 |

PRM generalizes well to out-of-distribution STEM problems, outperforming both ORM and majority voting.

### Active Learning Impact

| Strategy | Data Efficiency |
|---|---|
| Uniform sampling | 1x (baseline) |
| Active learning (convincing wrong-answer) | **2.6x** |

### PRM800K Dataset Statistics

| Phase | % Solutions Ending Correct | % Individual Steps Correct |
|---|---|---|
| Phase 1 (uniform) | 85.1 | 58.6 |
| Phase 2 (active learning) | 13.2 | 74.1 |
| Combined | 14.2 | 73.1 |

Phase 2 is heavily skewed toward wrong-answer solutions (by design), but still contains many correct individual steps since errors typically occur late in solutions.

## Key Takeaways

1. **Process supervision decisively outperforms outcome supervision** on challenging math (MATH), unlike prior work on easier benchmarks (GSM8K). The gap is +5.8pp at large scale and grows with more test-time compute (higher N in best-of-N).

2. **The credit assignment problem explains ORM's weakness.** For hard problems, most generated solutions contain errors somewhere, so a negative outcome label has low marginal information — the ORM must implicitly figure out *where* the error is. PRMs get this for free.

3. **False positives poison ORM training.** Solutions reaching the correct answer via wrong reasoning get positive labels. Using PRM_large as an outcome supervisor (instead of final-answer checking) improves ORM performance, confirming that false positives are a real problem.

4. **Active learning on "convincing wrong-answer" solutions is 2.6x more data-efficient** than uniform sampling. The key insight: label the solutions that your current model is most wrong about — high PRM score but incorrect final answer.

5. **Process supervision has a "negative alignment tax"** — it simultaneously improves performance AND produces more interpretable, human-endorsed reasoning chains. This is rare; most alignment techniques trade off performance for safety.

6. **Labeling only up to the first error is sufficient.** The paper deliberately limits process labels to the first mistake, yet still sees large gains. This means the critical signal is error *location*, not exhaustive step-by-step annotation of everything after the error.

7. **The PRM generalizes OOD.** Strong results on AP exams and AMC competitions (never seen during training) show the PRM learns general step-correctness patterns, not just MATH-specific heuristics.

8. **Solution scoring: product of step probabilities > minimum.** Taking the product over step-level correctness probabilities slightly outperforms taking the minimum, despite introducing a mild bias against longer solutions.

## Connection to Our Work

The PRM's core contribution — localizing errors at the step level — is closely related to our backoff token approach. Where the PRM trains a separate verifier to detect the first error, our `<backoff_N>` tokens train the generator itself to recognize errors mid-reasoning and self-correct. The PRM800K dataset's annotation scheme (label each step, stop at first error) maps directly onto our need to identify error points for backoff insertion. The active learning strategy of prioritizing "convincing wrong-answer" solutions parallels our focus on wrong rollouts as the source of backoff training examples.
