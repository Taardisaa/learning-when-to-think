# GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements

**Authors**: Alex Havrilla, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Roberta Raileanu (FAIR at Meta, Georgia Tech, StabilityAI)
**Date**: June 2024
**Venue**: arXiv 2402.10963v2

## Core Problem

LLMs can sometimes refine their reasoning, but struggle to identify **when** a draft needs refinement and **where** the first error is, without external feedback. The paper decomposes refinement into three sub-problems:

1. **When** to refine — is the draft correct or not?
2. **Where** to refine — which step has the first error?
3. **How** to refine — rewrite globally (from scratch) or locally (from the error point)?

## Key Concepts

### ORM (Outcome Reward Model)

An ORM answers: **"given this complete solution, is the final answer correct?"**

It's trained on (question, full_solution, correct?) triples sampled from a student model π. At test time, you can use it to rerank multiple candidate solutions — pick the one the ORM scores highest.

**The problem**: people tried using ORMs at intermediate steps too — "given this partial solution up to step S_i, will the final answer be correct?" This doesn't work well. The ORM learned the student's failure patterns, not step correctness. If the student always fails at division, the ORM assigns low scores to any step involving division, even if the step is perfectly correct. The paper calls this **overly pessimistic** — the ORM hallucinates errors.

### PRM (Process Reward Model)

A PRM answers: **"is this specific step S_i correct, given the preceding steps?"**

Each step is independently judged. This is what you actually want for error localization. But PRMs need **human annotations per step** (Lightman et al., 2023), which is expensive. The paper wants to avoid human labels entirely.

### SORM (Stepwise ORM) — the paper's contribution

A SORM answers: **"is the correct final answer still reachable from prefix P_i?"**

This is different from both ORM and PRM:
- ORM asks about the student π's likelihood of success → biased by π's weaknesses
- PRM asks if step S_i itself is correct → needs human labels
- SORM asks if the **optimal policy π\*** could succeed from here → unbiased, synthetic labels

How to generate SORM labels without humans:
1. Take a student-generated solution with steps S1, S2, ..., SL
2. At each prefix P_i = (S1, ..., Si), sample K=8 fresh continuations
3. If **any** continuation reaches the correct answer → label P_i as **positive** (recoverable)
4. If **none** do → label P_i as **negative** (stuck on a wrong path)

The first step where the label flips from positive to negative is the **first error**.

**Why this works**: rejection sampling with K attempts approximates whether the optimal policy could solve it from here. If 8 random attempts all fail from prefix P_i, the prefix is probably broken. This gives V* (optimal value function) labels without human annotation.

**Limitation**: SORM is slightly worse than ORM at predicting final-answer correctness (it's trying to be unbiased, which costs some discriminative power). So the paper uses ORM for reranking and SORM for error localization — each where it's strongest.

### SORM Training (the novel contribution)

For each step S_i in a student-generated solution:
1. Sample K=8 continuations from the student starting at prefix P_i
2. Check if any continuation reaches the correct final answer
3. Label S_i as **positive** if yes (correct answer is reachable), **negative** if no

Post-processing for quality:
- If S_i is positive, mark all S_j (j < i) as positive too (monotonicity)
- Enforce consistency: verification rollout must use intermediate results from the prefix
- Balance positive/negative labels at each prefix length

### Two Types of Refinement

**Global refinement** — "throw it all away and start over"

The global refiner sees the question Q and a draft A_D, and writes a completely new solution A_GR from scratch. It doesn't know where the error is. It just sees a bad draft and tries again with a different approach.

Training data: pair an incorrect rollout with a correct rollout for the same question. The model learns to "translate" bad solutions into good ones.

When it works well: when the entire approach is wrong (e.g., student tried addition when it should have used multiplication). Starting over lets it pick a new strategy.

When it fails: when only a small arithmetic error needs fixing. Starting over is wasteful and the new attempt may introduce new errors.

**Local refinement** — "fix it from where it broke"

The local refiner sees Q, the draft A_D, and an error location E (marked with a `[BAD]` token). It keeps everything before `[BAD]` and rewrites from there.

Training data: take an incorrect rollout, use SORM to find the first error at step i, insert `[BAD]` before step i. Pair with a correct continuation from step i-1 (obtained via rejection sampling from P_{i-1}).

When it works well: when the prefix is good but a specific calculation went wrong. Just fix that step and continue.

When it fails: when the correct prefix leads into a dead end. For example, if the student's approach requires division and the student can't divide, the prefix is "correct" but low-value — local refinement is stuck continuing a bad strategy.

```
Example (from Figure 2):

Q: If Tom eats two muffins a day three times a week and buys 12 packs
   of 8 muffins, how many weeks can he go before buying more muffins?

Draft A_D:
  If Tom eats two muffins three times a week
  he eats 2*3 = 6 muffins a week.
  This means he eats 6 / 8 = 0.75 packs of muffins a week.
  [BAD] Since he has 12 packs, he can go 12 * 0.75 = 9 weeks
  before buying more muffins.

Local Refinement A_LR:
  Every day when Tom eats muffins                    ← rewrites from [BAD]
  he eats 2 / 8 = 0.25 packs of muffins.
  Since Tom has 12 packs he has enough
  muffins for 12 / 0.25 = 48 days.
  Since Tom eats muffins three days a week,
  he has enough muffins for 48 / 3 = 16 weeks.       ✓

Global Refinement A_GR:
  If Tom eats two muffins three times a week          ← restarts entirely
  he eats 2*3 = 6 muffins a week.
  This means he eats 6 / 8 = 0.75 packs
  of muffins a week.
  Since he has 12 packs, he can go 12 / 0.75 = 18.33
  weeks before buying more muffins.                   ✗
```

**Critical finding**: global and local refinements solve **complementary, partially disjoint** problem sets. Local refinements can fix simple calculation errors efficiently. Global refinements can completely change the solution approach (e.g., avoid division entirely if the student always fails at it). Combined with ORM reranking, they fix **41% of incorrect GSM8K drafts** from a 13B student.

## Three-Stage Training Pipeline

```
Stage 1: Student fine-tuning
  Base model → SFT → Expert Iteration (sample K=96, filter correct) → Student π

Stage 2: ORM/SORM training
  Student π generates (Q, A, correct?) data → train ORM
  At each prefix P_i, rejection-sample K=8 continuations → SORM labels → train SORM

Stage 3: Refinement training
  Global: pair incorrect rollouts with correct rollouts → train global refiner
  Local:  pair incorrect prefix P_{i+1} with correct verification T+ from P_i
          → insert [BAD] at step i+1 → train local refiner
```

## Key Results (GSM8K, Llama-2 13B)

| Setting | Accuracy |
|---------|----------|
| Greedy baseline (EI student) | 53% |
| Best-of-3 sampling | ~58% |
| Global refinement only (on incorrect drafts) | 28.1% fix rate |
| Local refinement + SORM (on incorrect drafts) | 28.3% fix rate |
| **Global + Local + ORM reranking** | **65%** (+12% over greedy) |

- Refiners struggle with **when** to refine — they sometimes make correct drafts worse
- When given only incorrect drafts, both global and local refinements fix ~28% of errors
- Combined with ORM reranking (choosing best of draft, global, local), accuracy jumps to 65%
- SORM gives 8% better step accuracy than ORM (73→81%) but slightly worse final-answer accuracy

## Worked Example: Full Pipeline on One Problem

**Question**: "Betty has $100. She buys a wallet for $40 and a belt for $20. How much money does she have left?"
**Gold answer**: 40

### Stage 1: Student generates draft

```
Draft A_D (incorrect):
  S1: Betty starts with $100.
  S2: She spends $40 on a wallet.
  S3: She spends $20 on a belt.
  S4: Total spent is 40 + 20 = 50.
  S5: She has 100 - 50 = 60 left.        ← wrong (should be 40? no wait, 50 is correct)
```

Actually let me use a real error pattern. Suppose the student produces:

```
Draft A_D (incorrect):
  S1: Betty starts with $100.
  S2: She spends $40 on a wallet, leaving 100 - 40 = 50.     ← ARITHMETIC ERROR (should be 60)
  S3: She spends $20 on a belt, leaving 50 - 20 = 30.        ← cascading error
  #### 30                                                      ← wrong (gold = 40)
```

### Stage 2: ORM and SORM evaluate each step

**ORM scores** (predicts P(correct final answer | prefix)):
```
  ORM(Q, P1="Betty starts with $100")         = 0.85  (high — early step, most continuations work)
  ORM(Q, P2="...leaving 100 - 40 = 50")       = 0.15  (low — ORM detects this path likely fails)
  ORM(Q, P3="...leaving 50 - 20 = 30")        = 0.05  (very low — wrong answer incoming)
```

**SORM scores** (rejection-samples K=8 continuations from each prefix):
```
  SORM at P1: 6/8 continuations reach #### 40  → label = positive (0.75)
  SORM at P2: 0/8 continuations reach #### 40  → label = negative (0.00)
              (because all continuations start from the wrong "50", none recover)
  SORM at P3: 0/8 reach #### 40                → label = negative (0.00)
```

SORM identifies S2 as the **first error** (first step where label flips from positive to negative).

### Stage 3a: Global refinement

Global refiner sees (Q, A_D) and rewrites from scratch:

```
Global A_GR:
  Betty starts with $100.
  She spends 40 + 20 = $60 total.
  She has 100 - 60 = $40 left.
  #### 40                                      ✓
```

### Stage 3b: Local refinement

Local refiner sees (Q, A_D, E=2) with [BAD] marking step 2:

```
Input to local refiner:
  Betty starts with $100.
  [BAD] She spends $40 on a wallet, leaving 100 - 40 = 50.
  She spends $20 on a belt, leaving 50 - 20 = 30.
  #### 30

Local A_LR (rewrites from [BAD]):
  She spends $40 on a wallet, leaving 100 - 40 = 60.    ← fixed arithmetic
  She spends $20 on a belt, leaving 60 - 20 = 40.       ← cascading fix
  #### 40                                                 ✓
```

### Stage 4: ORM reranks candidates

```
  ORM(Q, A_D)   = 0.05    (original draft — wrong)
  ORM(Q, A_GR)  = 0.92    (global refinement — correct)
  ORM(Q, A_LR)  = 0.95    (local refinement — correct, cleaner)
  → Select A_LR                                          ✓
```

### How this maps to our backoff approach

```
Our approach (single model, online):
  <think>
  Betty starts with $100. <continue>
  She spends $40 on a wallet, leaving 100 - 40 = 50. <backoff> <depth_1> 50 is wrong, should be 60 <continue>
  She spends $40 on a wallet, leaving 100 - 40 = 60. <continue>
  She spends $20 on a belt, leaving 60 - 20 = 40. </think>
  #### 40                                              ✓

Key difference: no separate verifier, no post-hoc refinement, no reranking.
The model detects, corrects, and continues in one generation pass.
The wrong tokens (S2, S3) are deleted from the KV cache — context reclaimed.
```

## Relevance to Our Project

**Similarities**:
- Both decompose reasoning correction into detection + localization + correction
- SORM's "is the correct answer still reachable from here?" is conceptually similar to our meta-prover progress signal
- Local refinement with [BAD] token ≈ our `<backoff>` + directive mechanism

**Key differences**:
- GLoRe uses a **separate verifier model** (ORM/SORM) to decide when/where to refine. We learn this as an **integrated policy action** — the same model decides to backoff
- GLoRe's refinement is **post-hoc** (generate draft → refine). Ours is **online** (backoff during generation, reclaiming KV cache)
- GLoRe's local refinement keeps the wrong prefix visible (just marks it with [BAD]). Our approach **deletes** the wrong tokens to reclaim context space, preserving information only through the directive
- GLoRe requires 3 separate models (student + ORM/SORM + refiner). We use a single model for everything

**Insights we can use**:
- The complementarity of global vs local refinement suggests our depth tokens (depth_1 vs depth_3) serve a similar role — shallow backoff is "local" correction, deep backoff is closer to "global" restart
- SORM's rejection-sampling approach to generate step-level labels is directly applicable to our progress reward computation
- The finding that refiners struggle with "when to refine" reinforces that learning this as a policy action (via GRPO reward signal) rather than a separate classifier might be more robust
