# S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning

**Authors:** Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, et al. (Tencent, Tsinghua, HKU, Fudan)
**Date:** 2025-02-18 | **arXiv:** 2502.12853v1

## Problem

Existing approaches to incentivize deep thinking in LLMs require either large-scale
data, massive compute, or distillation from stronger models. S²R asks: can we teach
*smaller* models (7–8B) to think deeply by learning two specific skills — **self-verification**
and **self-correction** — with minimal data (~3k SFT samples)?

---

## Core Design: Interleaved Solve-Verify-Correct Loop

The model generates a sequence of **typed actions**:

$$y = (s_1, v_1, s_2, v_2, \ldots, s_k, v_k, \texttt{<end>})$$

where:
- $s_j$ = **solve** action (attempt to answer the problem)
- $v_j$ = **verify** action (self-assess correctness of $s_j$)
- **<end>** = terminate

The action type at position $i+1$ is determined by a fixed protocol:

$$Type(a_{i+1}) = \begin{cases} \texttt{verify}, & \text{if } Type(a_i) = \texttt{solve} \\ \texttt{solve}, & \text{if } Type(a_i) = \texttt{verify} \wedge Parser(a_i) = \texttt{INCORRECT} \\ \texttt{<end>}, & \text{if } Type(a_i) = \texttt{verify} \wedge Parser(a_i) = \texttt{CORRECT} \end{cases}$$

**Concrete example** (math problem, 2 trials):

```
[solve_1]  "Let x = ... → answer is 42"
[verify_1] "Wait, substituting back: 2(42) ≠ 100. Therefore, the answer is incorrect."
[solve_2]  "Let me redo: x = 50. Check: 2(50) = 100 ✓"
[verify_2] "Substituting: 2(50) = 100. Therefore, the answer is correct."
<end>
```

The model dynamically allocates test-time compute: easy problems get 1 trial,
hard problems get up to 4 trials.

---

## Two-Stage Training Pipeline

### Stage 1: Behavior Initialization (SFT)

#### Step 1a: Construct self-verification data

Two verification styles explored:

| Style | How it works | Accuracy | Bias |
|---|---|---|---|
| **Problem-solving** | Re-solve the problem, compare answers | Higher overall (77–80%) | Biased toward "correct" (predicts correct even when wrong) |
| **Confirmative** | Check the answer from a *different perspective* without re-solving | Lower overall (59–66%) | More balanced between correct/incorrect |

They use **confirmative verification** because it's less biased — problem-solving
verification is too influenced by the preceding solution (the model struggles to
"think outside the box" it just created).

**Verification construction pipeline:**
1. Prompt frontier LLMs to "verify the answer's correctness without re-solving"
2. Filter for valid verifications using LLM-as-judge
3. Refine with GPT-4-preview-1106 to read naturally as self-checks
4. Append conclusion: "Therefore, the answer is correct/incorrect/cannot verify"
5. Discard verifications that don't match ground truth

#### Step 1b: Construct self-correction data

Inspired by SCoRe (Kumar et al., 2024): concatenate incorrect attempts (each
followed by a verification flagging the error) with a final correct solution.

#### Step 1c: Build dynamic trial-and-error trajectories

Three principles for trajectory construction:
1. **Diverse lengths:** $k \in \{1, 2, 3, 4\}$ trials per trajectory
2. **Own errors:** incorrect attempts sampled from the base model itself
3. **Difficulty-aligned:** harder problems get more trials (based on accuracy of
   sampled responses)

**Difficulty bucketing:**

| Difficulty | Base model accuracy | # Trials in trajectory |
|---|---|---|
| Level 1 (easy) | High | $k = 1$ (solve once, verify correct, done) |
| Level 2 | Medium-high | $k = 2$ |
| Level 3 | Medium-low | $k = 3$ |
| Level 4 (hard) | Low | $k = 4$ |

#### Masked SFT objective

$$\mathcal{L} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{SFT}} \sum_{a_t \in y} \delta_{mask}(a_t) \log \pi(a_t \mid x, y_{:a_t})$$

where the mask is:

$$\delta_{mask}(a_t) = \begin{cases} 1, & \text{if } Type(a_t) = \texttt{verify} \\ 1, & \text{if } Type(a_t) = \texttt{solve} \text{ and } t = T-1 \text{ (last solve)} \\ 1, & \text{if } Type(a_t) = \texttt{<end>} \text{ and } t = T \\ 0, & \text{otherwise} \end{cases}$$

**Key insight:** Only train on all verifications + the final correct solution. Incorrect
intermediate solutions are *not trained on* — they're present as context only. This
prevents the model from learning to produce wrong answers.

**Data scale:** Only ~3.1k–4.6k SFT samples per base model (from MATH train set).

---

### Stage 2: Reinforcement Learning

Two RL algorithms explored, both using rule-based rewards only (no learned reward model):

#### Outcome-level RLOO (Reinforcement Learning with Leave-One-Out)

Reward based on correctness of the **final** solution $s_T$:

$$R_o(x, y) = \begin{cases} 1, & V_{golden}(s_T) = \texttt{correct} \\ -1, & \text{otherwise} \end{cases}$$

Advantage with RLOO baseline and KL penalty:

$$A(x, y) = R_o(x, y) - \hat{b} - \beta \log \frac{\pi_{\theta_{old}}(y|x)}{\pi_{ref}(y|x)}$$

where $\hat{b} = \frac{1}{M-1} \sum_{j \neq m} R_o(x, y^{(j)})$ is the leave-one-out baseline.

Policy update (clipped surrogate, applied to whole trajectory):

$$\mathcal{L}(\theta) = -\mathbb{E}_{\substack{x \sim \mathcal{D} \\ y \sim \pi_{\theta_{old}}(\cdot|x)}} \left[\min\!\Big(r(\theta) A(x,y),\; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A(x,y)\Big)\right]$$

**What outcome-level RL does:** lets the model freely explore different
trial-and-error paths — it only cares about the final answer. Good for stronger
base models that already have some reasoning ability.

#### Process-level Group-based RL

Each action $a_t$ gets its own reward:

$$R_a(s_j \mid x, y_{:s_j}) = \begin{cases} 1, & V_{golden}(s_j) = \texttt{correct} \\ -1, & \text{otherwise} \end{cases}$$

$$R_a(v_j \mid x, y_{:v_j}) = \begin{cases} 1, & Parser(v_j) = V_{golden}(s_j) \\ -1, & \text{otherwise} \end{cases}$$

**Key:** verify actions are rewarded for *accuracy of the verification itself*
(did it correctly identify whether the solution was right or wrong?), not just for
the final outcome.

**Group-based baseline estimation:** actions sharing the same **reward context**
(same sequence of previous rewards) are grouped together, and their average
reward serves as baseline:

$$\hat{b}(a_t \mid x, y) = \frac{1}{|\mathcal{G}(\mathbf{R}(a_t|x,y))|} \sum_{a \in \mathcal{G}(\mathbf{R}(a_t|x,y))} R_a(a | x^{(a)}, y^{(a)}_a)$$

where $\mathbf{R}(a_t \mid x, y) = (R_a(a_i \mid x, y_{:a_i}))_{i=1}^{t-1}$ is the reward history.

**Intuition:** All actions that have seen "one failed attempt + one correct
verification" share a baseline, regardless of which specific problem they came
from. This gives a natural per-step normalization.

Per-action advantage:

$$A(a_t \mid x, y) = R_a(a_t \mid x, y_{:a_t}) - \hat{b}(a_t \mid x, y) - \beta \log \frac{\pi_{\theta_{old}}(a_t \mid x, y))}{\pi_{ref}(a_t \mid x, y)}$$

**What process-level RL does:** directly supervises intermediate verification
accuracy. Better for weaker base models that need guidance at each step.

---

## Outcome-level vs Process-level: When to Use Which

| Property | Outcome-level (RLOO) | Process-level (Group RL) |
|---|---|---|
| Reward signal | Final answer only | Each solve + verify step |
| Exploration freedom | High — any path to correct answer | Constrained — intermediate steps supervised |
| Best for | Stronger base models (Qwen2.5-Math-7B) | Weaker base models (Qwen2-7B-Instruct) |
| Online vs Offline | Better online | Better offline (more stable baselines) |
| Verification accuracy | Improves Error Recall + Correct Precision | Improves raw Verification Accuracy more |

---

## Offline RL Variant

Key innovations for making offline RL competitive:

1. **Accuracy-grouped baselines:** group trajectories by problem difficulty
   (measured by sampling accuracy), then compute baselines within each group.
   Harder problems naturally have lower expected returns.

2. **Position-group refinement:** within each accuracy bin, further group by
   step index (actions at step 3 vs step 5 have different expected returns).

3. **Rejection sampling:** discard malformed trajectories (e.g., $s_1, s_2, v_1$)
   and trajectories with >20 actions.

4. **Prompt filtering:** retain only prompts with accuracy in $[0.1, 0.7]$ —
   moderately difficult problems give the best training signal.

Result: offline RL achieves **comparable performance to online RL** on most
benchmarks, with process-level slightly outperforming outcome-level (reversed
from online setting).

---

## Main Results

### MATH500 (Qwen2.5-Math-7B base)

| Method | MATH500 | AIME 2024 | AMC 2023 | GSM8K | Average (7 benchmarks) |
|---|---|---|---|---|---|
| Qwen2.5-Math-7B (base) | 51.0 | 16.7 | 45.0 | 58.3 | 35.6 |
| Qwen2.5-Math-7B-Instruct | 83.2 | 13.3 | 72.5 | **95.6** | 59.9 |
| Eurus-2-7B-PRIME | 79.2 | **26.7** | 57.8 | 88.0 | 56.6 |
| rStar-Math-7B | 78.4 | **26.7** | 47.5 | 89.7 | 58.2 |
| Qwen2.5-7B-SimpleRL | 82.4 | **26.7** | 62.5 | — | — |
| **S²R-BI (SFT only)** | 81.6 | 23.3 | 60.0 | 91.9 | 59.3 |
| **S²R-PRL** | 83.4 | **26.7** | 70.0 | 93.2 | **62.0** |
| **S²R-ORL** | **84.4** | 23.3 | **77.5** | 92.9 | **62.4** |

**51.0% → 84.4%** on MATH500 with only 3.1k SFT samples + 10k RL samples.
Outperforms long-CoT distillation from QwQ-32B-Preview (80.2% with same data scale).

### Cross-domain generalization (Qwen2.5-Math-7B → non-math tasks)

| Task | Base | S²R-BI | S²R-ORL |
|---|---|---|---|
| FOLIO (logic) | 57.9 | 58.1 | **61.6** |
| CRUXEval (code) | 40.8 | 48.0 | **50.9** |
| StrategyQA (multi-hop) | 61.1 | **88.7** | **90.8** |
| MMLUPro-STEM | 46.0 | 49.8 | **50.0** |

Self-verify + self-correct transfers as a general reasoning skill, not just math.

---

## Adaptive Test-Time Compute Allocation

After S²R training, models naturally learn to allocate effort by difficulty:

| Difficulty | SFT avg trials | SFT+RL avg trials | SFT accuracy | SFT+RL accuracy |
|---|---|---|---|---|
| Level 1 (easy) | ~1.2 | ~1.1 | ~95% | ~97% |
| Level 3 (medium) | ~2.0 | ~2.2 | ~75% | ~85% |
| Level 5 (hard) | ~2.5 | ~3.0 | ~40% | ~55% |

RL particularly improves accuracy on hard problems while maintaining efficiency on
easy ones.

---

## Key Architectural Decisions

### Why confirmative verification over problem-solving verification?

Problem-solving verification (re-solve and compare) has higher raw accuracy but is
**biased toward predicting "correct"** — the model re-derives the same answer and
confirms it. Confirmative verification (check from a different angle without re-solving)
is more balanced:

| Base Model | Problem-solving (correct/incorrect) | Confirmative (correct/incorrect) |
|---|---|---|
| Llama-3.1-8B | 87.3% / 66.9% | 77.3% / 78.2% |
| Qwen2-7B | 90.2% / 67.4% | 76.2% / 70.1% |
| Qwen2.5-Math-7B | 91.2% / 56.7% | 82.8% / 68.0% |

Problem-solving verification gets 91% accuracy on correct answers but only 57% on
incorrect ones (i.e., it almost never catches errors). Confirmative verification is more
balanced at 83%/68%.

### Masked SFT: why not train on incorrect solutions?

Training on all tokens including wrong intermediate solutions teaches the model to
*produce* wrong answers. Masking out incorrect solutions (only training on verifications
+ final correct solution) means the wrong attempts serve as *context* for learning
verification, not as targets.

---

## Hyperparameters

### SFT (Stage 1)

| Parameter | Value |
|---|---|
| Learning rate | 5e-6 |
| Batch size | 32 |
| Max sequence length | 6000–8000 |
| KL coefficient | 0.01–0.1 |
| Epochs | 3 |

### RL (Stage 2)

| Parameter | Value |
|---|---|
| Learning rate | 5e-7 |
| Training batch size | 64 |
| Forward batch size | 256 |
| KL coefficient | 0.01–0.05 |
| Sampling temperature | 0.7 |
| Clip range | 0.2 |
| Training steps | 500 |
| Samples per question (RLOO) | 4 |

---

## Key Takeaways for Our Project

1. **Extremely data-efficient.** 3.1k SFT samples achieves +30% on MATH500,
   outperforming methods trained on orders of magnitude more data. The key is
   *what* you train on (verify + correct behavior) not how much.

2. **Interleaved solve-verify-correct is a natural structure** for adaptive
   test-time compute. The model decides when to stop (verify → correct → end)
   rather than being forced into a fixed number of turns (unlike SCoRe's rigid
   2-turn protocol).

3. **Confirmative verification > problem-solving verification.** Re-solving
   the same problem is biased toward confirmation. Checking from a different
   angle (reverse-thinking, substitution) catches more actual errors.

4. **Process-level RL helps weak models; outcome-level RL helps strong models.**
   Weak models need intermediate guidance; strong models benefit from exploration
   freedom.

5. **Connection to our backoff work:** S²R's solve-verify-correct loop is
   structurally similar to our backoff token mechanism. Their "verify → INCORRECT
   → solve again" maps to our "generate → backoff → regenerate." The key
   difference: S²R uses natural-language verification as an explicit intermediate
   step, while our backoff tokens are a more compressed signal. S²R's
   difficulty-adaptive trajectory length (1–4 trials) is analogous to our model
   learning when to invoke backoff tokens.
