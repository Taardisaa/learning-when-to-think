# SCoRe: Training Language Models to Self-Correct via Reinforcement Learning

**Authors:** Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, JD Co-Reyes, et al. (Google DeepMind)
**Date:** 2024-10-07 | **arXiv:** 2409.12917v2

## Problem

Intrinsic self-correction (revising your own answer with no external feedback) is
consistently **negative** in base LLMs: Gemini 1.5 Flash scores 52.6% on attempt 1 but
drops to 41.4% on attempt 2 ($\Delta = -11.2\%$). Prior training approaches fail for two
diagnosed reasons:

| Failure mode | What happens | Why |
|---|---|---|
| **Distribution shift** | SFT learns to fix the *base model's* mistakes, but at test time the *fine-tuned* model makes different mistakes | Off-policy correction traces don't generalise |
| **Behavior collapse** | Model learns to produce a great first attempt and then copy it verbatim on the second turn | Maximising likelihood on correction traces rewards the "don't change anything" strategy equally |

---

## Core Design: Two-Stage Online RL

### Formulation

Given dataset $\mathcal{D} = \{(x_i, y_i^*)\}_{i=1}^{N}$ and binary reward
$\hat{r}(y, y^*)$ (answer-match checker), the model produces two attempts
$\hat{y}_1, \hat{y}_2$ for each problem $x$. The objective optimises the **sum of
rewards across both turns**:

$$\max_{\pi_\theta} \; \mathbb{E}_{x,y^* \sim \mathcal{D},\; \hat{y}_{l+1} \sim \pi_\theta(\cdot|[x, \hat{y}_{1:l}, p_{1:l}])} \left[\sum_{i=1}^{l+1} \hat{r}(\hat{y}_i, y^*)\right] \tag{1}$$

with $l = 1$ (two turns). Unlike standard RL that trains $\pi$ to directly produce $y^*$,
this trains over **multiple sequential attempts simultaneously**.

Base RL objective (REINFORCE with KL penalty):

$$\max_\theta \; \mathbb{E}_{x_i, y_i \sim \pi_\theta(\cdot|x_i)} \bigl[\hat{r}(y_i, y^*) - \beta_1 D_{KL}(\pi_\theta(\cdot|x_i) \| \pi_{\text{ref}}(\cdot|x_i))\bigr] \tag{2}$$

---

### Stage I — Initialization that Decouples Attempts

**Goal:** Break the base model's bias toward coupling turn 1 and turn 2 (which causes
collapse). Train the model to improve on turn 2 *while freezing turn 1 near the base
model*.

$$\max_\theta \; \mathbb{E}_{\substack{y_1 \sim \pi_\theta(\cdot|x),\\ y_2 \sim \pi_\theta(\cdot|[x_1, p_1])}} \bigl[\hat{r}(y_2, y^*) - \beta_2 D_{KL}(\pi_\theta(\cdot||x_1) \| \pi_{\text{ref}}(\cdot|x_1))\bigr] \tag{3}$$

- $\beta_2$ is **large** — it enforces a strict KL penalty *only on the first attempt*,
  keeping turn-1 distribution close to the base model
- Turn-2 is optimised freely for reward
- This prevents the model from "solving" the problem by just getting turn 1 right
  and copying

**Concrete example of what Stage I prevents:**
Without Stage I, multi-turn RL quickly converges to a policy where turn 2 proposes a
*different* answer only ~5% of the time (Figure 6b). With Stage I, this rate stays at
~35%, preserving exploration.

---

### Stage II — Multi-Turn RL with Reward Shaping

**Goal:** Jointly optimise both turns, but add a bonus that makes self-correction
*more attractive* than the degenerate "be right on turn 1, copy on turn 2" strategy.

$$\max_\theta \; \mathbb{E}_{\substack{y_1 \sim \pi_\theta(\cdot|x),\\ y_2 \sim \pi_\theta(\cdot|[x_1, p_1])}} \left[\sum_{i=1}^{2} \hat{r}(y_i, y^*) - \beta_1 D_{KL}(\pi_\theta(\cdot|x_i) \| \pi_{\text{ref}}(\cdot|x_i))\right] \tag{4}$$

**Reward shaping bonus** added to the turn-2 reward:

$$\hat{b}(y_2 | y_1, y^*) := \alpha \cdot \bigl(\hat{r}(y_2, y^*) - \hat{r}(y_1, y^*)\bigr)$$

where $\alpha > 1$ (they use $\alpha = 10$). The shaped turn-2 reward becomes:

$$r_2^{\text{shaped}} = \hat{r}(y_2, y^*) + \alpha \cdot \bigl(\hat{r}(y_2, y^*) - \hat{r}(y_1, y^*)\bigr)$$

**What this does concretely (binary rewards $\in \{0, 1\}$):**

| Turn 1 | Turn 2 | $\hat{b}$ | Shaped $r_2$ | Interpretation |
|--------|--------|-----------|---------------|----------------|
| 0 (wrong) | 1 (correct) | $+\alpha$ | $1 + \alpha = 11$ | **Self-correction rewarded heavily** |
| 1 (correct) | 1 (correct) | $0$ | $1$ | Maintaining correctness: neutral |
| 1 (correct) | 0 (wrong) | $-\alpha$ | $-\alpha = -10$ | **Regression punished heavily** |
| 0 (wrong) | 0 (wrong) | $0$ | $0$ | No change: neutral |

The bonus makes "incorrect→correct" transitions worth $11\times$ a normal correct answer,
while "correct→incorrect" transitions incur a $-10$ penalty. This asymmetry steers the
policy toward the self-correction strategy rather than the "get it right first time" shortcut.

---

## Why SFT Fails: Concrete Diagnosis

### STaR (Zelikman et al., 2022)
- Filter base-model rollouts for successful corrections, SFT on those traces
- Result on MATH: Accuracy@t1 rises to 55.4% but $\Delta(\text{t1,t2}) = -14.2\%$
- **Edit distance analysis:** SFT models spike at edit-distance-ratio $\approx 0$,
  meaning they learn to *not change their answer at all* (behavior collapse)

### Pair-SFT (Welleck et al., 2023)
- Synthetically pair incorrect turn-1 with correct turn-2 from base model
- Result: $\Delta(\text{t1,t2}) = 1.8\%$ (marginally positive)
- **Distribution shift:** correction accuracy on *fixed* (off-policy) first attempts
  improves during training, but on *self-generated* first attempts it degrades
  (Figure 5)

### Adding correct→correct data ($\mathcal{D}^+$)
- Including "don't change correct answers" examples into STaR reduces
  $\Delta^{c \to i}$ (correct→incorrect) from 19.6% to 2.2%, but $\Delta(\text{t1,t2})$
  only reaches 0.4%

**Takeaway:** SFT can partially fix one failure mode but not both simultaneously.
On-policy RL is necessary to address distribution shift, and reward shaping is necessary
to avoid behavior collapse.

---

## Metrics Defined

- **Accuracy@t1**: correctness of first attempt
- **Accuracy@t2**: correctness of second attempt
- **$\Delta$(t1,t2)**: $= \text{Acc@t2} - \text{Acc@t1}$ — net self-correction gain
- **$\Delta^{i \to c}$(t1,t2)**: fraction of problems wrong at t1, correct at t2 (problems *fixed*)
- **$\Delta^{c \to i}$(t1,t2)**: fraction of problems correct at t1, wrong at t2 (problems *broken*)

---

## Main Results

### MATH (Gemini 1.5 Flash)

| Method | Acc@t1 | Acc@t2 | $\Delta$ | $\Delta^{i \to c}$ | $\Delta^{c \to i}$ |
|--------|--------|--------|----------|---------------------|---------------------|
| Base model | 52.6% | 41.4% | -11.2% | 4.6% | 15.8% |
| Self-Refine | 52.8% | 51.8% | -1.0% | 3.2% | 4.2% |
| STaR $\mathcal{D}^+_{\text{STaR}}$ | 53.6% | 54.0% | +0.4% | 2.6% | 2.2% |
| Pair-SFT $\mathcal{D}^+_{\text{SFT}}$ | 52.4% | 54.2% | +1.8% | 5.4% | 3.6% |
| **SCoRe** | **60.0%** | **64.4%** | **+4.4%** | **5.8%** | **1.4%** |

SCoRe is the first method to achieve *significantly positive* intrinsic self-correction.

### HumanEval (Gemini 1.0 Pro, code generation)

| Method | MBPP-R | Acc@t1 | Acc@t2 | $\Delta$ | $\Delta^{i \to c}$ | $\Delta^{c \to i}$ |
|--------|--------|--------|--------|----------|---------------------|---------------------|
| Base model | 47.3% | 53.7% | 56.7% | +3.0% | 7.9% | 4.9% |
| **SCoRe** | **60.6%** | 52.4% | **64.6%** | **+12.2%** | **15.2%** | **3.0%** |

---

## Ablations (MATH)

| Variant | Acc@t1 | Acc@t2 | $\Delta$ |
|---------|--------|--------|----------|
| **SCoRe (full)** | 60.0% | 64.4% | +4.4% |
| w/o multi-turn (single-turn RL) | **61.8%** | 59.4% | -2.4% |
| w/o Stage I | 59.2% | 61.4% | +2.2% |
| w/o reward shaping | 60.0% | 62.6% | +2.6% |
| w/ STaR replacing REINFORCE in Stage II | 56.2% | 58.4% | +2.2% |

- Single-turn RL gets highest Acc@t1 but *negative* $\Delta$: no self-correction
- Removing Stage I: $-2\%$ $\Delta$ and $-3\%$ Acc@t2
- Removing reward shaping: $-1.8\%$ $\Delta$
- STaR in Stage II: much worse absolute accuracy + lower $\Delta$

---

## Inference-Compute Scaling

With a budget of $2K$ solution samples per problem, **sequential self-correction beats
parallel sampling**:

- Sample $K$ solutions → self-correct each → majority vote over $K$ corrected answers
- At $K = 32$: parallel-only gets ~64%, sequential (self-correct) gets ~69%
- Self-correction is a more compute-efficient use of the sample budget

---

## Hyperparameters (Table 5)

| | MATH (Flash) | MBPP (Pro) |
|---|---|---|
| Learning rate | 5e-6 | 1e-5 |
| Training steps | 3000 | 1500 |
| Batch size | 512 | 128 |
| Temperature | 1.0 | 1.0 |
| $\alpha$ (reward shaping) | 10 | 10 |
| $\beta_1$ (global KL) | 0.01 | 0.01 |
| $\beta_2$ (Stage I first-turn KL) | 0.1 | 0.25 |

---

## Key Takeaways for Our Project

1. **On-policy data is non-negotiable.** Off-policy correction traces (SFT/STaR) suffer
   distribution shift — the model's own mistakes differ from the data-collection policy's
   mistakes.

2. **Reward shaping prevents collapse.** Without the $\alpha(\hat{r}_2 - \hat{r}_1)$
   bonus, RL converges to "be right on turn 1, don't change anything" — identical to
   SFT's behavior collapse.

3. **Two-stage design is load-bearing.** Stage I (decouple attempts via first-turn KL
   anchor) is what keeps Stage II from collapsing. Skipping Stage I costs 2% $\Delta$
   and 3% absolute accuracy.

4. **Self-correction scales inference compute better than parallel sampling.** For a
   fixed sample budget, sequential correction + vote > parallel vote.

5. **Connection to our backoff/thinking work:** SCoRe's reward shaping bonus
   $\alpha(\hat{r}_2 - \hat{r}_1)$ is conceptually similar to rewarding a model for
   "knowing when to think harder." The key difference is SCoRe uses a fixed two-turn
   protocol, while our backoff tokens let the model decide *within a single generation*
   when to retry.
