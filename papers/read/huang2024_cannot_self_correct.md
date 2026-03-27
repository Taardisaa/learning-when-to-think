# Large Language Models Cannot Self-Correct Reasoning Yet

**Authors:** Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou (Google DeepMind)
**Venue:** ICLR 2024
**arXiv:** 2310.01798

---

## Core Thesis

LLMs **cannot intrinsically self-correct** their reasoning. When an LLM tries to revise its own answer using only its own judgment (no external oracle, no tool feedback), accuracy **stays flat or degrades**. Prior claims of self-correction success were artifacts of three confounds:

1. **Oracle labels** used to decide when to stop correcting
2. **Unfair compute comparison** against self-consistency
3. **Weaker initial prompt** making the correction prompt look beneficial

---

## Key Definition: Intrinsic Self-Correction

The model attempts to correct its initial response **using only its own capabilities**, with no external feedback (no ground-truth labels, no code execution, no retrieval, no human input).

Formally, a three-step prompting pipeline:

$$
y_0 = \text{LLM}(x) \quad \text{(initial generation)}
$$

$$
f = \text{LLM}(x, y_0, p_{\text{feedback}}) \quad \text{(self-generated feedback)}
$$

$$
y_1 = \text{LLM}(x, y_0, f, p_{\text{refine}}) \quad \text{(refined answer)}
$$

where $p_{\text{feedback}}$ is a prompt like *"Review your previous answer and find problems"* and $p_{\text{refine}}$ is *"Based on the problems you found, improve your answer."*

The central question: is $\text{Acc}(y_1) > \text{Acc}(y_0)$?

**Answer: No.** Across all tested models and benchmarks, $\text{Acc}(y_1) \leq \text{Acc}(y_0)$.

---

## Why Self-Correction Hurts: The Bias Argument

If the model is well-calibrated and the initial prompt is well-designed, then $y_0$ is already the model's best response to $(x, p_{\text{initial}})$ under its decoding strategy. The feedback prompt $p_{\text{feedback}}$ is an **additional input that can only bias** the model away from this optimum:

$$
y_1 = \arg\max_y \; P(y \mid x, y_0, p_{\text{feedback}})
$$

This conditional distribution is **not the same** as $P(y \mid x)$. The feedback prompt carries an implicit prior that *something is wrong*, which pushes the model to change correct answers into incorrect ones.

---

## Empirical Evidence

### Benchmarks
- **GSM8K** — grade-school math (1,319 problems)
- **CommonSenseQA** — multiple-choice commonsense (1,221 questions)
- **HotpotQA** — multi-hop QA (100 questions)

### Result 1: Oracle self-correction works; intrinsic does not

With oracle labels (stop correcting once answer is correct):

| Model   | Method              | GSM8K | CommonSenseQA | HotpotQA |
|---------|---------------------|-------|---------------|----------|
| GPT-3.5 | Standard Prompting | 75.9  | 75.8          | 26.0     |
| GPT-3.5 | Self-Correct (Oracle) | 84.3 | 89.7         | 29.0     |
| GPT-4   | Standard Prompting | 95.5  | 82.0          | 49.0     |
| GPT-4   | Self-Correct (Oracle) | 97.5 | 85.5         | 59.0     |

Without oracle labels (intrinsic self-correction):

| Model   | Method               | #calls | GSM8K  | CommonSenseQA | HotpotQA |
|---------|----------------------|--------|--------|---------------|----------|
| GPT-3.5 | Standard Prompting  | 1      | **75.9** | **75.8**    | **26.0** |
| GPT-3.5 | Self-Correct (rd 1) | 3      | 75.1   | 38.1          | 25.0     |
| GPT-3.5 | Self-Correct (rd 2) | 5      | 74.7   | 41.8          | 25.0     |
| GPT-4   | Standard Prompting  | 1      | **95.5** | **82.0**    | **49.0** |
| GPT-4   | Self-Correct (rd 1) | 3      | 91.5   | 79.5          | **49.0** |
| GPT-4   | Self-Correct (rd 2) | 5      | 89.0   | 80.0          | 43.0     |

Standard prompting **dominates** in every cell once oracle labels are removed.

### Result 2: Change-direction analysis (GSM8K, GPT-3.5)

After two rounds of self-correction, the model's answer changes break down as:

| Transition | Fraction |
|---|---|
| No Change | 74.7% |
| Correct $\Rightarrow$ Incorrect | 8.8% |
| Incorrect $\Rightarrow$ Correct | 7.0% |
| Incorrect $\Rightarrow$ Incorrect | 9.5% |

The model flips correct→incorrect **more often** than incorrect→correct. The net effect is negative.

---

## Concrete Example: Self-Correction Succeeds (GSM8K)

**Problem:** Christina is planning a birthday party and needs 0.75 gift bags per invited guest, because 1/4 of attendees don't show up. She invited 16 friends. Gift bags are \$2 each. How much will she spend?

**Initial answer (Incorrect):**
$$
\text{Guests not showing} = 16 \times \frac{1}{4} = 4, \quad \text{Attending} = 16 - 4 = 12
$$
$$
\text{Gift bags} = 0.75 \times 12 = 9, \quad \text{Cost} = 9 \times 2 = \$18 \quad \boxed{18} \; (\text{Wrong})
$$

**After self-correction (Correct):**
$$
\text{Attending} = 16 \times \frac{3}{4} = 12, \quad \text{Gift bags} = 12, \quad \text{Cost} = 12 \times 2 = \$24 \quad \boxed{24} \; (\text{Correct})
$$

Here the model caught the error that 0.75 bags/guest × attending guests = attending guests (i.e., each attending guest gets 0.75 bags, rounded to 1).

## Concrete Example: Self-Correction Fails (GSM8K)

**Problem:** Terry eats 2 yogurts a day. They are on sale at 4 yogurts for \$5. How much does he spend over 30 days?

**Initial answer (Correct):**
$$
\text{Sets per day} = \frac{2}{4} = 0.5, \quad \text{Cost/day} = 0.5 \times 5 = \$2.50
$$
$$
\text{Total} = 2.50 \times 30 = \$75.00 \quad \boxed{75} \; (\text{Correct})
$$

**After self-correction (Incorrect):**
The model "realizes a mistake" that doesn't exist, re-derives the answer as \$37.50 by incorrectly dividing again:
$$
\text{"Corrected"} = \$2.50 \times 15 = \$37.50 \quad \boxed{37.50} \; (\text{Wrong})
$$

The feedback prompt's implicit assumption that *something is wrong* biased the model into fabricating an error.

---

## Multi-Agent Debate ≈ Self-Consistency (Section 4)

Multi-agent debate (Du et al., 2023) uses $k$ copies of the same LLM to critique each other. The paper shows this is **no better than self-consistency** (majority voting over $k$ independent samples) at equal compute:

| Method | #responses | GSM8K |
|---|---|---|
| Standard Prompting | 1 | 76.7 |
| Self-Consistency | 3 | 82.5 |
| Multi-Agent Debate (rd 1) | 6 | 83.2 |
| Self-Consistency | 6 | 85.3 |
| Multi-Agent Debate (rd 2) | 9 | 83.0 |
| Self-Consistency | 9 | **88.2** |

At 9 responses, self-consistency beats multi-agent debate by **5.2 points**. The "debate" is just a less efficient form of sampling + voting.

---

## Prompt Design Confound (Section 5)

Some papers (e.g., Self-Refine, Madaan et al. 2023) show gains because their **initial prompt is deliberately weak** — it omits key task constraints that only appear in the feedback prompt.

**Example — Constrained Generation task:** Generate a paragraph using ALL of 20-30 given concepts.

- **Original initial prompt (Madaan et al.):** does not say "include ALL concepts"
- **Feedback prompt:** asks "what concepts are missing?" — this adds new information!
- **Fixed initial prompt (this paper):** adds "*Write a paragraph that includes \*ALL\* of the above concepts*"

| Method | #calls | CommonGen-Hard |
|---|---|---|
| Standard Prompting (Madaan) | 1 | 44.0 |
| Self-Correct (Madaan) | 7 | 67.0 |
| **Standard Prompting (fixed)** | **1** | **81.8** |
| Self-Correct (fixed) | 7 | 75.1 |

The fixed prompt **without any self-correction** beats the original self-correction pipeline by **14.8 points**. And applying self-correction to the fixed prompt **hurts** by 6.7 points.

---

## When Self-Correction CAN Work

The paper does not claim self-correction is universally useless. It works when:

1. **External feedback is available** — code execution results, unit test outputs, tool/API responses, human feedback. These provide *new information* the model didn't have.
2. **Verifiable constraints** — e.g., "output must be valid JSON", "code must compile". The verifier is external to the LLM's reasoning.
3. **Style/safety alignment** — LLMs can judge whether a response is inappropriate (Ganguli et al., 2023), even if they cannot judge whether their *reasoning* is correct.

The critical distinction:

$$
\text{Self-correction works} \iff \text{feedback carries information not in } (x, y_0)
$$

---

## Implications for "Learning When to Think"

This paper establishes that **the LLM itself is a poor judge of its own reasoning quality**. For a system that learns when to allocate more compute (think harder), this means:

- Don't rely on the model's self-assessment ("am I confident?") as the signal for when to think more — the model cannot reliably distinguish correct from incorrect reasoning.
- External verifiers (reward models, code execution, ground-truth checks) are necessary for reliable self-correction.
- Simply re-prompting "try again" is worse than sampling multiple independent answers and voting (self-consistency), which is a pure compute-scaling baseline.
- The value of "thinking harder" must come from **structurally different** computation (e.g., search, tool use, decomposition), not from iterative self-critique of the same reasoning chain.

---

## Beyond Huang: Why Humans Can Self-Correct But Naive LLM Self-Correction Can't

Huang et al.'s result is **narrower** than it first appears. It kills the pattern `generate → "is this right?" → "try again"`. But humans don't self-correct that way either. Human self-correction decomposes into at least three distinct cognitive processes:

### 1. Calibrated Uncertainty (Fast, Pre-Verbal)

Humans *feel* unsure before they know *why*. This is a fast signal that precedes any explicit review. LLMs have a direct analogue: **logprob entropy / token-level confidence**. The key insight is that this signal is **not** the same as asking the model "are you confident?" in natural language (which Huang shows fails). It's a statistical property of the output distribution:

$$
H(y_t) = -\sum_{v \in V} P(v \mid x, y_{<t}) \log P(v \mid x, y_{<t})
$$

High $H(y_t)$ at critical reasoning steps = the model is uncertain, even if its generated text sounds confident. This is an **implicit** signal, not an explicit self-judgment — and it sidesteps Huang's critique entirely.

### 2. Strategy Switching (Not "Try Again" — Try Differently)

When uncertain, humans don't re-read their work and "try harder." They **switch approach**: check with a simpler example, verify boundary conditions, decompose differently. Huang's self-correction prompt does none of this — it just says "review your answer," which is the same model doing the same computation at the same level.

The failed pattern:
$$
y_1 = \text{LLM}(x, y_0, \text{"review your answer"}) \quad \text{— same } P(\cdot), \text{ same level of reasoning}
$$

What humans actually do:
$$
y_1 = \text{LLM}(x, \text{strategy}_2) \quad \text{— independent sample from a different reasoning path}
$$

This is why self-consistency ($k$ independent samples + majority vote) beats self-correction at equal compute. The diversity comes from **independent draws**, not from iterative self-critique.

### 3. Invariant Checking (Decomposed Verification)

Humans verify against known constraints that are **independently easier** than the original problem. This is structured and *different* from the original reasoning, not a re-run of it.

**Example — self-verify instead of self-correct:**
```
Original problem: "What is 17 × 23?"  →  model answer: 391

Bad self-correct:  "Review your answer."  →  same computation, biased by "something is wrong"

Good self-verify (decomposed checks):
  - "Is 391 close to 20×23=460?"             → order-of-magnitude check ✓
  - "Is 391 odd? (17 odd × 23 odd → odd)"    → parity invariant ✓
  - "Does 391 end in 1? (7×3=21 → yes)"      → last-digit invariant ✓
```

Each check is a **new computation** that provides information not in $(x, y_0)$. This satisfies the condition Huang identifies:

$$
\text{Self-correction works} \iff \text{feedback carries information not in } (x, y_0)
$$

Decomposed verification checks **do** carry new information because they compute different functions of the input.

---

## Design Directions for Our Backoff Architecture

Our backoff system (logprob-gated extra compute) already has the right skeleton. The question is: **what should the extra compute do when triggered?**

### Direction A: Entropy-Gated Diverse Sampling (self-consistency, not self-correction)

When logprob entropy is high, don't ask the model to fix its answer. Instead, **sample $k$ independent reasoning paths and vote**. Huang et al. prove the compute efficiency of this:

$$
\text{Self-Consistency}(k{=}9) = 88.2 \quad \gg \quad \text{Self-Correct}(k{=}5) = 74.7 \quad \text{on GSM8K}
$$

Our value-add over naive self-consistency: **spend those extra samples only when entropy says you need them**. Most questions don't need $k=9$; a few hard ones do. This is adaptive compute allocation:

$$
k(x) = \begin{cases} 1 & \text{if } H(y \mid x) < \tau_{\text{low}} \quad \text{(confident — accept first answer)} \\ k_{\text{med}} & \text{if } \tau_{\text{low}} \leq H(y \mid x) < \tau_{\text{high}} \quad \text{(uncertain — sample \& vote)} \\ k_{\text{high}} & \text{if } H(y \mid x) \geq \tau_{\text{high}} \quad \text{(very uncertain — sample many)} \end{cases}
$$

### Direction B: Decomposed Self-Verification (not "is this right?" but "check these properties")

Instead of the generic feedback prompt that Huang shows fails, generate **specific verification sub-questions** that are independently easier. The backoff mechanism triggers this only when entropy is high:

$$
\text{verify}(x, y_0) = \bigwedge_{i=1}^{m} \text{LLM}(c_i(x, y_0))
$$

where each $c_i$ is a **cheap, targeted check** (parity, magnitude, dimensional analysis, edge-case substitution). If any check fails, resample; if all pass, accept.

The critical difference from Huang's setup: each $c_i$ computes a **different function** of the input, providing genuinely new information. The model isn't asked "is this right?" — it's asked "does this specific property hold?"

### Direction C: Train Solver ≠ Judge (Asymmetric GRPO)

We're already doing GRPO training. We can shape the reward to create an **asymmetry** between generation and verification:

- **Solver mode**: generates a chain of reasoning (generative task — hard)
- **Verifier mode**: predicts whether a *completed* chain reaches the right answer (discriminative task — easier)

Even with the same model weights, the verifier uses a different prompt framing + task structure. This is essentially a lightweight **ORM (outcome reward model)** that lives inside the generation loop. The GRPO reward can be structured to train both modes:

$$
R(y, \hat{y}) = \begin{cases} +1 & \text{if solver correct} \\ +1 & \text{if verifier correctly identifies solver error} \\ -1 & \text{if verifier incorrectly flags correct answer (Huang's failure mode)} \end{cases}
$$

This directly penalizes the Correct→Incorrect flip that Huang identifies as the core failure of self-correction (8.8% of cases on GSM8K).

### Summary: What Huang Kills vs. What Survives

| Pattern | Status | Why |
|---|---|---|
| `generate → "is this right?" → "try again"` | **Dead** | Same $P(\cdot)$, feedback biases toward change |
| Logprob entropy as uncertainty signal | **Alive** | Implicit signal, not explicit self-judgment |
| Diverse sampling + voting (self-consistency) | **Alive** | Independent draws, no self-critique |
| Decomposed verification (check properties) | **Alive** | Each check is new information |
| Trained verifier mode (discriminative judge) | **Alive** | Different task than generation |
| Adaptive compute: think more only when uncertain | **Alive** | Orthogonal to Huang — it's about *when*, not *how* to correct |
