# MRT: Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning

**Authors:** Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, Aviral Kumar
**Affiliations:** Carnegie Mellon University, Hugging Face
**arXiv:** 2503.07572 (March 2025)
**Venue:** ICML 2025

---

## What MRT Does

MRT reframes test-time compute as a **meta-RL problem**. The core insight: standard outcome-reward RL only gives a 0/1 signal at the very end of a long reasoning trace. This means the model gets no credit for *making progress* in intermediate steps — it can waste tokens going in circles and still get rewarded if it stumbles on the answer, or make excellent partial progress but get nothing if it doesn't finish.

MRT fixes this by adding a **dense "progress" reward** to each intermediate episode in the reasoning trace.

---

## Core Concepts

### 1. Segmenting reasoning into "episodes"

MRT treats the long output stream as a sequence of episodes $\mathbf{z} = [\mathbf{z}_0, \mathbf{z}_1, \dots, \mathbf{z}_{k-1}]$. Two parameterizations:

- **Open-ended**: episodes are chunks of thought between natural break points (e.g., "Wait", "Alternatively") inside `<think>...</think>` markers
- **Backtracking search**: episodes are explicit step-by-step solution attempts separated by backtrack markers ("This seems off. Let's backtrack to step 3.")

### 2. Progress reward

For each episode $\mathbf{z}_j$, progress measures: **did the model's chance of eventually getting the right answer go up or down after this episode?**

$$r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}) = J_r(\mu(\cdot \mid \mathbf{z}_j, \mathbf{c})) - J_r(\mu(\cdot \mid \mathbf{c}))$$

where $\mu$ is a "meta-prover" policy (in practice, the same LLM forced to stop thinking and give its best-guess answer). $J_r$ is the expected 0/1 correctness of that best-guess.

### 3. The MRT objective

Standard RL fine-tuning loss + a weighted progress bonus:

$$\ell_{\text{MRT}}(\pi; \pi_{\text{old}}) := \ell_{\text{FT}}(\pi) + \alpha \cdot \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{train}}} \left[ \sum_{j=0}^{k-1} \mathbb{E}_{\mathbf{c}_{j-1} \sim \pi_{\text{old}}(\cdot|\mathbf{x}),\; \mathbf{z}_j \sim \pi(\cdot|\mathbf{c}_{j-1})} \left[ r_{\text{prg}}^{\mu}(\mathbf{z}_j; \mathbf{c}_{j-1}) \right] \right]$$

This directly optimizes for **cumulative regret minimization** over the token budget — every episode should make the model more likely to succeed.

### 4. Cumulative regret (Definition 4.1)

$$\Delta_k^{\mu}(\mathbf{x}; \pi) \stackrel{\text{def}}{=} \mathbb{E}_{\mathbf{z} \sim \pi(\cdot|\mathbf{x})} \left[ \sum_{j=0}^{k-1} \left( J_r(\mathbf{x}; \pi_j^*) - J_r(\mathbf{x}; \mu(\cdot|\mathbf{x}, \mathbf{z}_{0:j})) \right) \right]$$

where $\pi_j^*$ is the best budget-agnostic comparator policy for a $j$-episode budget. The red shaded area in Figure 1(b) of the paper. Lower cumulative regret = the model makes steadier progress toward the answer with each episode.

---

## Concrete Example

Imagine the model solving a hard AIME problem:

```
<think>
[Episode 0] "Okay, so I have this problem where ... let me try substitution"
             -> meta-prover best-guess accuracy: 0.25
             -> progress = 0.25 - 0.10 = +0.15  good, rewarded

[Episode 1] "Wait, let me parse the problem again ... actually the constraint is ..."
             -> meta-prover best-guess accuracy: 0.40
             -> progress = 0.40 - 0.25 = +0.15  good, rewarded

[Episode 2] "Alternatively, perhaps it's better to use generating functions ..."
             -> meta-prover best-guess accuracy: 0.35
             -> progress = 0.35 - 0.40 = -0.05  went backwards, penalized

[Episode 3] "But let me double-check ... yes, substitution was right, and the answer is 42"
             -> meta-prover best-guess accuracy: 0.75
             -> progress = 0.75 - 0.35 = +0.40  big progress, big reward
</think>
**Final Answer: 42** -> outcome reward = 1
```

With **standard GRPO**: all four episodes get equal credit from the final r=1. Episode 2 (which went backwards) is reinforced just as much.

With **MRT**: episode 2 gets negative signal, episodes 0/1/3 get positive signal proportional to how much they helped. The model learns to *make steady progress* and avoid unproductive detours.

---

## Two Practical Instantiations

### MRT (STaR variant)

1. Generate complete rollouts from base model for each query x
2. Segment thinking traces into episodes
3. For each prefix $\mathbf{z}_{0:j}$, compute meta-prover reward $J_r(\mu(\cdot|\mathbf{x}, \mathbf{z}_{0:j}))$ by forcefully terminating thought and evaluating best-guess accuracy
4. Compute progress $r_{\text{prg}}(\mathbf{z}_j)$ per episode
5. Filter: keep traces that (a) maximize cumulative progress and (b) eventually produce correct answer
6. SFT on filtered traces; repeat iteratively

### MRT (RL variant)

1. Generate partial rollouts, terminating after a random number of episodes
2. Compute meta-prover rewards at each prefix
3. Sample on-policy rollouts conditioned on prefix, split between continuing to reason and producing best-guess solution
4. Normalize rewards across this set of traces to compute progress bonus
5. Train with GRPO/PPO using outcome reward + dense progress bonus

---

## Case Study: DeepSeek-R1 Does Not Optimize Regret

The paper analyzes DeepSeek-R1-Distill-Qwen-32B on AIME 2024 and OmniMATH using the $\lfloor\text{maj}@p\rfloor_j$ metric (truncate thought after $j$ episodes, force answer, majority vote over $p$ samples).

Key finding: with many episodes (41-45), accuracy sometimes *decreases* with each new episode. The model trained with outcome-reward RL does not make consistent progress — it can go backwards. A naive strategy of just running fewer episodes + majority voting often beats the full long trace. This motivates MRT's explicit progress optimization.

---

## Key Results

| Metric | MRT vs GRPO |
|--------|-------------|
| Accuracy gain | **2-3x** relative improvement over outcome-reward RL |
| Token efficiency | **1.5x** over GRPO, **5x** over base model |
| Extrapolation | Keeps making progress even at **2x** the training token budget |
| Length penalty comparison | Length penalty hurts accuracy; MRT reduces length *and* improves accuracy |

### Table 1: Pass@1 on math benchmarks (1.5B scale)

| Base model + Approach | AIME 2024 | AIME 2025 | AMC 2023 | MinervaMATH | MATH500 | Avg. |
|---|---|---|---|---|---|---|
| DeepScaleR-1.5B + GRPO | 44.5 (+1.7) | 39.3 (+2.6) | 81.5 (-1.5) | 24.7 | 84.9 | 55.0 (+0.5) |
| DeepScaleR-1.5B + length penalty | 40.3 (-2.5) | 30.3 (-6.4) | 77.3 (-5.7) | 23.0 | 83.2 | 50.8 (-3.7) |
| **DeepScaleR-1.5B + MRT** | **47.2 (+4.4)** | **39.7 (+3.0)** | **83.1 (+0.1)** | 24.2 | 85.1 | **55.9 (+1.4)** |
| R1-Distill-Qwen-1.5B + GRPO | 29.8 (+1.1) | 27.3 (+1.3) | 70.5 (+0.6) | 22.1 | 80.3 | 46.0 (+1.1) |
| **R1-Distill-Qwen-1.5B + MRT** | **30.3 (+1.6)** | **29.3 (+3.3)** | **72.9 (+3.0)** | **22.5** | **80.4** | **47.1 (+2.2)** |

### Backtracking search setting (Llama-3.1 8B/3B)

- MRT (STaR) on 8B: **1.7x** token efficiency improvement, **5.4x** over base
- MRT (RL) on 3B: **1.6x** token efficiency over GRPO, **4.6x** over base
- Both outperform RISE (self-correction without backtracking) in efficiency and peak performance

---

## Ablation Insights

### Progress vs length

- Length penalties improve token efficiency but **sacrifice peak accuracy**
- MRT's dense progress reward increases accuracy while *slightly* reducing length — net positive on both axes
- Length oscillates during RL training (~5000 tokens); MRT keeps it slightly lower than GRPO

### Budget curriculum explains DeepScaleR gains

- Training first at 8K budget then expanding to 16K (as DeepScaleR does) achieves lower regret than training at 16K from scratch
- The 8K phase forces more progress per token; the curriculum is an *implicit* form of regret minimization
- MRT makes this explicit

---

## Explore/Exploit Spectrum (Figure 4)

```
Each episode          Sequential episodes
exploits              allow exploration
    |                       |
    v                       v
Self-correct / RL^2 --- MRT (Ours) --- R1 / E-RL^2
Per-episode outcome     Progress-based   Outcome reward
reward                  dense reward     (end only)
```

- Pure outcome reward (right): encourages unstructured exploration, no signal for intermediate quality
- Per-episode outcome reward (left, e.g. SCoRe): too exploitative, each episode forced to succeed on its own
- MRT (center): strikes explore/exploit balance by rewarding information gain

---

## Relevance to Our Project

MRT's "progress" reward is conceptually close to what we need for learning backoff/undo as a first-class action. Key differences:

1. **MRT is still append-only** — episodes add to context but never reclaim it. Our approach makes rewind + KV cache truncation a discrete learned action.
2. **MRT doesn't learn action selection** — it rewards *what happened* in each episode but doesn't teach the model to choose between action types (continue, verify, backoff). We want the policy to learn *when* to take which action.
3. **MRT's progress metric could be reused** — the $r_{\text{prg}}$ formulation (change in meta-prover accuracy) is a natural dense reward signal for our GRPO training. We could adapt it to also reward backoff decisions that improve the meta-prover's accuracy after context truncation.
4. **Budget-agnosticism aligns with our goals** — we also want a policy that works across varying compute budgets, not one tuned to a specific token limit.
