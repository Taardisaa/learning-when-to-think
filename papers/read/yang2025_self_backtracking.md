# Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models

**Authors:** Xiao-Wen Yang, Xuan-Yi Zhu, Wen-Da Wei, Ding-Chu Zhang, Jie-Jing Shao, Zhi Zhou, Lan-Zhe Guo, Yu-Feng Li (Nanjing University)

**Date:** 2025 | **Venue:** arXiv:2502.04404

## Problem

Current o1-like reasoning models suffer from two key limitations:

| Problem | Description | Consequence |
|---|---|---|
| Inefficient overthinking | Models produce long reasoning traces even for simple problems | Wasted compute, no performance gain |
| External reward model dependency | Search (MCTS, beam search) requires a separate verifier/PRM to evaluate states | Extra model cost, reward hacking risk |

The root cause: **LLMs have not internalized the search process itself.** Search is bolted on externally rather than being a learned capability. The authors argue that **backtracking** — the ability to recognize dead ends and revert to earlier states — is the key missing primitive. Classical search algorithms (DFS, BFS, MCTS) all rely on backtracking, but LLMs generate left-to-right without the ability to rewind.

## Key Concepts

### Self-Backtracking
The ability of an LLM to autonomously decide **when** to backtrack (detect that the current path is unproductive) and **where** to backtrack (revert to the correct earlier state), without an external reward model. Implemented via a special `<backtrack>` token that the model learns to emit.

**Example:** In a Countdown task (reach target 62 from numbers [2, 39, 45, 10]):
```
45-39=6
39+6=45       ← wrong path, doesn't lead to 62
<backtrack>    ← model recognizes dead end
45-39=6       ← reverts to this state
10*6=60       ← tries new direction
2+60=62
Goal Reached!
```

**Distinction from o1-style self-correction:** O1 models emit natural-language self-correction ("Wait, that's wrong, let me try again") but still generate linearly. Self-backtracking uses a **structured token** that triggers actual state reversion in the inference algorithm, enabling tree search.

### Backtracking Data ($\mathcal{D}_{back}$)
Training examples that teach the model when/where to backtrack. Each example consists of:
1. A prefix of the optimal solution path: `prefix(y_j)`
2. An erroneous action appended: `a_{err}`
3. The `<backtrack>` token

**Example construction:**
- Optimal solution: `45-39=6 → 10*6=60 → 2+60=62`
- Backtrack sample: `45-39=6 → 39+6=45 → <backtrack>`

The model learns: given the prefix `45-39=6` followed by wrong action `39+6=45`, emit `<backtrack>`.

### Expansion, Backtracking, Selection (Inference Algorithm)
The three-phase inference loop, parameterized by breadth $N$ and depth $b$:
- **Expansion:** Sample $N$ continuations from the current state. Separate into those containing `<backtrack>` and those without. Non-backtrack predictions go into the candidate set.
- **Backtracking:** For predictions containing `<backtrack>`, roll back one reasoning step and re-expand. Repeat up to $b$ times (depth budget).
- **Selection:** Score all candidate paths by negative perplexity, return the highest-scoring one.

**Key parameters:**
- $N$ = breadth (number of samples per expansion) — controls exploration width
- $b$ = backtracking depth budget — controls how many times the model can revert ($b=0$ means no actual backtracking, just sampling + selection)

### Self-Improvement (Expert Iteration)
A distillation loop that converts slow-thinking (search with backtracking) into fast-thinking (single-pass greedy):
1. Generate solutions using the slow-thinking self-backtracking algorithm
2. Filter for correct solutions (verified by an evaluator)
3. Retrain the model via SFT on the correct solutions
4. Repeat

This transfers the exploration gains from tree search into the model's greedy decoding path.

## Core Design / Method

### Training Data Construction

The final training dataset is a mixture:

$$\mathcal{D} = \mathcal{D}_{op} \cup \mathcal{D}_{back}$$

| Term | Meaning |
|---|---|
| $\mathcal{D}_{op}$ | Optimal-path solutions (correct, no backtracking) |
| $\mathcal{D}_{back}$ | Backtracking examples (prefix + error + `<backtrack>`) |

Both sets share the same questions (balanced construction). The recommended ratio is $\mathcal{D}_{op} : \mathcal{D}_{back} \geq 1 : 0.5$ (i.e., at least twice as many optimal examples as backtrack examples). Increasing backtrack data beyond 1:1 slightly hurts performance.

Three types of errors are injected to construct $a_{err}$:
1. **Exploration errors** (1/4): DFS branches that don't match the correct solution steps
2. **Computational errors** (1/2): Invalid mathematical equations inserted into the solution
3. **Rule violations** (1/4): Using operands not in the available set

### Training Loss

$$\mathcal{L}(\theta) = \mathcal{L}_{SFT}(\theta) + \mathcal{L}_{backtrack}(\theta)$$

The SFT loss on optimal solutions:

$$\mathcal{L}_{SFT}(\theta) = -\frac{1}{n_{op}} \sum_{i=1}^{n_{op}} \log p_\theta(y_i | x_i)$$

| Term | Meaning |
|---|---|
| $n_{op}$ | Number of optimal-path training examples |
| $y_i$ | The optimal solution for question $x_i$ |
| $p_\theta(y_i \mid x_i)$ | Model's probability of generating the optimal solution |

The backtrack loss has two parts:

$$\mathcal{L}_{backtrack}(\theta) = -\frac{1}{n_{back}} \sum_{j=1}^{n_{back}} \log p_\theta(\text{prefix}(y_j) | x_j) - \frac{1}{n_{back}} \sum_{j=1}^{n_{back}} \log p_\theta(\langle\text{backtrack}\rangle | x_j \circ \text{prefix}(y_j) \circ a_{err})$$

| Term | Meaning |
|---|---|
| $n_{back}$ | Number of backtracking training examples |
| $\text{prefix}(y_j)$ | Correct partial solution up to the error point |
| $a_{err}$ | The erroneous action appended after the prefix |
| $\langle\text{backtrack}\rangle$ | Special backtrack token |

**Critical design choice:** The loss **masks** $a_{err}$ — the model is NOT trained to predict the erroneous action itself. This prevents the model from learning to generate errors. It only learns: (1) generate correct prefixes, and (2) emit `<backtrack>` after seeing an error.

**Concrete example:** For the training input `[Target: 62, Numbers: 2, 39, 45, 10] → 45-39=6 | 39+6=45 | <backtrack>`:
- Loss is computed on `45-39=6` (prefix) and `<backtrack>` (detection)
- Loss is **not** computed on `39+6=45` (the error) — this region is masked

### Inference Algorithm

At inference time with parameters $(N, b)$:

1. Start at initial state $s_0$
2. Sample $N$ predictions from state $s$
3. Split: predictions without `<backtrack>` → candidate set; predictions with `<backtrack>` → trigger rollback
4. For `<backtrack>` predictions: revert one step, re-expand (repeat up to $b$ times)
5. After budget exhausted, score all candidates by **negative perplexity**
6. Return highest-scoring candidate

**No external reward model needed** — the model's own perplexity serves as the scoring function, because the `<backtrack>` token implicitly performs state evaluation (if the model doesn't backtrack, it believes the path is viable).

## Results

### Main Results: Countdown Task (Llama3.2-1B / 3B)

| Method | Llama3.2-1B Seen | Llama3.2-1B New | Llama3.2-3B Seen | Llama3.2-3B New |
|---|---|---|---|---|
| SFT + Greedy | 28.60 | 28.92 | 33.98 | 32.68 |
| SFT + Beam Search (16) | 31.68 | 31.90 | 35.82 | 34.36 |
| DPO | 29.06 | 27.64 | 34.46 | 32.72 |
| Best-of-N (N=8) | 41.26 | 40.68 | 47.84 | 48.56 |
| Best-of-N (N=32) | 25.60 | 27.04 | 44.38 | 45.88 |
| Self-Backtrack (b=0, N=32) | 70.66 | 72.14 | 60.18 | 58.06 |
| **Self-Backtrack (b=1, N=32)** | **73.54** | **73.78** | **64.12** | **61.98** |

Self-backtracking achieves **+40pp** over SFT+Greedy on Llama3.2-1B. Even without actual backtracking ($b=0$, just sampling + selection), the backtracking-trained model outperforms Best-of-N by ~30pp, suggesting the `<backtrack>` token implicitly acts as a state evaluator.

### Comparison with Search-Augmented Methods (Llama3.2-1B)

| Method | Seen Targets | New Targets |
|---|---|---|
| DFS (b=32) | 49.12 | 48.90 |
| DFS (b=64) | 60.00 | 61.06 |
| SoS (Gandhi et al.) | 57.50 | 53.40 |
| GSoS (Moon et al.) | 69.00 | 67.20 |
| **Self-Backtracking (best)** | **73.54** | **73.78** |

Outperforms even symbolic DFS with 64-step budget and search-augmented methods (SoS, GSoS).

### Self-Improvement via Expert Iteration (Llama3.2-1B)

| Iteration | Slow-thinking (Seen) | Fast-thinking Greedy (Seen) |
|---|---|---|
| 0 | ~72 | 28.92 |
| 1 | ~74 | ~55 |
| 2 | ~76 | ~66 |
| 3 | ~77 | ~70 |

After 3 rounds of expert iteration, fast-thinking (greedy) **surpasses** the original slow-thinking performance. The model distills its search capabilities into single-pass generation.

### $\mathcal{D}_{op}$ to $\mathcal{D}_{back}$ Ratio (Llama3.2-1B, Seen Targets)

| Ratio (op:back) | b=0, N=8 | b=1, N=8 | b=0, N=16 | b=1, N=16 |
|---|---|---|---|---|
| **1:0.5** | **66.70** | **68.00** | **69.80** | **71.12** |
| 1:1 | 66.66 | 67.60 | 67.80 | **72.50** |
| 1:2 | 64.92 | 64.66 | 68.18 | 65.66 |
| 1:3 | 64.78 | 65.18 | 68.18 | 66.56 |
| 1:4 | 65.06 | 65.94 | 69.16 | 70.12 |

Best ratio: **1:0.5 for small N, 1:1 for large N**. Too much backtracking data (1:3, 1:4) hurts.

### Effect of Backtracking Depth $b$

Increasing $b$ from 0 to 1 gives significant gains. $b > 1$ yields **diminishing returns** — the diversity of outputs from secondary backtracking drops sharply. The authors attribute this to the model generating similar alternatives after deeper rollbacks.

## Key Takeaways

1. **A single `<backtrack>` token enables tree search without external reward models.** The model learns both error detection (when to backtrack) and state reversion (where to backtrack) from SFT alone. At inference, the model's own perplexity replaces the reward model for path scoring.

2. **Even without actual backtracking ($b=0$), backtracking-trained models massively outperform baselines.** The `<backtrack>` token functions as an implicit state evaluator — paths where the model doesn't backtrack are inherently higher-quality, so sampling + selection works extremely well. This is the most surprising finding.

3. **Masking the error action $a_{err}$ during training is critical.** The model should learn to *detect* errors, not to *generate* them. Training loss is only computed on the correct prefix and the `<backtrack>` token, with the erroneous action masked out.

4. **Optimal data mix: at least 1:0.5 optimal-to-backtrack ratio.** Too much backtracking data hurts. The model needs sufficient examples of correct paths to learn what "good" looks like, with backtrack examples teaching it what "bad" looks like.

5. **Depth budget $b=1$ is sufficient; deeper backtracking has diminishing returns.** The diversity of alternatives after rolling back 2+ steps collapses. This aligns with the intuition that single-step correction captures most of the value.

6. **Self-improvement (expert iteration) converts slow-thinking into fast-thinking.** After 3 rounds, greedy single-pass generation surpasses the original search-based performance. This is a powerful paradigm: train with search, distill into fast inference.

7. **Best-of-N with reward models degrades at high N due to reward hacking.** BoN peaks around N=8 then drops, while self-backtracking consistently improves with N, showing the internalized evaluator is more robust than external reward models.

8. **The method is only tested on Countdown, a structured combinatorial task.** Generalization to open-ended math reasoning (like MATH) or natural language tasks remains unvalidated. The authors acknowledge this as a limitation.

## Connection to Our Work

**Closest prior work to `<backoff_N>`.** The key differences:

| Aspect | Self-Backtracking | Our `<backoff_N>` |
|---|---|---|
| Token | Single `<backtrack>` (always reverts 1 step) | `<backoff_1>`, `<backoff_2>`, `<backoff_3>` (variable rewind depth) |
| Rewind depth | Always 1 step; multi-step via repeated application | Encoded in the token itself (N = number of chunks to rewind) |
| Error masking | $a_{err}$ is masked in training loss | Error chunks are kept verbatim (real model errors, not injected) |
| Error source | Synthetically injected (3 types: exploration, computation, rule violation) | Real errors from the model's own wrong rollouts |
| Directive | None — just backtracks | Explicit corrective directive after `<backoff_N>` explaining the error |
| Task domain | Countdown (combinatorial search) | MATH (open-ended mathematical reasoning) |
| Inference | Tree search with expansion/selection | Standard autoregressive generation (the model self-corrects inline) |

**Implications for our design:**
- Their finding that **$b=1$ suffices** supports our heavier weighting of `<backoff_1>` (60%) vs deeper backoffs.
- Their **error masking** strategy is the opposite of ours — we keep errors verbatim to match real model behavior. Worth considering whether masking would help or hurt in our setting.
- Their **data ratio finding (1:0.5 optimal-to-backtrack)** informs our mix of clean vs. perturbed trajectories. We should ensure clean examples outnumber backoff examples.
- The **self-improvement loop** is directly applicable: after training with backoff tokens, we could use the model to generate solutions, filter correct ones, and retrain for fast inference.
