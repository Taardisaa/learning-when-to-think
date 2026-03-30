# Improve Mathematical Reasoning in Language Models by Automated Process Supervision

**Authors:** Liangchen Luo*, Yinxiao Liu*, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Meng, Jiao Sun, Abhinav Rastogi (Google DeepMind, Google)

**Date:** 2024 | **Venue:** arXiv:2406.06592

## Problem

Training PRMs requires step-level correctness labels. Two existing approaches:

| Method | Cost | Quality | Scale |
|---|---|---|---|
| Human annotation (Lightman et al. 2023) | Very high ($$$) | High | 800K labels |
| Brute-force Monte Carlo (Math-Shepherd) | High compute | Noisy | Limited by O(kM) rollouts per solution |

Both are bottlenecked: human labeling doesn't scale, brute-force MC estimation requires rolling out from *every* step (O(kM) policy calls for k rollouts × M steps). The paper asks: **can we automate process supervision data collection efficiently enough to beat both human labels and brute-force MC?**

## Key Concepts

### Monte Carlo Estimation of Step Correctness

Given a question $q$ and a partial solution up to step $t$ ($x_{1:t}$), sample $k$ completions ("rollouts") from that prefix. The fraction that reach the correct final answer estimates the correctness of the prefix:

$$c_t = \text{MonteCarlo}(q, x_{1:t}) = \frac{\text{num(correct rollouts from } t\text{-th step)}}{\text{num(total rollouts from } t\text{-th step)}}$$

**Example:** 3 rollouts from step 4 of a solution — 2 reach the correct answer, 1 doesn't. $c_4 = 2/3 \approx 0.67$. The prefix up to step 4 is "probably correct" since correct completions exist.

Key insight from Lightman et al.: you only need to label up to the **first error**. If all rollouts from step $m$ fail ($c_m = 0$), the error is at or before step $m$.

### Binary Search for First Error

Instead of estimating $c_t$ for every step (O(kM)), use binary search: split the solution at midpoint $m$, roll out from there. If $c_m > 0$, error is in the second half; if $c_m = 0$, error is in the first half. Repeat. Finds the first error in $O(k \log M)$ instead of $O(kM)$.

**Example:** 8-step solution with error at step 7:
- Split at step 4: $c_4 > 0$ → error in steps 5-8
- Split at step 6: $c_6 > 0$ → error in steps 7-8
- Split at step 7: $c_7 = 0$ → step 7 is the first error

3 rounds of binary search × $k$ rollouts each, vs. 8 rounds brute-force.

### State-Action Tree

A tree where each node is a state $s = (q, x_{1:t})$ (question + prefix) and each edge is an action $a$ (one or more reasoning steps). The tree stores per-node statistics:

- $N(s)$: visit count
- $\text{MC}(s)$: Monte Carlo correctness estimate
- $Q(s, r)$: selection priority for rollout $r$

This reuses rollouts across binary searches — a rollout sampled for one node's MC estimation can seed new nodes in the tree, amortizing compute.

### "Supposed-to-be-correct Wrong-answer" Rollouts

Rollouts from states with high MC(s) (≈ correct prefix) that nonetheless reach a wrong final answer. These are the most informative for PRM training — the error must be in this specific rollout's continuation, not the prefix. Directly inspired by Lightman et al.'s "convincing wrong-answer" active learning strategy.

## Core Design / Method

### OmegaPRM: MCTS for Process Data Collection

Three phases per iteration (inspired by AlphaGo but simplified):

**1. Select:** Pick the most valuable rollout to search next.

$$Q(s, r) = \alpha^{1 - \text{MC}(s)} \cdot \beta^{\frac{\text{len}(r)}{L}}$$

| Term | Meaning |
|---|---|
| $\alpha \in (0,1]$ | Preference weight — higher $Q$ when MC(s) is close to 1 (supposed-to-be-correct states) |
| $\beta \in (0,1]$ | Length penalty — discourages excessively long rollouts |
| $\text{len}(r)$ | Token length of the rollout |
| $L$ | Normalizing constant for length (set to 500) |

Combined with PUCT exploration bonus:

$$U(s) = c_{\text{puct}} \frac{\sqrt{\sum_i N(s_i)}}{1 + N(s)}$$

| Term | Meaning |
|---|---|
| $c_{\text{puct}}$ | Exploration constant (set to 0.125) |
| $N(s)$ | Visit count of state $s$ |
| $\sum_i N(s_i)$ | Total visits across sibling states |

Selection: $(s, r) = \arg\max_{(s,r)} [Q(s,r) + U(s)]$

**Concrete example:** State $s$ has MC(s) = 0.9 (high — prefix likely correct). A rollout $r$ from $s$ has wrong answer, length 200 tokens. With $\alpha = 0.5$, $\beta = 0.9$, $L = 500$:

$Q = 0.5^{1-0.9} \cdot 0.9^{200/500} = 0.5^{0.1} \cdot 0.9^{0.4} = 0.933 \cdot 0.959 = 0.895$

High Q — this is a "supposed-to-be-correct wrong-answer" rollout, exactly what we want to investigate.

**2. Binary Search:** On the selected rollout, perform binary search to locate the first error step. All intermediate states and their MC values become new tree nodes.

**3. Maintain:** Update $N(s)$, $\text{MC}(s)$, and $Q(s,r)$ for affected nodes. No recursive backpropagation (simpler than AlphaGo).

### PRM Training

Each edge $(s, a)$ in the tree is a training example: (question, prefix, next step(s), correctness label). Three training objectives compared:

**Pointwise soft label** (best, 70.1% PRM accuracy):

$$\mathcal{L}_{\text{pointwise}} = \sum_{i=1}^{N} \hat{y}_i \log y_i + (1 - \hat{y}_i) \log(1 - y_i)$$

where $\hat{y}_i = \text{MC}(s)$ (soft label = MC estimate).

| Objective | PRM Accuracy |
|---|---|
| Pointwise soft label | **70.1%** |
| Pointwise hard label ($\hat{y} = \mathbf{1}[\text{MC} > 0]$) | 63.3% |
| Pairwise (Bradley-Terry) | 64.2% |

### Inference: Weighted Self-Consistency

At test time, generate $k$ solutions, score each step with the PRM, compute solution-level score as product of step scores (following Lightman), then use PRM-weighted majority voting.

## Results

### Main Results (PRM-weighted majority voting, k=64)

| PRM Training Data | MATH500 (Gemini Pro) | MATH500 (Gemma2 27B) | GSM8K (Gemini Pro) | GSM8K (Gemma2 27B) |
|---|---|---|---|---|
| MajorityVote@64 | 67.2 | 54.7 | 92.7 | 90.6 |
| + Math-Shepherd | 67.2 | 57.4 | 92.7 | 90.5 |
| + PRM800K | 67.6 | 57.2 | 92.9 | 91.7 |
| **+ OmegaPRM** | **69.4** | **58.2** | **93.6** | **92.2** |

OmegaPRM-trained PRM beats both human-annotated PRM800K and automatic Math-Shepherd across all settings.

### Algorithm Efficiency

| Method | Data Points (same compute budget) | Efficiency |
|---|---|---|
| Brute-force MC | 200K | 1x |
| **OmegaPRM** | **15M** | **75x** |

In practice, they downsample to 1.5M for PRM training.

### Data Scale

- 12K MATH training questions, 100 search iterations per question
- 1.5M per-step process supervision annotations
- $k = 8$ rollouts per MC estimation
- $\alpha = 0.5$, $\beta = 0.9$, $L = 500$, $c_{\text{puct}} = 0.125$

### Step Distribution

- Flexible step splitting (divide solution by average length / 16) produces step length distribution similar to rule-based methods
- Most solutions have < 20 steps
- Semantically explicit step cutting is not necessary for training a PRM

## Key Takeaways

1. **Automated process supervision can beat human labels.** OmegaPRM-trained PRMs outperform PRM800K-trained ones despite zero human annotation. The key is data quantity and quality balance — 1.5M automated labels > 800K human labels.

2. **Binary search + MCTS gives 75x efficiency over brute-force MC.** By focusing rollouts on locating the first error (binary search) and reusing rollouts across searches (tree structure), compute is dramatically reduced.

3. **Prioritize "supposed-to-be-correct wrong-answer" rollouts.** The $\alpha^{1-\text{MC}(s)}$ term in the selection function surfaces states where the prefix is correct but the rollout went wrong — these contain the most informative error examples. Directly mirrors Lightman's active learning insight.

4. **Soft labels > hard labels for PRM training.** Using MC(s) as a continuous label (70.1%) significantly outperforms binary thresholding (63.3%) or pairwise ranking (64.2%).

5. **You only need the first error.** Following Lightman's insight, binary search targets the first incorrect step. This is sufficient for PRM training — no need to annotate every step.

6. **No human supervision needed, but gold answers are required.** The method is fully automated but requires (question, gold answer) pairs to compute MC estimates. Not applicable to open-ended tasks without verifiable answers.

7. **Step boundaries don't need to be semantically precise.** Dividing by average token length works as well as rule-based step splitting. This is relevant for our semantic boundary framework — strict boundary detection may not be critical for PRM training.

## Connection to Our Work

OmegaPRM solves the data collection problem for training PRMs at scale without human labels. For our `<verify>` pivot:

- **We can generate our own process supervision data** using OmegaPRM on Qwen3-1.7B rollouts on MATH train — no need to rely on PRM800K (which was generated with GPT-4)
- **The binary search + MCTS pipeline can run on our 2×A6000 setup** — it only needs the policy model for rollouts and answer checking, no separate reward model during data collection
- **The 75x efficiency gain matters for us** — with limited compute, brute-force MC estimation over 7.5K MATH problems would be expensive; OmegaPRM makes it feasible
- **Soft MC labels can train a small PRM** that serves as the verification signal when the model emits `<verify>` — this is the missing piece between Lightman's concept and a practical implementation
