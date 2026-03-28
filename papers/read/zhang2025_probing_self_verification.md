# Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification

**Authors:** Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He (NYU & NYU Shanghai)

**Date:** 2025 | **Venue:** arXiv:2504.05419 (preprint, under review)

## Problem

Reasoning models (DeepSeek-R1, QwQ, o1) achieve strong performance via search-based reasoning — exploring multiple reasoning paths with intermediate answers before converging. But they **overthink**: they keep exploring even after finding a correct answer, wasting compute.

The question: **do reasoning models internally "know" when an intermediate answer is correct?** If so, why don't they stop? And can we extract this signal to build an early-exit strategy?

| Behavior | What happens | The waste |
|----------|-------------|-----------|
| Overthinking | Model finds correct answer at chunk 3, continues to chunk 8 | 60%+ of tokens are unnecessary |
| Self-verification gap | Hidden states encode "this answer is correct" but model doesn't act on it | Internal knowledge ≠ external behavior |

## Key Concepts

**Intermediate answer:** During long CoT reasoning (inside `<think>` tokens), reasoning models produce multiple candidate answers along different reasoning paths. Each time the model writes something like "so the answer is 5" or "therefore x = 3", that's an intermediate answer. The final answer is the last one before `</think>`.

**Chunk:** A segment of the reasoning trace corresponding to one reasoning path, ending with an intermediate answer. The paper splits the `<think>` content into paragraphs (by `\n\n`), detects path boundaries via keywords ("wait", "alternatively", "double-check", "let me reconsider"), and merges paragraphs within the same path into one chunk. Each chunk has one intermediate answer extracted by Gemini 2.0 Flash.

```
<think>
[Chunk 1: "Okay so John has 13 toys... he should have none left, right?"]  → answer: 0 (wrong)
[Chunk 2: "Wait, let me go back... So, he sold 11 and got 165..."]          → answer: 0 (wrong)
[Chunk 3: "Let me make sure... So, he sold 11, has 2 left. The answer is 2"] → answer: 2 (correct)
[Chunk 4: "Alternatively, let me verify... He still has 2 LEGO sets left."]  → answer: 2 (correct)
</think>
```

Chunks 3 and 4 are both correct, but the model kept going — this is overthinking.

**Reasoning model vs. non-reasoning model:** Reasoning models (R1-Distill series, QwQ) are trained on long CoT data and/or with RL to produce extended reasoning traces. Non-reasoning models (Llama-3.1-8B-Instruct) produce short direct answers. The paper shows that the self-verification signal is **much stronger in reasoning models**, suggesting it emerges from long-CoT training.

## Core Design / Method

### Step 1: Data Collection

1. Run reasoning model on math/logic problems, collect full `<think>` traces
2. Split traces into chunks using keyword-based paragraph boundary detection
3. Use Gemini 2.0 Flash to extract the intermediate answer from each chunk
4. Label each chunk as correct/incorrect by comparing intermediate answer to gold answer
5. For each chunk $c_i$, extract the **last-layer hidden state at the last token position** as representation $e_i$

Result: probing dataset $\mathcal{D} = \{(e_i, y_i)\}_{i=1}^N$ where $y_i \in \{0, 1\}$ is correctness.

### Step 2: Train Probe

A 2-layer MLP with weighted binary cross-entropy:

$$p_i = \sigma(\text{ReLU}(e_i \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + b_2)$$

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^{N} (w \alpha y_i \log p_i + (1 - y_i) \log(1 - p_i))$$

| Term | Meaning |
|------|---------|
| $e_i \in \mathbb{R}^m$ | Last-layer hidden state at last token of chunk $c_i$ ($m$ = model hidden dim) |
| $\mathbf{W}_1 \in \mathbb{R}^{m \times d}$ | First layer weights ($d$ = probe hidden dim) |
| $\mathbf{W}_2 \in \mathbb{R}^{d \times 1}$ | Second layer weights |
| $w$ | Ratio of negative to positive samples (class imbalance correction) |
| $\alpha$ | Scaling factor for imbalance weight (hyperparameter) |
| $p_i$ | Predicted probability that intermediate answer in chunk $i$ is correct |

**Key finding from grid search:** Most probes converge to $d = 0$ (i.e., a **linear probe**), meaning correctness is linearly encoded in the hidden states. The MLP architecture is there for safety but often unnecessary.

### Step 3: Early-Exit Strategy

Use the trained probe as a **verifier** during inference:

1. Set confidence threshold $Thr$
2. As the model generates, evaluate each chunk's intermediate answer
3. When probe confidence $p_i > Thr$, stop generation and output that intermediate answer
4. If no chunk exceeds threshold, let the model complete naturally

### Models Tested

| Model | Params | Type |
|-------|--------|------|
| R1-Distill-Llama-8B | 8B | Reasoning (distilled from DeepSeek-R1) |
| R1-Distill-Llama-70B | 70B | Reasoning (distilled) |
| R1-Distill-Qwen-1.5B | 1.5B | Reasoning (distilled) |
| R1-Distill-Qwen-7B | 7B | Reasoning (distilled) |
| R1-Distill-Qwen-32B | 32B | Reasoning (distilled) |
| QwQ-32B | 32B | Reasoning (RL-trained) |
| Llama-3.1-8B-Instruct | 8B | Non-reasoning (baseline) |

### Datasets

| Dataset | Domain | Size (train) |
|---------|--------|-------------|
| GSM8K | Grade school math | ≤1000 |
| MATH | Competition math | ≤1000 |
| AIME | AMC/AIME competition | ≤1000 |
| KnowLogic | Logical reasoning (MC) | ≤1000 |

## Results

### In-Distribution: Probes detect correctness with high accuracy

ROC-AUC scores across models and datasets (Figure 2):

| Model | GSM8K | MATH | AIME | KnowLogic |
|-------|-------|------|------|-----------|
| R1-Distill-Llama-8B | ~0.82 | ~0.84 | ~0.76 | ~0.67 |
| R1-Distill-Qwen-1.5B | ~0.78 | ~0.78 | ~0.72 | ~0.65 |
| R1-Distill-Qwen-7B | ~0.82 | ~0.82 | ~0.78 | ~0.72 |
| R1-Distill-Qwen-32B | ~0.85 | ~0.88 | **~0.92** | ~0.75 |
| R1-Distill-Llama-70B | ~0.83 | ~0.85 | ~0.80 | ~0.70 |
| QwQ-32B | ~0.85 | ~0.87 | ~0.85 | ~0.78 |

All probes achieve ROC-AUC > 0.7, most > 0.8. ECE (calibration error) is consistently below 0.1, often below 0.05 — the probe's confidence scores are **well-calibrated**.

Larger models encode stronger correctness signals. Qwen family > Llama family (possibly reflecting training data distribution). Math probes > logic probes.

### Calibration Quality (Table 1, ECE and Brier Score)

| Model | GSM8K ECE | MATH ECE | AIME ECE | KnowLogic ECE |
|-------|-----------|----------|----------|---------------|
| R1-Distill-Llama-8B | 0.05 | 0.03 | 0.10 | 0.07 |
| R1-Distill-Qwen-32B | 0.01 | 0.08 | 0.09 | 0.10 |
| QwQ-32B | 0.03 | 0.13 | 0.08 | 0.03 |

ECE < 0.1 across the board — when the probe says 80% confident, it's right ~80% of the time. This is critical for using it as a verifier.

### Cross-Dataset Generalization (Table 2, R1-Distill-Llama-8B)

| Train → Eval | GSM8K AUC | MATH AUC | AIME AUC | KnowLogic AUC |
|-------------|-----------|----------|----------|---------------|
| GSM8K | **0.82** | 0.80 | 0.69 | 0.56 |
| MATH | 0.83 | **0.84** | 0.76 | 0.63 |
| KnowLogic | 0.77 | 0.74 | 0.81 | **0.67** |

Math ↔ math transfer works well (GSM8K ↔ MATH: ~0.80-0.83). Math → logic transfer is weaker (~0.56-0.63). AIME is hardest to generalize to from easier datasets.

### Reasoning Models vs. Non-Reasoning Models (Figure 3, MATH)

| Model | Accuracy | ROC-AUC | ECE |
|-------|----------|---------|-----|
| R1-Distill-Llama-8B (intermediate) | **0.92** | **0.85** | 0.03 |
| R1-Distill-Llama-8B (final) | 0.80 | 0.86 | 0.03 |
| Llama-3.1-8B-Instruct (final) | 0.66 | 0.82 | **0.23** |

The self-verification signal is **much stronger in reasoning models** (0.92 accuracy, 0.03 ECE) than in non-reasoning models (0.66 accuracy, 0.23 ECE). This suggests long-CoT training enhances the model's internal self-verification ability.

### Look-Ahead: Correctness encoded before the answer is written (Figure 4)

Probing hidden states at different positions *within* a chunk (before the intermediate answer is generated):

| Position in chunk | ROC-AUC | ECE |
|-------------------|---------|-----|
| 0% (start of chunk) | ~0.60 | ~0.15 |
| 10% | ~0.72 | ~0.08 |
| 50% | ~0.77 | ~0.04 |
| 90% | ~0.83 | ~0.03 |
| 100% (at the answer) | ~0.85 | ~0.03 |

Two critical phases: a steep improvement from 0-10% (initial reasoning steps), then gradual improvement with a second bump at 90-100% (near the answer). The model "knows" the answer will be correct well before it writes it.

### Early-Exit Results (Figure 5, R1-Distill-Llama-8B on MATH)

| Strategy | Threshold / Chunks | Accuracy | Token Savings |
|----------|-------------------|----------|---------------|
| No early-exit | — | 88.6% | 0% |
| Confidence-based | $Thr = 0.85$ | **88.2%** | **24%** |
| Confidence-based | $Thr = 0.90$ | 88.6% | 19% |
| Confidence-based | $Thr = 0.80$ | 87.4% | ~30% |
| Static (stop after chunk $m$) | $m = 5$ | ~85% | ~20% |
| Static | $m = 3$ | ~82% | ~35% |

At $Thr = 0.85$: **24% token reduction with essentially no accuracy loss** (88.2% vs. 88.6%). The confidence-based strategy consistently outperforms static early-exit by ~5% accuracy at similar token budgets.

## Key Takeaways

1. **Reasoning models encode intermediate answer correctness in their hidden states, and it's linearly separable.** A simple probe (often just a linear classifier) achieves ROC-AUC > 0.8 with ECE < 0.1. The signal is there and well-calibrated.

2. **The signal is stronger in reasoning models than non-reasoning models.** Long-CoT training (distillation or RL) appears to enhance the model's internal self-verification ability. Llama-3.1-8B-Instruct has the signal but weaker and poorly calibrated (ECE = 0.23 vs. 0.03).

3. **Correctness is encoded before the answer is fully generated.** Hidden states at just 10% into a reasoning chunk already have ROC-AUC ~0.72 for predicting the chunk's answer correctness. The model "knows" early.

4. **Models don't utilize their own correctness knowledge.** Despite encoding high-confidence correctness signals, models continue reasoning unnecessarily (overthinking). A probe-based verifier saves 24% of tokens with no accuracy loss — proving the model could have stopped but didn't.

5. **Cross-domain generalization is limited.** Math → math probes transfer well, but math → logic probes degrade significantly. The correctness signal is somewhat domain-specific.

6. **Larger models encode stronger signals.** R1-Distill-Qwen-32B reaches ROC-AUC > 0.9 on AIME, while the 1.5B model stays around 0.72. Model scale matters for self-verification.

7. **The probe is extremely lightweight.** Often converges to a linear classifier ($d = 0$). Training on ≤1000 examples is sufficient. This is a practical, deployable verifier.

## Connection to Our Work

This paper is **the most directly relevant** of the four structural syntax papers to our backoff project.

**Their chunks ≈ our chunks.** Their method of splitting reasoning traces at keywords like "wait", "alternatively", "let me reconsider" maps almost exactly onto our semantic boundary detection heuristics (logical connectives / discourse markers from `02_method.md`). The keywords they use (Table 3 in their appendix) are essentially a subset of our boundary detection rules.

**Their probe detects correctness at the chunk level — exactly what we need.** If we could train a similar probe on our model (Qwen3-1.7B/4B), it could:
- Identify which intermediate answers in a reasoning trace are wrong
- Potentially automate the error-point detection that our subagents currently do manually
- Provide a training signal for when to emit `<backoff_N>`

**Their finding that reasoning models have stronger signals is important for us.** We're training on a base/instruct model (Qwen3), not a reasoning model. The correctness signal may be weaker in our case (more like the 0.66 accuracy / 0.23 ECE they see on Llama-Instruct). But our backoff training might actually *create* this signal — by training the model to emit backoff tokens, we're essentially training it to act on correctness information, which may simultaneously strengthen the internal representation.

**Early-exit vs. backoff — complementary directions.** They use the probe to *stop early* when the answer is correct (save compute on overthinking). We use backoff to *rewind* when the answer is wrong (fix errors). These are two sides of the same coin: both require the model to assess intermediate correctness and act on it. A combined system could both stop early on correct paths AND backtrack on incorrect ones.

**Key gap:** Their probe verifies correctness of the *final answer* in each chunk. We need to detect errors in the *reasoning steps themselves* — not just "is the answer right?" but "is the reasoning that led to it valid?" A step could produce the right answer by luck. But for practical backoff placement, chunk-level answer correctness is probably sufficient.
