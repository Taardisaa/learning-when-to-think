# Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion

**Authors:** Anum Afzal, Florian Matthes (Technical University of Munich), Gal Chechik (Nvidia Research & Bar-Ilan University), Yftah Ziser (Nvidia Research)

**Date:** 2025 | **Venue:** ACL 2025 Findings

**Code:** [github.com/anum94/CoTpred](https://github.com/anum94/CoTpred)

## Problem

Chain-of-Thought prompting is effective but computationally expensive — it generates many intermediate tokens. Two open questions:

1. Does the LLM already "know" whether CoT will succeed **before generating any reasoning tokens**?
2. If later reasoning steps don't improve this prediction, can we **stop CoT early** without losing accuracy?

Prior work (Azaria & Mitchell, 2023) probed LLM hidden states for truthfulness, but nobody had applied probing to predict CoT success specifically — whether the full reasoning chain will lead to the correct answer.

## Key Concepts

**Probing classifier:** A small neural network trained on frozen LLM hidden states to predict a property of interest. If a simple probe can predict the property accurately, the information is considered to be **encoded** in the LLM's representations. The LLM itself is not fine-tuned — only the probe is trained.

| Aspect | Probing | Fine-tuning |
|--------|---------|-------------|
| LLM weights | Frozen | Updated |
| What's trained | Small classifier on top | LLM parameters |
| Purpose | Detect what's already encoded | Change model behavior |
| Scale | ~10K params | Millions/billions of params |

**SVCCA (Singular Vector Canonical Correlation Analysis):** A method for measuring similarity between two sets of neural network representations. Used here to compare hidden states at different CoT generation steps — if early steps have high SVCCA similarity with the final step, the model has "already computed" the answer early on.

**White-box access:** The probe needs access to the LLM's internal hidden states (not just its text output). This means the approach works only with open-weight models, not API-only models like GPT-4.

## Core Design / Method

### Setup

1. Run LLM on math problems with zero-shot CoT ("Let's think step by step") at temperature=0
2. Label each generation as correct/incorrect by comparing to reference answer
3. Create balanced datasets (50/50 correct/incorrect), 10K train + 1K test per dataset

### Probe Architecture

A 3-layer feedforward network trained on the **last token's hidden state** from a single layer:

$$\text{Probe}(h_L) = \sigma(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot h_L + b_1) + b_2) + b_3)$$

| Term | Meaning |
|------|---------|
| $h_L$ | Hidden state vector from layer $L$ of the LLM (dim=4096) |
| $W_1, W_2, W_3$ | Weight matrices (256→128→64→1) |
| $\sigma$ | Sigmoid activation (binary output: success/failure) |

A separate probe is trained **per layer** to identify which layers encode the most CoT-success information.

**Design choice:** They use the last token's hidden state (not mean-pooled or attention-weighted) because prompt lengths vary. This gives a fixed-size input regardless of sequence length.

### Two Experimental Settings

**1. Before generation (T=0):** Extract hidden states from the initial forward pass of question + CoT prompt, before any reasoning token is generated. Train probe on these states.

**2. Prediction over time (T=10%, 20%, ..., 100%):** Extract hidden states after the LLM has generated 10%, 20%, ... of its full CoT output. This tracks how the model's "knowledge" evolves during generation.

### Baseline: BERT Text Classifier

BERT (bert-base-uncased, 768-dim) reads only the **generated text tokens** — no access to LLM hidden states. This is a black-box baseline: if BERT performs as well as the probe, then the signal is in the text surface, not the internal representations.

### Models and Datasets

| Model | Params | Layers | Hidden dim |
|-------|--------|--------|------------|
| Llama-3.1-8B-Instruct | 8B | 33 | 4096 |
| Mistral-7B-Instruct-v0.3 | 7B | 32 | 4096 |

| Dataset | Domain | Source |
|---------|--------|--------|
| AQuA | Algebraic word problems (multiple-choice) | Ling et al., 2017 |
| Olympiad | International math competitions | NuminaMath-CoT |
| Cn-k12 | Chinese K-12 math (translated to English) | NuminaMath-CoT |

## Results

### Before Generation (T=0): Can the probe predict CoT success before any reasoning?

| Dataset | BERT (text only) | Probe (hidden states) | Best layers |
|---------|------------------|----------------------|-------------|
| **Llama-3.1-8B** | | | |
| AQuA | 53.5% | **60.0%** | 11, 12, 13, **14**, 16 |
| Olympiad | 69.1% | **76.4%** | 8, **14**, **16**, 17, 31 |
| Cn-k12 | 66.2% | **69.1%** | 13, **14**, **16**, 17, 22 |
| **Mistral-7B** | | | |
| AQuA | 60.1% | **64.7%** | 15, 16, **18**, 23, **28** |
| Olympiad | 68.1% | **71.8%** | 7, 9, **18**, 26, **28** |
| Cn-k12 | 65.5% | **67.1%** | 12, 14, **18**, 21, 24 |

Random baseline = 50% (balanced dataset). The probe consistently beats BERT, confirming that **LLM hidden states encode CoT success information that is not accessible from the text surface alone.**

Layers 14 and 16 (Llama) / 18 and 28 (Mistral) are consistently the most informative — middle and late-middle layers.

### Prediction Over Time: Does more CoT context help?

| Dataset | 0% | 10% | 30% | 50% | 70% | 100% |
|---------|-----|------|------|------|------|-------|
| **Probe (Llama-3.1-8B)** | | | | | | |
| AQuA | 60.0 | 51.5 | 60.5 | 60.1 | 63.7 | **69.4** |
| Olympiad | 76.4 | 75.1 | 73.8 | **76.5** | 75.3 | 75.9 |
| Cn-k12 | 69.1 | 67.7 | 69.2 | 67.2 | 70.7 | 70.9 |
| **BERT baseline (Llama-3.1-8B)** | | | | | | |
| AQuA | 53.5 | 54.2 | 54.1 | 53.5 | 53.7 | 50.6 |
| Olympiad | 69.1 | 67.9 | 66.2 | 67.4 | 66.8 | 66.9 |
| Cn-k12 | 66.2 | 57.8 | 64.1 | 64.3 | 61.7 | 62.4 |

Key observations:
- **For Olympiad and Cn-k12, the probe at T=0 is already nearly as good as T=100%.** Later reasoning steps barely improve prediction. The model "knows" early.
- **For AQuA, later steps do help** (60.0% → 69.4%), suggesting some problems require intermediate computation to resolve.
- **BERT gets worse over time** on Llama outputs — longer CoT text confuses BERT's shallow pattern matching, while the probe benefits from richer hidden states.

### SVCCA Analysis: Why don't later steps help?

SVCCA similarity between early representations and the final representation (Figure 3):
- For Olympiad: similarity rises quickly, reaching ~0.9 by 50% generation → early states already resemble the final state
- For AQuA: similarity rises more slowly → early states are less settled, explaining why the probe improves with more context

**Interpretation:** When early representations are already similar to the final step, the model has essentially "completed the computation" internally — the remaining tokens are just externalizing what it already knows.

### Early Stopping Experiment

They halt generation at various points and prompt: *"Stop all computation and give me the correct answer in 2-3 words, if you already know it."*

| Dataset | Gen % | Consistent | Corrected | Overall Correct | Full CoT Correct |
|---------|-------|------------|-----------|-----------------|------------------|
| AQuA | 50 | 40/100 | 15/100 | 37/100 | 38/100 |
| AQuA | 99 | 81/100 | 5/100 | 59/100 | 38/100 |
| Olympiad | 50 | 28/100 | 2/100 | 22/100 | 19/100 |
| Cn-k12 | 50 | 42/100 | 7/100 | 33/100 | 24/100 |

- "Consistent" = early-stopped answer matches full CoT answer
- "Corrected" = early stopping actually produces a **better** answer than full CoT
- Even at 50% generation, early stopping sometimes outperforms no-CoT baseline
- But zero-shot early stopping is brittle (low consistency) — the model isn't trained to stop early, so it often gives inconsistent answers

## Key Takeaways

1. **LLMs encode CoT success in their hidden states before generating a single reasoning token.** A simple 3-layer probe achieves 60-76% accuracy at predicting whether CoT will succeed, using only the pre-generation hidden state. This is well above the 50% random baseline.

2. **The signal is in the representations, not the text.** The probe (using hidden states) consistently outperforms BERT (using generated text), especially at detecting true negatives (predicting failure). The LLM's internal states capture information about intermediate computations that surface-level text patterns cannot.

3. **Later reasoning steps don't always improve predictability.** For 2 of 3 datasets (Olympiad, Cn-k12), the T=0 probe is nearly as accurate as the T=100% probe. The model "already knows" before it starts writing.

4. **Layers 14-16 (Llama) and 18-28 (Mistral) are the most informative.** Middle-to-late layers encode the most CoT-success information, consistent with prior findings that middle layers build semantic/reasoning representations.

5. **SVCCA confirms early representations resemble final ones.** When the probe doesn't improve over time, it's because early hidden states are already similar to the final hidden state — the model has internally "finished computing" before the text catches up.

6. **Zero-shot early stopping is promising but insufficient.** Halting CoT and asking for the answer sometimes works (and occasionally even corrects errors), but the approach is inconsistent without training. The authors suggest SFT or RL to teach models to stop early more reliably.

7. **The probe is tiny and cheap.** Three linear layers (256→128→64→1) on a 4096-dim hidden state. Training takes 5 epochs on 10K examples. This is negligible compared to the LLM itself.

## Connection to Our Work

This paper is **directly relevant** to the learning-when-to-think project in several ways:

**The core finding validates our approach.** The model's hidden states encode correctness signals that the model itself fails to act on. Our `<backoff_N>` training is essentially teaching the model to **convert these latent signals into explicit behavior** — emitting a backoff token when the hidden state indicates the current reasoning path is wrong.

**Probe as a training signal.** Their probe architecture (3-layer MLP on frozen hidden states) could potentially be used as an auxiliary training signal. Instead of relying solely on subagent-generated backoff data, we could train a correctness probe and use its predictions to identify where in the CoT the model "knows" it's going wrong — automating the error-point detection that our subagents currently do manually.

**Layer selection matters.** Their finding that layers 14-16 are most informative for CoT success could guide where we attach auxiliary heads if we ever move to a probe-guided approach.

**Early stopping ↔ backoff.** Their early stopping experiment is conceptually the inverse of our backoff mechanism. They ask "can we stop early when the model already knows the answer?" We ask "can the model backtrack when it knows the answer is wrong?" Both require the model to act on internal correctness signals.

**Limitation for us:** They only probe binary success/failure of the entire CoT. We need something finer-grained — detecting **which specific step** is wrong and how far to backtrack. Their work shows the signal exists; we need it at step-level granularity, not sequence-level.
