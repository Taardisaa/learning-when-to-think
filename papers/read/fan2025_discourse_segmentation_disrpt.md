# HITS at DISRPT 2025: Discourse Segmentation, Connective Detection, and Relation Classification

**Authors:** Yi Fan, Banerjee Souvik, Michael Strube (Heidelberg Institute for Theoretical Studies)

**Date:** 2025 | **Venue:** DISRPT 2025 (4th Shared Task on Discourse Relation Parsing and Treebanking), ACL Workshop

## Problem

The DISRPT 2025 shared task poses three multilingual discourse processing challenges under a strict constraint: a single model per sub-task, ≤4B parameters (closed track):

1. **Discourse unit segmentation** — partition text into elementary discourse units (EDUs) across formalisms (RST, SDRT, etc.) and 16 languages. Key difficulty: massive class imbalance (boundary vs. non-boundary tags), plus cross-formalism annotation differences.
2. **Discourse connective identification** — locate explicit words/phrases (e.g., "but", "because", "on the other hand") that signal relations between text spans. Annotated under PDTB and ISO formalisms. Challenge: linguistic diversity of connectives + structural differences between annotation schemes.
3. **Discourse relation classification** — classify the relation type (from 17 unified labels) between two discourse units across formalisms. Hardest sub-task: implicit connectives, cross-lingual transfer, and cross-formalism generalization.

Each sub-task requires one multilingual model serving all languages — no per-language models allowed.

## Key Concepts

**Elementary Discourse Unit (EDU):** The smallest text segment that carries a single proposition or rhetorical function — typically a clause. Not a token or a sentence, but a semantic unit between those granularities.

```
[I went to the store]_EDU1 [because I needed milk]_EDU2 [but it was closed.]_EDU3
```
This sentence has many tokens but only 3 EDUs. Each EDU is one atomic reasoning/narrative step. The boundaries between EDUs are what Task 1 tries to detect.

| Level | Granularity | Example |
|-------|-------------|---------|
| Token | Sub-word / word | "I", "went", "store" |
| EDU | Clause-level semantic unit | "I went to the store" |
| Sentence | One or more EDUs | "I went to the store because I needed milk." |

**Discourse Connective:** An explicit word or phrase that signals a rhetorical relation between two text spans — e.g., "but" (contrast), "because" (cause), "on the other hand" (comparison). Task 2 identifies these. Not all relations have explicit connectives (implicit relations are harder).

**Discourse Relation:** The rhetorical/semantic relationship between two discourse units. The 2025 task uses 17 unified labels across formalisms. Examples: *Cause*, *Contrast*, *Elaboration*, *Condition*. Task 3 classifies these.

**Discourse Formalisms:** Different theoretical frameworks for annotating discourse structure. The main ones in this task:
- **RST** (Rhetorical Structure Theory) — tree-structured, hierarchical relations between EDUs
- **SDRT** (Segmented Discourse Representation Theory) — graph-based, allows crossing dependencies
- **PDTB** (Penn Discourse Treebank) — flat, relation-level annotations anchored to connectives
- **ISO** — standardized framework for discourse relations

The shared task requires models to work across all formalisms simultaneously.

## Core Design / Method

### Task 1: Discourse Segmentation — mT5-XL + LoRA + FGM

**Base model:** google/mt5-xl (3.7B parameters), fine-tuned with LoRA.

**Training strategy:** Multilingual joint fine-tuning on all languages combined. They compared three configurations (all languages joint, monolingual per-language, and group-specific with Chinese separated) — the group-specific split (Chinese vs. rest) performed best.

**Weighted cross-entropy** to handle Seg=O (non-boundary) dominance:

$$w_c = \frac{N}{C \times N_c}$$

| Term | Meaning |
|------|---------|
| $w_c$ | Weight for class $c$ |
| $N$ | Total number of tokens in training set |
| $C$ | Total number of unique classes (2: boundary/non-boundary) |
| $N_c$ | Count of occurrences of class $c$ |

**Concrete example:** If a training set has 100,000 tokens, with 95,000 Seg=O and 5,000 Seg=B-seg:
- $w_O = 100{,}000 / (2 \times 95{,}000) = 0.526$
- $w_{B} = 100{,}000 / (2 \times 5{,}000) = 10.0$

The boundary class gets ~19x more weight than the non-boundary class.

**Fast Gradient Method (FGM)** for adversarial training — adds perturbation to embedding layer each step:

$$r_{\text{adv}} = \epsilon \frac{\nabla_{\theta_{\text{emb}}} L(\theta)}{\|\nabla_{\theta_{\text{emb}}} L(\theta)\|_2}$$

| Term | Meaning |
|------|---------|
| $r_{\text{adv}}$ | Adversarial perturbation added to embeddings |
| $\epsilon$ | Hyperparameter controlling perturbation magnitude |
| $\theta_{\text{emb}}$ | Embedding layer parameters |
| $L(\theta)$ | Loss function |
| $\nabla_{\theta_{\text{emb}}}$ | Gradient of loss w.r.t. embedding parameters |

The model does a standard forward+backward pass, computes $r_{\text{adv}}$, adds it to embeddings, then does a second forward+backward to compute adversarial loss. This smooths decision boundaries.

### Task 2: Connective Identification — 3-Encoder Ensemble + CRF

**Architecture:** Ensemble of three heterogeneous encoders whose representations are fused via multi-head attention:

| Encoder | Size | Rationale |
|---------|------|-----------|
| RemBERT | — | Strong cross-lingual transfer |
| XLM-RoBERTa (Large) | — | Proven multilingual performance |
| mDeBERTa-v3 (Base) | — | Improved disentangled attention |

Each encoder produces $H_i \in \mathbb{R}^{L \times D_i}$. Three fusion strategies were tested:

1. **Concatenation:** $H_{\text{fused}} = [H_1, H_2, \ldots, H_N]$
2. **Weighted fusion:** $H_{\text{fused}} = \sum_{i=1}^{N} \text{softmax}(\mathbf{w})_i \cdot \text{Linear}_i(H_i)$
3. **Attention fusion (chosen):** Multi-head attention over concatenated hidden states — dynamically learns token-level combinations.

**Linguistic features:** POS tags and dependency relations from CoNLL-U annotations are embedded via separate layers ($E_{\text{pos}}$, $E_{\text{dep}}$), concatenated, passed through a linear+ReLU layer, then concatenated with the fused encoder output.

**CRF layer** on top enforces valid label sequences (e.g., I-conn must follow B-conn) via Viterbi decoding.

**Hybrid loss:** CRF loss + Focal Loss + Label Smoothing.

Focal Loss to handle O/B-conn/I-conn imbalance:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

| Term | Meaning |
|------|---------|
| $p_t$ | Model's predicted probability for the correct class |
| $\alpha_t$ | Class-balancing weight |
| $\gamma$ | Focusing parameter (down-weights easy examples) |

### Task 3: Relation Classification — Two-Stage RECL (Rationale-Enhanced Curriculum Learning)

**Student model:** google/gemma-2-2b-it (2B parameters), fine-tuned with LoRA. Formulated as a generative task: given two text units, context sentence, direction, and the 17 labels, output `{"label": "classification"}`.

**Stage 1 — Initial SFT:** Standard LoRA fine-tuning on all training data combined across languages.

**Stage 2 — Rationale-Enhanced Curriculum Learning** (three phases):

1. **Hard-sample mining:** Run Stage 1 model on training set; samples it gets wrong become "hard samples."
2. **Knowledge distillation:** Prompt a teacher model (Qwen2.5-72B-Instruct) to generate Chain-of-Thought rationales for each hard sample — explaining *why* a specific label is correct, citing linguistic evidence. 4 handwritten rationales serve as few-shot examples. Generation done via vLLM for throughput.
3. **Weighted fine-tuning (curriculum):** Retrain from Stage 1 weights on a mix of:
   - Hard samples with CoT rationales (loss weight = **1.5**)
   - Easy samples with original simple prompt (loss weight = **1.0**)

This curriculum forces the model to prioritize learning from its mistakes while replaying easy samples to prevent catastrophic forgetting.

## Results

### Task 1: Discourse Segmentation (F1, test set)

| Corpus | F1 | Corpus | F1 |
|--------|-----|--------|-----|
| nld.rst.nldt | 97.47 | eng.rst.gentle | 92.00 |
| eng.rst.rstdt | 97.40 | eus.rst.ert | 90.97 |
| eng.sdrt.msdc | 95.64 | zho.rst.gcdt | 90.90 |
| por.rst.cstn | 95.64 | eng.rst.umuc | 88.21 |
| eng.dep.scidtb | 95.08 | fra.sdrt.annodis | 88.06 |
| deu.rst.pcc | 94.52 | zho.dep.scidtb | 87.83 |
| fra.sdrt.summre | 65.04 | zho.rst.sctb | 73.24 |
| **Mean** | | | **90.09** |

Weakest performance on fra.sdrt.summre (multi-party meeting dialogues — spoken language with disfluencies, non-standard punctuation) and zho.rst.sctb (data imbalance + genre diversity).

### Task 2: Connective Identification (F1, test set)

| Corpus | F1 | Corpus | F1 |
|--------|-----|--------|-----|
| eng.pdtb.pdtb | 93.15 | deu.pdtb.pcc | 79.37 |
| tur.pdtb.tdb | 93.07 | por.pdtb.tedm | 78.38 |
| eng.pdtb.gentle | 89.20 | eng.pdtb.tedm | 78.18 |
| ita.pdtb.luna | 70.81 | tur.pdtb.tedm | 65.80 |
| **Mean** | | | **81.00** |

TED talk corpora (tedm) consistently score low — they are test-only (no training data), revealing poor cross-corpus generalization.

### Task 3: Relation Classification (Accuracy, test set)

| Corpus | Acc | Corpus | Acc |
|--------|------|--------|------|
| tha.pdtb.tdtb | 93.97 | eng.pdtb.tedm | 66.38 |
| eng.sdrt.msdc | 88.90 | nld.rst.nldt | 64.92 |
| eng.dep.scidtb | 81.41 | eng.rst.rstdt | 64.64 |
| eng.pdtb.pdtb | 79.32 | eng.rst.sts | 52.74 |
| ces.rst.crdt | 52.03 | eus.rst.ert | 53.20 |
| **Macro Average** | | | **66.78** |
| **Micro Average** | | | **72.24** |

Stage 2 (curriculum + rationales) gave a **2.12% micro average improvement** over Stage 1 alone. Scores are generally low — highlights the difficulty of cross-formalism, cross-lingual relation classification.

## Key Takeaways

1. **Language grouping matters for segmentation.** Splitting Chinese into its own group and training a separate model for it (vs. all-languages-jointly) improved Task 1 performance. Typological distance affects transfer quality.

2. **Adversarial training (FGM) helps segmentation but not connective identification.** Adding FGM to Task 1 improved robustness; trying it for Task 2 had negligible effect. The benefit is task-dependent.

3. **Encoder ensembles with attention fusion outperform single encoders for token-level tasks.** The 3-encoder ensemble with multi-head attention fusion was the strongest configuration for connective identification, beating concatenation and weighted fusion.

4. **Linguistic features (POS + dependency) improve connective detection.** Injecting explicit syntactic information helps the model, consistent with the intuition that connective identification is fundamentally a linguistic/syntactic task.

5. **CRF layers enforce structural constraints in sequence labeling.** Ensuring valid BIO tag sequences (I-conn must follow B-conn) via a CRF layer prevents structurally invalid predictions.

6. **Knowledge distillation via CoT rationales provides modest gains (+2.12%) for relation classification.** The two-stage curriculum (SFT then rationale-enhanced fine-tuning on hard samples) helps, but the improvement is smaller than hoped — the student may not fully internalize the teacher's reasoning. Future work: task vectors or representation-space interventions to force internalization.

7. **Spoken/conversational text is a major failure mode.** Discourse models trained on written text (news, academic articles) struggle with spoken language (meeting dialogues, TED talks) due to disfluencies, non-standard punctuation, and informal structure.

8. **Test-only corpora expose generalization limits.** Corpora with no training split (TED talk datasets) consistently score lowest, showing these models rely heavily on in-domain data and don't generalize well from related corpora.

## Connection to Our Work

The discourse segmentation aspect is directly relevant to **semantic boundary detection** in our backoff-token project. The paper's finding that discourse boundaries can be identified with high F1 (90+%) using token-level classification suggests that similar approaches could potentially be used to automatically detect semantic boundaries in CoT reasoning — the points where `<backoff_N>` tokens should be placed.

The three boundary detection categories from our `02_method.md` (punctuation, logical connectives, structural markers) align with what this paper targets: discourse connectives and segment boundaries. Their CRF-based approach to enforcing valid sequences is analogous to our requirement that backoff tokens must be placed at valid semantic boundaries, not mid-sentence.

The knowledge distillation framework (Stage 2 of Task 3) mirrors our own SFT approach: using a larger teacher model to generate corrective reasoning for samples the student gets wrong, then training the student on these enhanced examples. The modest 2.12% gain they observe is a useful calibration point.
