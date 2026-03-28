# A Structural Probe for Finding Syntax in Word Representations

**Authors:** John Hewitt, Christopher D. Manning (Stanford)
**Date:** 2019 | **Venue:** NAACL-HLT 2019

## Problem

Prior probing methods test whether representations encode *individual* linguistic properties (POS tags, morphology, sentence length) but not whether **entire syntax trees** are embedded in the geometry of a model's representation space. Hewitt & Manning ask: do deep models like ELMo and BERT embed full parse trees as a structural property of their vector spaces — recoverable via a simple linear transformation?

---

## Core Idea: Two Structural Probes

The key insight is that syntax trees define two natural quantities on words: **pairwise distances** (path length between words in the tree) and **depth** (distance from root). If a linear transformation of the representation space recreates these quantities, then the model has implicitly learned syntax as a geometric property.

### Probe 1: Parse Distance Probe

**Claim:** There exists a linear transformation $B$ such that the squared distance between transformed word vectors approximates the tree distance between the words.

$$d_B(\mathbf{h}_i^\ell, \mathbf{h}_j^\ell)^2 = (B(\mathbf{h}_i^\ell - \mathbf{h}_j^\ell))^T (B(\mathbf{h}_i^\ell - \mathbf{h}_j^\ell))$$

| Term | Meaning |
|---|---|
| $\mathbf{h}_i^\ell$ | Contextual word representation of word $i$ at layer $\ell$ |
| $B \in \mathbb{R}^{k \times m}$ | Learned linear transformation ($k$ = probe rank, $m$ = representation dim) |
| $B(\mathbf{h}_i^\ell - \mathbf{h}_j^\ell)$ | Difference vector projected into the syntactic subspace |
| $d_B(\cdot,\cdot)^2$ | Squared L2 distance in transformed space — should approximate tree distance $d_T(w_i, w_j)$ |

Equivalently, $d_B^2$ defines an inner product via $A = B^T B$ where $A$ is positive semi-definite. This means the probe finds a **metric on the original space** (not a new space) under which syntax trees emerge.

**Why the probe must be linear:**
The probe is deliberately restricted to a linear transformation. A linear map can only rotate, scale, and project — it cannot create new features. So if a linear probe recovers parse trees from a model's representations, the syntactic structure **must already be encoded in the geometry of that space**; the probe merely isolates the subspace where it lives. If the probe were nonlinear (e.g., an MLP), it would have enough capacity to *compute* syntactic relations from weak signals on its own, making it impossible to tell whether the high score comes from the model's representations or from the probe itself. The paper's baselines validate this: the same linear probe on ELMo0 (context-free character embeddings) gets only 26.8 UUAS — the probe cannot manufacture syntax from representations that don't already encode it.

**Training objective:**

$$\min_B \sum_\ell \frac{1}{|s^\ell|^2} \sum_{i,j} \left| d_{T^\ell}(w_i^\ell, w_j^\ell) - d_B(\mathbf{h}_i^\ell, \mathbf{h}_j^\ell)^2 \right|$$

| Term | Meaning |
|---|---|
| $d_{T^\ell}(w_i^\ell, w_j^\ell)$ | Gold parse tree distance (# edges on path between words $i$ and $j$), derived from human-annotated Penn Treebank constituency trees converted to dependency trees via Stanford Dependencies (de Marneffe et al., 2006) |
| $\|s^\ell\|^2$ | Normalization by sentence length squared (since there are $O(n^2)$ word pairs) |

Minimizes L1 loss between predicted squared distances and gold tree distances across all word pairs in all sentences.

**Concrete example:**

Sentence: *"The cat sat on the mat"*

Parse tree (dependency):
```
      sat
     / | \
   cat  on  .
   |    |
  The  mat
        |
       the
```

Gold tree distances (# edges on shortest path):

| | The | cat | sat | on | the | mat |
|---|---|---|---|---|---|---|
| **The** | 0 | 1 | 2 | 3 | 4 | 3 |
| **cat** | 1 | 0 | 1 | 2 | 3 | 2 |
| **sat** | 2 | 1 | 0 | 1 | 2 | 1 |

The probe learns $B$ such that $d_B(\mathbf{h}_\text{cat}, \mathbf{h}_\text{sat})^2 \approx 1$ and $d_B(\mathbf{h}_\text{The}, \mathbf{h}_\text{mat})^2 \approx 3$, etc. A minimum spanning tree on these predicted distances recovers the parse tree.

### Probe 2: Parse Depth (Norm) Probe

**Claim:** There exists a linear transformation $B$ such that the squared norm of the transformed word vector approximates the word's depth in the parse tree.

$$\|\mathbf{h}_i\|_B^2 = (B\mathbf{h}_i)^T(B\mathbf{h}_i)$$

| Term | Meaning |
|---|---|
| $\|\mathbf{h}_i\|_B^2$ | Squared L2 norm of word $i$'s representation after transformation — should approximate parse depth $\|w_i\|$ |
| $\|w_i\|$ | Gold parse depth = number of edges from word $i$ to root |

**Concrete example** (same sentence):

| Word | The | cat | sat | on | the | mat |
|---|---|---|---|---|---|---|
| **Depth** | 2 | 1 | 0 | 1 | 3 | 2 |

The probe learns $B$ such that $\|B\mathbf{h}_\text{sat}\|^2 \approx 0$ (root), $\|B\mathbf{h}_\text{cat}\|^2 \approx 1$, $\|B\mathbf{h}_\text{the(2)}\|^2 \approx 3$ (deepest), etc. This norm ordering recovers the hierarchical structure: root has smallest norm, leaves have largest.

---

## Why Squared Distances/Norms?

The authors found empirically that approximating **squared** vector quantities works substantially better than raw distances. Squared distance doesn't obey the triangle inequality, but it encodes the same graph structure (same tree is recovered). The authors speculate this relates to gradient properties — squared distance loss has smoother gradients for matching exact integer tree distances.

---

## Evaluation Metrics

### Distance probe

| Metric | Definition |
|---|---|
| **UUAS** | Undirected Unlabeled Attachment Score — % of edges in the minimum spanning tree on predicted distances that match gold parse edges |
| **DSpr.** | Average Spearman correlation between predicted and gold pairwise distances (averaged per sentence length, then macro-averaged over lengths 5–50) |

### Depth probe

| Metric | Definition |
|---|---|
| **Root%** | Accuracy of identifying the root word (word with minimum predicted norm) |
| **NSpr.** | Average Spearman correlation between predicted and gold depth orderings |

---

## Key Results

### Baselines confirm the probe isn't trivial

| Method | UUAS | DSpr. | Root% | NSpr. |
|---|---|---|---|---|
| LINEAR (left-to-right chain) | 48.9 | 0.58 | 2.9 | 0.27 |
| ELMo0 (char embeddings, no context) | 26.8 | 0.44 | 54.3 | 0.56 |
| DECAY0 (weighted avg of ELMo0) | 51.7 | 0.61 | 54.3 | 0.56 |
| PROJ0 (random BiLSTM on ELMo0) | 59.8 | 0.73 | 64.4 | 0.75 |

- ELMo0 (no contextual info) fails badly at distance — can't recover tree structure from character embeddings alone
- PROJ0 (random contextualization) is surprisingly strong but far below actual models

### Contextualized models embed syntax

| Method | UUAS | DSpr. | Root% | NSpr. |
|---|---|---|---|---|
| ELMo1 (layer 1) | 77.0 | 0.83 | 86.5 | 0.87 |
| BERT-base7 | 79.8 | 0.85 | 88.0 | 0.87 |
| BERT-large15 | **82.5** | 0.86 | 89.4 | 0.88 |
| BERT-large16 | 81.7 | **0.87** | **90.1** | **0.89** |

- BERT-large16 achieves 82% UUAS and 0.87 DSpr. — meaning a simple linear transformation of its representations recovers most of the dependency parse tree
- Root identification reaches 90% — the model clearly encodes hierarchical depth
- These results hold despite the models **never being trained on parse trees**

### Layer-by-layer analysis

- **ELMo:** Layer 1 (first BiLSTM) is best for syntax; layer 0 (char CNN) and layer 2 are worse
- **BERT-base:** Layers 6–9 are best (middle layers)
- **BERT-large:** Layers 14–19 are best (also middle layers)

Syntax is concentrated in **middle layers** of both architectures — early layers handle local features, later layers shift toward semantic/task-specific features.

### Low-rank sufficiency

The effective rank of $B$ needed for good performance is **surprisingly low**:
- $k = 64$ suffices for near-maximum UUAS on all models
- Increasing beyond $k = 128$ yields no further improvement
- All three models (ELMo, BERT-base, BERT-large) require approximately the same rank

This means syntax is encoded in a **low-dimensional subspace** (~64 dimensions) of the 768–1024 dimensional representation, suggesting the model devotes a compact, specialized portion of its capacity to syntactic structure.

---

## Concrete Example: Full Pipeline

**Sentence:** *"The complex financing plan in the S+L bailout law includes raising $30 billion from debt issued by the newly created RTC."*

1. Feed sentence through BERT-large, extract layer 16 representations $\mathbf{h}_1^{16}, \ldots, \mathbf{h}_n^{16}$
2. **Distance probe:** Compute $d_B(\mathbf{h}_i, \mathbf{h}_j)^2$ for all pairs → build minimum spanning tree → this tree closely matches the gold dependency parse (Figure 2 in paper)
3. **Depth probe:** Compute $\|B\mathbf{h}_i\|^2$ for each word → the predicted depth ordering closely tracks gold parse depth (Figure 3 in paper)

The predicted MST from BERT-large16 matches nearly all gold edges, even for this complex 20+ word sentence. The depth probe correctly identifies "includes" as near-root and deeply nested modifiers like "newly" as high-depth.

---

## Key Takeaways

1. **Syntax is a geometric property.** Parse trees aren't just extractable with complex classifiers — they're embedded in the *metric structure* of the representation space, recoverable with a linear map. This is analogous to word2vec's vector-offset analogies, but for tree structure.

2. **Linear probe = strong falsifiability.** Because the probe is simple (linear), high performance means the information is genuinely structured in the representation, not memorized by a powerful classifier. The baselines (ELMo0, random BiLSTM) confirm this — the probe can't extract syntax from representations that don't encode it.

3. **Low-rank subspace.** Syntax occupies ~64 dimensions of a 1024-dim space. The model allocates a compact subspace to syntactic structure while using the rest for other information (semantics, pragmatics, etc.).

4. **Middle layers specialize in syntax.** Consistent across ELMo and BERT — syntax peaks in middle layers, suggesting a processing hierarchy: surface features → syntax → semantics.

5. **Squared distances matter.** Using squared L2 distance (not L2 distance) is critical for matching integer tree distances. This is a design choice with significant empirical impact.
