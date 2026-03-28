# Linear Algebra Basics for Understanding Probing Methods

> Supplement to [Hewitt 2019 — A Structural Probe for Finding Syntax in Word Representations](../read/hewitt2019_structural_probe_syntax.md)

## Linear Transformation

A matrix $B \in \mathbb{R}^{k \times m}$ defines a linear transformation: takes an $m$-dim vector, outputs a $k$-dim vector.

$$\mathbf{y} = B\mathbf{x}$$

"Linear" means it can only do three things:
- **Rotate** (change direction)
- **Scale** (stretch/shrink)
- **Project** (drop dimensions, e.g. 1024-dim → 64-dim)

It **cannot** create new nonlinear features — no squaring, no thresholding, no combining features multiplicatively. This is why linear probes are useful: if a linear probe extracts information, that information was already geometrically present in the input.

---

## Inner Product and Quadratic Form

The standard dot product $\mathbf{x}^T \mathbf{y} = \sum_i x_i y_i$ is one inner product, but you can define a **family** of inner products by inserting a matrix:

$$\langle \mathbf{x}, \mathbf{y} \rangle_A = \mathbf{x}^T A \mathbf{y}$$

> Note that an inner product produces a scalar.

The standard dot product is just the special case where $A = I$ (identity matrix — ones on the diagonal, zeros elsewhere): $\mathbf{x}^T I \mathbf{y} = \mathbf{x}^T \mathbf{y}$. Multiplying by $I$ does nothing. Choosing a different $A$ warps the space before measuring — stretching some directions, compressing others — so you get a different notion of "closeness" between the same two vectors.

When you set $\mathbf{y} = \mathbf{x}$ (a vector with itself), you get a **quadratic form**: $\mathbf{x}^T A \mathbf{x}$. It's called "quadratic" because expanding it gives a polynomial where every term is degree 2 (e.g. $a x_1^2 + (b+c) x_1 x_2 + d x_2^2$ for a 2-dim vector). A quadratic form maps a vector to a single scalar — in the structural probe, the depth probe $\|\mathbf{h}_i\|_B^2 = \mathbf{h}_i^T B^T B \mathbf{h}_i$ is exactly this: it takes a word vector and outputs a scalar (the word's parse depth).

Different $A$ matrices define different "rulers" — different ways to measure distance and angle in the same vector space. The structural probe is essentially learning which "ruler" ($A = B^T B$) makes Euclidean distance match syntactic tree distance.

---

## Positive Semi-Definite (PSD)

A symmetric matrix $A$ is **positive semi-definite** if for all vectors $\mathbf{x}$:

$$\mathbf{x}^T A \mathbf{x} \geq 0$$

Intuition: the quadratic form never gives a negative value.

**Why it matters:** PSD guarantees that the inner product $\mathbf{x}^T A \mathbf{x}$ behaves like a "squared length" — always non-negative. This means distances derived from it are always non-negative and symmetric, so you get a valid distance metric.

**Key property:** Any matrix of the form $A = B^T B$ is automatically PSD:

$$\mathbf{x}^T B^T B \mathbf{x} = (B\mathbf{x})^T (B\mathbf{x}) = \|B\mathbf{x}\|^2 \geq 0$$

The squared norm of any vector is always $\geq 0$. So the structural probe's parameterization ($B$ matrix, distance = $\|B(\mathbf{h}_i - \mathbf{h}_j)\|^2$) automatically gives valid distances.

**Positive definite** vs **semi-definite**: "semi" means $\mathbf{x}^T A \mathbf{x} = 0$ is allowed for some $\mathbf{x} \neq 0$ (some directions get collapsed to zero length). Positive definite requires $> 0$ for all $\mathbf{x} \neq 0$. When $B$ is low-rank ($k < m$), $A = B^T B$ is only semi-definite because the $(m - k)$ dimensions outside $B$'s column space get mapped to zero.

---

## Matrix Rank

The **rank** of $B \in \mathbb{R}^{k \times m}$ is the number of linearly independent rows (or columns). It tells you the dimensionality of the output space.

- If $k = m$ and $B$ is full rank: no information is lost (rotation + scaling only)
- If $k < m$: projection — the output lives in a $k$-dim subspace of the original $m$-dim space

In Hewitt 2019, $B$ projects from 1024-dim (BERT) to ~64-dim. This means syntax is encoded in a 64-dimensional subspace. The other 960 dimensions encode other things (semantics, position, etc.) that are irrelevant to syntax and get discarded by $B$.

---

## L2 Norm and Squared L2 Distance

**L2 norm** (Euclidean length):

$$\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}$$

**Squared L2 distance** between two vectors:

$$d(\mathbf{x}, \mathbf{y})^2 = \|\mathbf{x} - \mathbf{y}\|^2 = \sum_i (x_i - y_i)^2$$

The structural probe uses **squared** distance, not raw distance.

**Why squared distance is not a proper metric:** A proper metric requires the triangle inequality: $d(a,c) \leq d(a,b) + d(b,c)$. Squaring breaks this. Example: three points on a line, each 2 apart. Raw distance: $d(a,c) = 4 \leq 2 + 2 = 4$, holds. Squared: $d(a,c)^2 = 16 \leq 4 + 4 = 8$? No — 16 > 8. Squaring amplifies large distances disproportionately, so the inequality fails.

**Why it still works for trees:** Squared distance preserves the same relative ordering between word pairs as raw distance (if $d_1 > d_2$ then $d_1^2 > d_2^2$), so the minimum spanning tree — and thus the recovered parse tree — is identical.

**Why it works *better* than raw distance (Appendix A.1):** The authors speculate two reasons: (1) Raw distance involves a square root, whose gradient $\frac{1}{2\sqrt{x}}$ explodes near zero, making optimization unstable. Squared distance has smoother gradients. (2) Tree distances are integers (1, 2, 3, ...). Squared distance outputs a positive number directly without needing a square root step, making it easier to match these integer targets. The authors note this is an open question — they observed squared distance consistently outperforms raw distance across all experiments, but have no formal proof of why.

---

## Dependency Tree (依存树)

A dependency tree connects every word in a sentence to exactly one "head" word it grammatically depends on. The main verb is the root (depends on nothing). Each word has exactly one head (except root), so $n$ words → $n-1$ edges, no cycles, fully connected — i.e. a tree.

**Example:** *"The cat sat on the mat"*

```
sat  (root)
 ├── cat  (subject depends on verb)
 │    └── The  (determiner depends on noun)
 ├── on   (preposition depends on verb)
 │    └── mat  (noun depends on preposition)
 │         └── the  (determiner depends on noun)
 └── .
```

Because a dependency tree is literally a tree, MST is the natural algorithm to recover it from a pairwise distance matrix.

---

## Minimum Spanning Tree (MST)

Given $n$ nodes and pairwise distances, the MST connects all nodes with $n-1$ edges that minimize total edge weight — no cycles. For the structural probe:

1. Compute predicted distance $d_B(\mathbf{h}_i, \mathbf{h}_j)^2$ for all word pairs
2. Run MST algorithm (e.g. Prim's or Kruskal's)
3. The resulting tree = predicted dependency parse
4. Compare edges against gold parse → UUAS score

---

## Spearman Correlation

A rank-based correlation: measures whether the **ordering** of predicted values matches the ordering of gold values, ignoring exact magnitudes.

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

| Term | Meaning |
|---|---|
| $n$ | Number of data points (e.g. number of words in a sentence) |
| $d_i$ | Difference in rank of item $i$ between predicted and gold orderings |
| $6$ | Constant from algebraic simplification — Spearman is "Pearson correlation applied to ranks", and simplifying with $\text{Var}(\text{ranks}) = \frac{n^2-1}{12}$ produces the factor of 6 |

- $\rho = 1$: perfect rank agreement
- $\rho = 0$: no correlation
- Used in the paper as DSpr. (distance) and NSpr. (norm/depth)

**Concrete example (depth probe, 4 words):**

| Word | gold depth | predicted $\|B\mathbf{h}\|^2$ | gold rank | predicted rank | $d_i$ | $d_i^2$ |
|---|---|---|---|---|---|---|
| sat | 0 | 1.2 | 1 | 1 | 0 | 0 |
| cat | 1 | 3.5 | 2.5 | 3 | -0.5 | 0.25 |
| on | 1 | 2.8 | 2.5 | 2 | 0.5 | 0.25 |
| The | 2 | 7.1 | 4 | 4 | 0 | 0 |

> Note: cat and on tie at gold depth 1, so they share the average rank $(2+3)/2 = 2.5$.

$\sum d_i^2 = 0.5$, $n = 4$:

$$\rho = 1 - \frac{6 \times 0.5}{4(16 - 1)} = 1 - \frac{3}{60} = 0.95$$

Close to 1 — the predicted depth ordering nearly matches gold. The probe doesn't need to predict exact depth values (1.2 vs 0, 3.5 vs 1), only the correct **relative ordering**.
