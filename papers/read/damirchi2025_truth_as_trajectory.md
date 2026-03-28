# Truth as a Trajectory: What Internal Representations Reveal About Large Language Model Reasoning

**Authors:** Hamed Damirchi, Ignacio Meza De la Jara, Zhen Zhang, Javen Shi (Australian Institute for Machine Learning, Adelaide University), Ehsan Abbasnejad (Monash University), Afshar Shamsi (Concordia University)

**Date:** 2025 | **Venue:** arXiv:2603.01326

## Problem

Existing methods for understanding LLM reasoning treat hidden states as **static points** in activation space — pick a layer, train a linear probe, classify correct vs. incorrect. This has two fundamental problems:

1. **Polysemanticity:** Raw activations at any single layer encode lexical content, syntactic structure, AND task-specific artifacts simultaneously. Linear probes latch onto surface-level lexical correlations (e.g., presence of specific tokens) rather than genuine reasoning validity.
2. **No consistent "right layer":** The activation steering literature shows that effective intervention is only possible in a narrow mid-layer band, with inconsistent results across datasets. The "geometry of truth" is task-specific and orthogonal across domains — a probe trained on one dataset fails on another.

| Approach | What it analyzes | Failure mode |
|----------|-----------------|--------------|
| Linear probe (single layer) | Static activation at layer $L$ | Learns lexical confounds, not reasoning; poor OOD transfer |
| Activation steering | Static direction at layer $L$ | Inconsistent across datasets; non-linear manifold resists linear intervention |
| Kinematic descriptors (velocity, curvature) | Scalar summaries of trajectory | Better than random, but no single descriptor generalizes across tasks |
| **TaT (this paper)** | **Full displacement trajectory across all layers and tokens** | **Captures non-linear structural invariants; strong OOD transfer** |

## Key Concepts

**Residual stream / residual update:** In a Transformer, each layer applies an additive update to the hidden state: $h_{\ell+1} = h_\ell + f_\ell(h_\ell)$. This is mathematically equivalent to one step of Euler's method for an ODE. The sequence $(h_0, h_1, \ldots, h_L)$ across all layers forms a **trajectory** through activation space.

**Displacement vector:** The difference $\mathbf{d}_{t,\ell} = h_{t,\ell+1} - h_{t,\ell}$ between consecutive layers for token $t$. This isolates what the layer **actively changed** (the "how") from what was already present in the representation (the "what"). Raw activations are dominated by persistent static features (token identity, position); displacement strips those away.

| | Raw activation $h_{t,\ell}$ | Displacement $\mathbf{d}_{t,\ell}$ |
|---|---|---|
| Contains | Token identity + position + syntax + semantics + reasoning | Only the update applied by layer $\ell$ |
| Dominated by | Static, high-magnitude persistent components | Active refinement signal |
| Polysemantic? | Yes — everything superimposed | Much less — isolates the process, not the content |

**Concrete example:** Consider token "5" at layer 10. The raw activation $h_{\text{"5"},10}$ encodes that this is the digit 5, its position in the sequence, its syntactic role, etc. The displacement $\mathbf{d}_{\text{"5"},10} = h_{\text{"5"},11} - h_{\text{"5"},10}$ tells you only what layer 10's computation *did* to that representation — e.g., linking it to an equation context. This is the reasoning signal.

**Activation trajectory:** For a candidate continuation $c_i$ with $N_i$ tokens processed through $L$ layers, stack all displacement vectors into a single sequence:

$$\mathbf{S}_i = [\mathbf{d}_{1,0}, \ldots, \mathbf{d}_{1,L-1}, \mathbf{d}_{2,0}, \ldots, \mathbf{d}_{N_i,L-1}]$$

This is a sequence of length $M_i = N_i \times L$ in $\mathbb{R}^d$. It unrolls the entire inference process (across both tokens and layers) into a single temporal path. The LSTM reads this path and classifies whether the reasoning trajectory is valid.

**Kinematic descriptors:** Scalar measures borrowed from physics, applied to the activation trajectory:

| Descriptor | Formula | Intuition |
|------------|---------|-----------|
| Velocity | $v_\ell = \|\Delta h_\ell\|_2$ | How much the representation changes per layer |
| Acceleration | $a_\ell = v_\ell - v_{\ell-1}$ | Rate of change of velocity |
| Jerk | $j_\ell = a_\ell - a_{\ell-1}$ | Smoothness of trajectory |
| Directional curvature | $\kappa_\ell = \frac{\langle \Delta h_\ell, \Delta h_{\ell-1} \rangle}{\|\Delta h_\ell\|_2 \|\Delta h_{\ell-1}\|_2}$ | Cosine similarity between consecutive updates |
| Arc length | $S = \sum_{\ell=0}^{L-1} \|h_{\ell+1} - h_\ell\|_2$ | Total geometric effort |

These are tested as baselines — they capture *some* signal but don't generalize across tasks, motivating the learned LSTM approach.

## Core Design / Method

### Step 1: Extract Displacement Trajectories

For each input (prompt + candidate answer), run a single forward pass through the frozen LLM. Collect hidden states $h_{t,\ell}$ for all tokens $t$ at all layers $\ell$. Compute displacements:

$$\mathbf{d}_{t,\ell} = h_{t,\ell+1} - h_{t,\ell}$$

| Term | Meaning |
|------|---------|
| $h_{t,\ell}$ | Hidden state of token $t$ at layer $\ell$ (dim $d$) |
| $\mathbf{d}_{t,\ell}$ | Displacement: what layer $\ell$ changed for token $t$ (dim $d$) |
| $t$ | Token index (1 to $N_i$) |
| $\ell$ | Layer index (0 to $L-1$) |

Stack into trajectory: $\mathbf{S}_i \in \mathbb{R}^{M_i \times d}$ where $M_i = N_i \times L$.

**Why displacement instead of raw activations?** Raw activations are dominated by high-magnitude persistent components (token identity, position). The displacement isolates the active residual update $f_\theta(h_{t,\ell})$. Due to element-wise non-linearities, these updates are constrained to a consistent reference frame (Residual Alignment). Key distinction: raw $h_{t,\ell}$ signals if a feature is *present*; displacement $\mathbf{d}_{t,\ell}$ signals if the model is *actively writing* that feature at this layer.

### Step 2: LSTM Classifier

Feed $\mathbf{S}_i$ sequentially into an LSTM:

$$\hat{y}_b = \sigma(\mathbf{W}^T \mathbf{z}_{M_i} + b)$$

| Term | Meaning |
|------|---------|
| $\mathbf{S}_i$ | Displacement trajectory sequence (length $M_i$) |
| $\mathbf{z}_{M_i}$ | LSTM's final hidden state after processing all $M_i$ steps |
| $\mathbf{W} \in \mathbb{R}^{d_{lstm} \times 1}$ | Classification head weights |
| $\hat{y}_b$ | Predicted probability that the candidate continuation is correct |

**Why LSTM over linear probe?** The reasoning signal is non-linear and sequential — it depends on *how* updates compose across depth and tokens, not just their aggregate. An LSTM captures these sequential dependencies. They verified this: an order-invariant Set MLP baseline (same displacement vectors, no sequential structure) consistently underperforms the LSTM on OOD transfer (Table 6), confirming the *order* of updates matters.

**Why not Transformer-based probe?** Not discussed, but the LSTM is tiny (4.76M params, 0.06% of LLaMA 3.1-8B) and already works well. The key insight is that even a simple sequential model can capture the trajectory structure.

### Overhead

| Metric | LLaMA 3.1-8B (fp16) | LSTM Classifier | Overhead |
|--------|---------------------|-----------------|----------|
| Parameters | 8.0B | 4.76M | 0.06% |
| Inference time | 64.0 ms | 10.5 ms | 16%* |
| Memory | ~15,000 MB | 18.1 MB | 0.12% |

*The 16% inference overhead is for the naive implementation (extract all activations, then run LSTM separately). In practice, the LSTM can be embedded within each layer and pipelined with generation, making overhead negligible.

## Results

### Cross-Task Generalization (Table 1, Llama-3.1-8B)

Trained on one dataset, evaluated on all others. Showing average OOD accuracy:

| Train Dataset | Method | Avg Accuracy | ID Acc. | OOD Avg. |
|---------------|--------|-------------|---------|----------|
| ARC-C | Linear Probe | 71.03 | 75.32 | 70.49 |
| ARC-C | **TaT (Disp.)** | **79.63** | **82.17** | **79.31** |
| ARC-E | Linear Probe | 72.44 | 83.99 | 71.00 |
| ARC-E | **TaT (Disp.)** | **77.76** | **89.10** | **76.34** |
| OpenQA | Linear Probe | 67.11 | 83.15 | 65.11 |
| OpenQA | **TaT (Disp.)** | **78.38** | **90.80** | **76.83** |

TaT consistently outperforms linear probing, especially on OOD transfer (+5-12 points). It also beats the base model's own zero-shot (67.9%) and few-shot (75.7%) accuracy in most cases — despite being zero-shot at inference and only seeing training data from a single source task.

### TaT vs. LoRA (Table 2, trained on ARC-E)

| Method | StoryCloze | OpenQA | ARC-E | ARC-C | BoolQ | Hellaswag |
|--------|-----------|--------|-------|-------|-------|-----------|
| Linear Probe | 69.62 | 80.95 | 83.99 | **75.55** | 58.82 | 71.07 |
| **TaT** | **94.98** | 77.20 | **89.10** | 73.81 | **78.56** | **79.19** |
| LoRA (rank 16) | 83.76 | 52.40 | 85.98 | 61.26 | **79.97** | 75.86 |

TaT outperforms LoRA on 5 of 6 transfer tasks. LoRA overfits to the source task's semantic distribution; TaT observes the *geometry* of inference, which transfers better.

### Displacement vs. Raw Trajectories (Table 4)

| Train | Method | Avg | ID Acc. | OOD Avg. |
|-------|--------|-----|---------|----------|
| ARC-C | Linear Probe | 71.03 | 75.32 | 70.49 |
| ARC-C | TaT (Raw) | 83.76 | **84.90** | **83.62** |
| ARC-C | TaT (Disp.) | 79.63 | 82.17 | 79.31 |
| OpenQA | TaT (Raw) | 71.78 | 87.20 | 69.85 |
| OpenQA | **TaT (Disp.)** | **78.38** | **90.80** | **76.83** |

Raw trajectories can have higher ID accuracy but worse OOD transfer (OpenQA case). Raw activations overfit to lexical content; displacement is more robust to distribution shift. The gap is especially clear on toxicity detection (Table 3): TaT (Disp.) achieves **84.23%** on ToxiGen vs. 81.99% for Raw and 79.62% for Linear Probe.

### Ablation: What Matters in the Trajectory? (Table 5)

| Method | ID Acc. | OOD Avg. |
|--------|---------|----------|
| TaT (full: all tokens × all layers) | **82.17** | **79.31** |
| TaT-Mid Layer (single layer, all tokens) | 66.98 | 70.41 |
| TaT-Final Token (all layers, last token only) | 73.38 | 73.68 |
| Linear Probe | 75.32 | 70.49 |

Collapsing to a single layer destroys most of the signal. Using only the final token is better but still worse than full trajectory. **Both depth (layers) AND breadth (tokens) are needed.**

### Sequential Order Matters (Table 6)

| Method | Avg | ID Acc. | OOD Avg. |
|--------|-----|---------|----------|
| TaT (LSTM) | **79.63** | 82.17 | **79.31** |
| Set MLP (same vectors, no order) | 72.67 | 68.65 | 73.17 |

The LSTM consistently beats the order-invariant baseline, confirming that the *sequence* of updates matters — it's not just which updates happen, but in what order they compose.

## Key Takeaways

1. **Displacement > raw activation for reasoning validity.** Taking the difference between consecutive layers strips away static lexical content and isolates the active computation. This makes the signal more robust to distribution shift and lexical confounds.

2. **Reasoning has a consistent geometric trajectory.** A classifier trained on displacement trajectories from a single dataset (e.g., ARC-C) transfers to 9+ other reasoning benchmarks without fine-tuning. This suggests valid reasoning follows structural invariants that transcend task-specific content.

3. **The full token × layer grid is necessary.** Collapsing to a single layer (standard probing) or single token (last-token probing) significantly degrades performance. The reasoning signal is distributed across the entire inference computation.

4. **Sequential order of updates carries information.** An LSTM on ordered displacements outperforms a Set MLP on the same unordered displacements. How the model refines representations layer-by-layer matters, not just the aggregate change.

5. **TaT outperforms LoRA on OOD generalization.** LoRA modifies model weights and overfits to the source task's semantic distribution. TaT observes the frozen model's geometry and transfers better — it learns what valid reasoning *looks like*, not what specific answers look like.

6. **Overhead is minimal.** The LSTM adds 4.76M params (0.06% of LLaMA-8B), 18MB memory, and can be pipelined with generation. This is practical for deployment as a runtime monitor.

7. **Works on toxicity detection too.** TaT distinguishes quoted/educational use of toxic vocabulary from actual toxic intent, outperforming probes that latch onto keyword presence. The geometric signature of "reasoning toward toxic output" differs from "mentioning toxic content."

8. **Future direction stated in the paper: extend to self-generated multi-step reasoning chains.** The authors explicitly call out detecting reasoning errors in CoT as the next step (Section 7), noting it needs ground-truth construction and new evaluation protocols.

## Connection to Our Work

This paper provides the **strongest theoretical framework** for our internal-state-based approach to backoff detection:

**Displacement as the right representation.** If we ever build a probe to detect erroneous reasoning steps in CoT, we should use layer-wise displacement $\mathbf{d}_{t,\ell} = h_{t,\ell+1} - h_{t,\ell}$, not raw activations. The displacement isolates the active computation and resists lexical confounds — exactly what we need when the model writes "Therefore x = 5" (lexically similar whether correct or wrong, but the *computational trajectory* differs).

**Trajectory divergence = error point.** Their Figure 1 shows that correct trajectories follow smooth paths while incorrect ones exhibit sharp deviations. This is exactly the phenomenon we want to detect: the point where the trajectory diverges from a valid reasoning path is where `<backoff_N>` should be placed. Their future work section (Section 7) explicitly proposes this: *"Identifying where in the token×layer computation a candidate begins to diverge from a valid trajectory."*

**Cross-task generalization is encouraging.** If the geometry of valid reasoning transfers across tasks, then a probe trained on math CoT could potentially detect errors in diverse reasoning domains — making the backoff mechanism more general than task-specific training data alone.

**The LSTM architecture might be overkill for our use case.** We don't need full trajectory classification — we need step-level error detection. A lighter approach might work: compute displacement statistics within each reasoning step (chunk) and train a per-step classifier. But the displacement-over-raw insight is directly applicable.

**Gap between this work and ours:** They classify entire prompt+answer trajectories as correct/incorrect (sequence-level). We need to identify *which step* went wrong and *how far back* to rewind (step-level, with depth estimation). Their framework gives us the right representation; we need to develop the right granularity.
