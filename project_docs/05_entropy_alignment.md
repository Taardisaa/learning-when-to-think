# 05 — Entropy–Backoff Alignment Hypothesis

## Motivation: Huang et al. (ICLR 2024) and What It Doesn't Kill

Huang et al. show that **intrinsic self-correction** — asking an LLM to review and fix its own completed answer — **degrades reasoning accuracy** across GPT-3.5, GPT-4, GPT-4-Turbo, and Llama-2 on GSM8K, CommonSenseQA, and HotpotQA. The failure mode: the feedback prompt's implicit prior that "something is wrong" biases the model to flip correct answers into incorrect ones more often than the reverse (8.8% Correct→Incorrect vs 7.0% Incorrect→Correct on GSM8K).

However, Huang's result specifically targets the pattern:

$$
y_0 = \text{LLM}(x), \quad f = \text{LLM}(x, y_0, p_{\text{feedback}}), \quad y_1 = \text{LLM}(x, y_0, f, p_{\text{refine}})
$$

This is the same model, at the same level of reasoning, asked to **explicitly judge** a completed output. Our backoff mechanism is structurally different and may survive Huang's critique.

---

## The Indirect Alignment Argument

Our backoff is **entirely model-driven** — the model emits `<backoff_N>` tokens as a learned policy decision shaped by GRPO. There is no explicit entropy threshold, no rule-based trigger. At each semantic boundary the model chooses from:

$$
A = \{\texttt{<backoff\_1>},\; \texttt{<backoff\_2>},\; \texttt{<backoff\_3>},\; \texttt{</think>}\}
$$

The probability assigned to backoff:

$$
P(\texttt{<backoff\_1>} \mid x, y_{<t}) = \frac{\exp(z_{\text{backoff}})}{\sum_{a \in A} \exp(z_a)}
$$

For GRPO to have trained the model to emit backoff when reasoning has gone wrong, the model must have learned an **internal feature that correlates with "I'm on a bad path."** The most natural candidate is the same information that produces **high entropy in the content-token distribution** during the preceding chunk.

When the model is uncertain about what to say next — hedging between contradictory reasoning steps — that manifests as high entropy in the content tokens, and it's also the signal that the preceding reasoning was shaky. The chain:

$$
\text{bad reasoning path} \;\rightarrow\; \text{high } H(y_t) \text{ in content tokens} \;\rightarrow\; \text{hidden state encodes confusion} \;\rightarrow\; P(\texttt{<backoff\_N>}) \uparrow
$$

The model doesn't compute entropy explicitly. But GRPO **shaped the policy so that the same internal state** that would manifest as high content-token entropy also triggers backoff emission. They're two readouts of the same underlying representation.

---

## Why This Differs From Huang's Failure Mode

| | Huang's self-correction | Our backoff |
|---|---|---|
| **Signal** | Explicit prompt: "review your answer" | Implicit: learned internal state |
| **When** | After full generation is complete | Mid-generation, at semantic boundaries |
| **Action** | Re-generate from the same starting point | Rewind KV cache, try a different path |
| **Training** | None (prompt-only, zero-shot) | GRPO reward shapes when to trigger |
| **Information** | No new info — same $P(\cdot)$ queried twice | KV rewind + directive tokens = structurally different continuation |

Huang's model is asked to **explicitly judge** a completed answer — and fails because the generative model is bad at discrimination. Our model never explicitly judges anything. It has a **learned reflex** to backoff when its internal state is confused, the same way a human's "this doesn't feel right" triggers before they can articulate why.

Key structural advantages:

1. **Mid-stream, not post-hoc.** Backoff triggers at semantic boundaries *during* generation, before the model has committed to a full answer and built up self-consistency pressure.
2. **Rewind, not re-judge.** The action isn't "review what you wrote" — it's "discard the last $d$ chunks and try a different path from the cache snapshot." This produces a genuinely different continuation.
3. **Trained, not prompted.** GRPO reward directly penalizes bad backoff decisions (backing off from correct reasoning, or failing to backoff from incorrect reasoning). The policy learns from outcome signal, not from a generic "find problems" instruction.

---

## Empirical Verification Plan

If the indirect alignment holds, we expect a strong correlation between chunk-level content entropy and backoff probability. Diagnostic:

### Step 1: Log entropy and action at each boundary

During generation, at each semantic boundary, record:
- $\bar{H}_{\text{chunk}}$: average per-token entropy over the preceding chunk

$$
\bar{H}_{\text{chunk}} = \frac{1}{|\text{chunk}|} \sum_{t \in \text{chunk}} H(y_t), \quad H(y_t) = -\sum_{v \in V} P(v \mid x, y_{<t}) \log P(v \mid x, y_{<t})
$$

- $a_t \in A$: the action token chosen at this boundary
- Whether the final trajectory was correct (from the reward signal)

### Step 2: Plot and correlate

1. **$P(\text{backoff})$ vs $\bar{H}_{\text{preceding chunk}}$**: bin boundaries by entropy, compute backoff rate per bin. Expect monotonically increasing.
2. **Conditional accuracy**: among high-entropy chunks, compare accuracy when the model continued normally vs emitted `<backoff_N>`. If backoff helps, accuracy-after-backoff should exceed accuracy-after-continue in high-entropy regions.
3. **False positive rate**: among low-entropy chunks where model emitted `<backoff_N>`, how often was the original path actually correct? This is the Huang failure mode (Correct→Incorrect flip) — we want this to be low.

### Step 3: Interpret

| Outcome | Interpretation |
|---|---|
| Strong $\bar{H}$–backoff correlation | GRPO learned to use entropy as the backoff signal; our mechanism is an implicit, trained version of entropy-gated self-consistency |
| Weak correlation | GRPO found a different internal feature (possibly more structured than raw entropy — e.g., logical contradiction detection); worth probing with more targeted features |
| High false-positive rate in low-$\bar{H}$ bins | The model is over-backing-off on confident-but-wrong reasoning; need to reshape GRPO reward |

---

## Connection to Adaptive Compute Allocation

If entropy–backoff alignment is confirmed, our system implements a learned version of the adaptive sampling strategy suggested by the Huang analysis:

$$
k(x) = \begin{cases} 1 & \text{if } H(y \mid x) < \tau_{\text{low}} \quad \text{(confident — accept first path)} \\ 1 + \text{backoffs} & \text{if } H(y \mid x) \geq \tau_{\text{low}} \quad \text{(uncertain — model rewinds and retries)} \end{cases}
$$

But instead of a fixed threshold $\tau$ and fixed $k$, both are **learned and context-dependent**: the model decides *when* to backoff and *how far* to rewind, per-boundary, per-problem. This is strictly more expressive than the binned self-consistency approach, and it avoids the need to generate $k$ full independent trajectories — the model reuses partial computation via KV cache snapshots.

---

## Mechanistic Inspection: Looking Inside the Model

**Target model:** Qwen3-1.7B (GQA, bf16). Comfortably fits on a single GPU for inference with forward hooks. All methods below require only inference — no additional training except the linear probe.

### Inspection 1: Linear Probe on Residual Stream

**Goal:** Is "should I backoff?" linearly decodable from the hidden state at semantic boundaries?

**Method:** At each semantic boundary, extract the residual stream vector $h_t^{(L)} \in \mathbb{R}^{d}$ at the last token position of the chunk (just before the action token is sampled). Train a logistic regression:

$$
P(\text{backoff} \mid h_t) = \sigma(w^\top h_t + b), \quad w \in \mathbb{R}^d, \; b \in \mathbb{R}
$$

on (hidden state, action) pairs collected from generation rollouts.

**What it tells you:**
- High AUROC → backoff decision is a **linear feature** in the residual stream, not a complex emergent property
- The weight vector $w$ is the "backoff direction" in activation space. You can then test:

$$
\text{alignment} = \frac{w \cdot \nabla_h \bar{H}_{\text{chunk}}}{\|w\| \cdot \|\nabla_h \bar{H}_{\text{chunk}}\|}
$$

If $|\text{alignment}|$ is high, entropy and backoff are literally the **same direction** in the representation — confirming the indirect alignment hypothesis at the mechanistic level.

**Practical notes:**
- Collect ~5k–10k boundary examples (mix of continue/backoff) from rollouts on the training set
- Train/test split 80/20; logistic regression converges in seconds
- Can also probe per-layer ($h_t^{(l)}$ for each $l$) to see at which depth backoff becomes decodable

### Inspection 2: Logit Lens Across Layers

**Goal:** At which layer does the model "decide" to backoff?

**Method:** At each semantic boundary, unembed the residual stream at every layer through the final LM head + layer norm:

$$
\text{logits}^{(l)} = W_{\text{unembed}} \cdot \text{RMSNorm}(h_t^{(l)})
$$

Then compute the probability assigned to backoff tokens at each layer:

$$
P_{\text{backoff}}^{(l)} = \sum_{b \in \{\texttt{backoff\_1}, \texttt{backoff\_2}, \texttt{backoff\_3}\}} \text{softmax}(\text{logits}^{(l)})_b
$$

Plot $P_{\text{backoff}}^{(l)}$ vs layer $l$ for continue-examples and backoff-examples separately.

**What it tells you:**
- **Early divergence** (layers 5–15): the model detects "bad path" from shallow features (local incoherence, pattern mismatch). Suggests the backoff signal is relatively surface-level.
- **Late divergence** (layers 25–36): the model needs deep reasoning to decide. Suggests backoff depends on semantic understanding of the reasoning chain.
- **Gradual buildup**: information accumulates across layers — no single "decision layer." This is the typical pattern for complex decisions.

**Practical notes:**
- Register forward hooks on each layer's output. Single forward pass per example.
- Aggregate over ~1k boundary examples for clean curves.
- A larger model would likely show a **sharper layer transition** — more depth gives more separation between "still computing" and "decided."

### Inspection 3: Activation Patching (Causal Tracing)

**Goal:** Which layers/components are **causally responsible** for the backoff decision?

**Method:** Find paired examples — same or similar prefix, but one continues normally and the other emits `<backoff_N>`. Then:

1. Run clean forward pass on the backoff example → gets backoff logit $z_{\text{backoff}}^{\text{clean}}$
2. Run clean forward pass on the continue example → gets its activations at every layer
3. **Patch**: re-run the backoff example, but at layer $l$, replace its residual stream with the continue example's residual stream
4. Measure how much the backoff logit drops:

$$
\Delta_l = z_{\text{backoff}}^{\text{patched}(l)} - z_{\text{backoff}}^{\text{clean}}
$$

Large $|\Delta_l|$ at a specific layer = that layer is **causally necessary** for the backoff decision.

**Variants:**
- **Layer-wise patching**: patch entire residual stream at layer $l$ → identifies critical layers
- **Component-wise patching**: patch only the MLP output or only the attention output at layer $l$ → identifies whether backoff is driven by attention (contextual reasoning) or MLP (stored knowledge/patterns)
- **Head-wise patching**: patch individual attention heads → identifies specific "backoff heads"

**What it tells you:**
- If a small number of layers/heads are causally responsible, the backoff circuit is **modular** and interpretable
- If many layers contribute equally, the decision is **distributed** — harder to interpret but expected for complex decisions

**Practical notes:**
- Need paired examples with similar prefixes. Can construct these by taking a prompt, running generation twice with different seeds, and selecting pairs where one continued and one backed off.
- Each patch requires one forward pass, so total cost = (num_layers × num_examples) forward passes. At 4B bf16, ~500 examples × 36 layers ≈ 18k forward passes — feasible in a few hours on one GPU.

### Inspection 4: Top "Backoff Neurons" in MLP Layers

**Goal:** Find individual MLP neurons that fire specifically for backoff decisions, and understand what they respond to.

**Method:** Collect MLP intermediate activations at boundary positions across many rollouts. For each neuron $i$ in layer $l$:

$$
r_{l,i} = \text{corr}\big(\text{MLP}_l^{(i)}(h_t),\; \mathbb{1}[\text{action} = \text{backoff}]\big)
$$

Rank neurons by $|r_{l,i}|$. The top-$k$ are "backoff neurons."

**Then, for each top backoff neuron:**
- Find the top-20 input examples that maximally activate it
- Inspect: do these examples share a pattern? (hedging language, contradictions, arithmetic errors, repeated phrases)
- Find the top-20 examples that minimally activate it — are these clean, confident reasoning chunks?

**What it tells you:**
- If top neurons respond to **hedging/uncertainty language** ("however", "wait", "actually", "alternatively") → the model is using surface linguistic cues, similar to how entropy would manifest in token distributions
- If top neurons respond to **logical contradictions** (e.g., "X = 5 ... therefore X = 3") → the model learned a deeper structural feature beyond raw entropy
- If top neurons respond to **specific error patterns** (arithmetic mistakes, entity confusion) → the model has specialized error detectors

**Practical notes:**
- Larger models have wider MLP layers, so expect more specialized neurons with cleaner activation patterns
- Collect ~5k–10k boundary examples. Correlation computation is fast (just matrix ops).
- Can also check if top backoff neurons overlap with neurons that have high correlation with $\bar{H}_{\text{chunk}}$ — direct test of whether entropy and backoff share neural substrate.

---

## Recommended Inspection Order

| Priority | Method | GPU cost | What it answers |
|---|---|---|---|
| **1** | Log entropy + action (Step 1–2 above) | Negligible — add ~10 lines to generation loop | Does the correlation exist at all? |
| **2** | Logit lens across layers | 1 forward pass per example | When does the model decide? |
| **3** | Linear probe on residual stream | ~5k forward passes + logistic regression | Is backoff linearly decodable? Is it the same direction as entropy? |
| **4** | Top backoff neurons (MLP correlation) | ~5k forward passes + correlation | What features drive backoff? Surface uncertainty vs deep logic? |
| **5** | Activation patching | ~18k forward passes | Causal circuit: which layers/heads are necessary? |

Priority 1 requires no new infrastructure — just logging in the existing generation loop. Priorities 2–4 need a simple hook-based script. Priority 5 is the most expensive but gives the strongest causal story. All are single-GPU feasible on Qwen3-1.7B.
