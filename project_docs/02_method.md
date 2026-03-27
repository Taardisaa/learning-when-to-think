# Method: Non-Monotonic Reasoning via Learned Backoff

## The Problem

Current LLMs reason **forward-only**: tokens are appended sequentially, and once generated, they permanently occupy context window space. When the model goes down a wrong path — a miscalculation, a bad algorithmic choice, a confused tangent — those tokens cannot be undone. The model can only append corrections ("wait, actually..."), which wastes even more context.

In a finite-context setting, this is not just inefficient — it **actively degrades reasoning quality**. Every useless token shrinks the window available for productive reasoning.

## The Idea

Give the LLM an **undo operation**: $\langle\texttt{backoff}\rangle$. When the model recognizes it has gone down a bad path, it can:

1. Hard-truncate the KV cache to a prior semantic boundary (reclaiming context space)
2. Optionally emit a short directive ($\leq K_{\text{dir}}$ tokens) describing what to do differently
3. Resume generation from the truncated state

This is trained end-to-end via RL (GRPO). The model learns **when** to backoff and **what directive** to give, through the outcome reward signal.

---

## Action Space

The model generates tokens from its vocabulary $\mathcal{V}$. We add $2 + D_{\max}$ (= 5) **new special tokens** and reuse one existing token:

$$\mathcal{A} = \{\langle\texttt{continue}\rangle,\; \langle\texttt{backoff}\rangle,\; \langle\texttt{terminate}\rangle\} \subset \mathcal{V}$$

$$\mathcal{D} = \{\langle 1 \rangle,\; \langle 2 \rangle,\; \langle 3 \rangle\} \subset \mathcal{V} \quad (D_{\max} = 3)$$

**$\langle\texttt{terminate}\rangle$ is just `</think>`.** CoT models like Qwen3 already have `</think>` in their vocabulary with a trained embedding — it already means "stop reasoning, output the answer." After `</think>`, the model naturally transitions to answer generation. We don't need to add a new token for this or "extract" anything; the model knows what to do.

**$\langle\texttt{continue}\rangle$ is a no-op action.** Since we force a 3-way decision at every semantic boundary (via logits masking), we need a "keep going" option. Without it, the model would be forced to either backoff or terminate at every boundary. `<continue>` means "I checked and the current reasoning is fine, proceed to the next chunk." It enters the KV cache (costs 1 token per boundary — negligible) and keeps every decision point explicitly logged in the trajectory, which makes RL training and analysis cleaner.

Only $\langle\texttt{continue}\rangle$, $\langle\texttt{backoff}\rangle$, and $\mathcal{D}$ are new. These are single-token entries added to the tokenizer (not multi-token sequences). Same mechanism chat models use for `<|im_start|>`, `<|endoftext|>`, etc.:

```python
tokenizer.add_special_tokens({"additional_special_tokens": [
    "<continue>", "<backoff>",
    "<depth_1>", "<depth_2>", "<depth_3>",
]})
model.resize_token_embeddings(len(tokenizer))
# <terminate> = tokenizer.convert_tokens_to_ids("</think>")  # already exists
```

The new token embeddings ($\langle\texttt{continue}\rangle$, $\langle\texttt{backoff}\rangle$, $\mathcal{D}$) are **randomly initialized**. SFT warm-up (Phase 1) gives them meaning; GRPO (Phase 2-3) refines when to use them. No other training is needed. $\langle\texttt{terminate}\rangle$ = `</think>` already has a trained embedding — one less token to cold-start.

### Two-mode generation

Generation alternates between two modes:

1. **Free generation**: the model generates reasoning tokens from the full vocabulary $\mathcal{V}$ (normal autoregressive decoding)
2. **Action decision**: at semantic boundaries, we **mask all logits to $-\infty$ except $\mathcal{A}$** — the model's LM head produces a distribution over the 3 action tokens, and we sample from that

```python
# At a semantic boundary:
logits = model(input_ids).logits[:, -1, :]        # [batch, vocab_size]
mask = torch.full_like(logits, float('-inf'))
mask[:, action_token_ids] = 0.0
logits = logits + mask                             # only action tokens survive
action = torch.multinomial(logits.softmax(-1), 1)
```

There is no separate action head or controller. The same LM head that predicts the next word also decides the action — we just mask at decision points. If the sampled action is $\langle\texttt{backoff}\rangle$, logits are further constrained to $\mathcal{D}$ for one step to select the rewind depth.

### Backoff granularity

Backoff operates **per semantic boundary** (sentences, logical connectives, structural markers), not per chunk. The depth token controls how far to rewind: undo a single bad sentence ($\langle 1 \rangle$), a few confused steps ($\langle 3 \rangle$), or a whole wrong approach ($\langle D_{\max} \rangle$).

### What happens after backoff: directive injection

A bare backoff (rewind + resample) is just inference-time retry — at temperature=0 it regenerates the exact same tokens, and even with sampling, the model has no signal about what went wrong. This is not meaningfully different from GRPO's group sampling.

Our backoff includes a **text directive**: after rewinding the KV cache, the model generates $\leq K_{\text{dir}}$ tokens of free text before `<continue>` resumes normal generation. These directive tokens enter the KV cache and **condition the continuation differently** — e.g., "try division instead", "recheck the second step", "use a different variable name."

This is what makes backoff non-redundant with resampling. GRPO learns both *when* to backoff and *what directive to write*, purely from outcome reward. No extra architecture or training objective — it's just more tokens.

**Ablation (bare vs. directive):** comparing backoff-without-directive against backoff-with-directive directly tests whether learned steering matters, or if simple retry is sufficient.

### Note on SFT loss

During SFT, the loss must include the new tokens — $\langle\texttt{continue}\rangle$, $\langle\texttt{backoff}\rangle$, and $\mathcal{D}$ (not masked out). Some chat-template trainers skip loss on special tokens by default — this must be disabled, otherwise the model never learns the new embeddings. `</think>` is already learned and doesn't have this problem.

---

## Generation Loop (Inference)

Standard LLM inference is a single loop: generate token, append, repeat. Ours has an outer loop on top: generate a chunk of tokens until a semantic boundary, then make an action decision. If the action is $\langle\texttt{continue}\rangle$, generate the next chunk. If $\langle\texttt{terminate}\rangle$, stop and extract the answer. If $\langle\texttt{backoff}\rangle$, rewind the KV cache to a prior boundary, optionally inject a directive, and resume generating from there.

The key difference from normal generation: the KV cache can **shrink** mid-inference (on backoff), not just grow. This is what makes the reasoning trajectory non-monotonic.

```
Input: prompt x, base model p_θ, boundary set B,
       limits K_min, K_max, K_dir, D_max, B_max, T_max

kv_cache ← encode(x)
boundaries ← [len(x)]      # stack of semantic boundary positions
backoff_count ← 0

for t = 1, 2, ..., T_max:

    # 1. Generate tokens until next semantic boundary
    chunk_len ← 0
    while True:
        y_j ~ p_θ(· | kv_cache)
        append y_j to kv_cache
        chunk_len += 1
        if (y_j ∈ B and chunk_len ≥ K_min) or chunk_len ≥ K_max:
            boundaries.push(len(kv_cache))
            break

    # 2. Model emits action token
    allowed ← A if backoff_count < B_max else {<continue>, <terminate>}
    a_t ~ p_θ(· | kv_cache)    # logits restricted to `allowed`

    # 3. Execute action
    if a_t == <continue>:
        pass                    # proceed to next chunk

    elif a_t == <backoff>:
        backoff_count += 1

        # 3a. Model emits depth token: how many boundaries to rewind
        d_t ~ p_θ(· | kv_cache)    # logits restricted to D = {<1>, ..., <D_max>}
        depth ← min(d_t, len(boundaries) - 1)   # can't rewind past prompt

        # 3b. Hard-truncate KV cache to target boundary
        for i in range(depth):
            boundaries.pop()
        target_pos ← boundaries[-1]
        kv_cache ← kv_cache[:target_pos]

        # 3c. Generate directive (≤ K_dir tokens, until <continue>)
        directive_len ← 0
        while directive_len < K_dir:
            y_j ~ p_θ(· | kv_cache)
            append y_j to kv_cache
            directive_len += 1
            if y_j == <continue>:
                break
        # if K_dir reached without <continue>, force-append <continue>
        if y_j != <continue>:
            append <continue> to kv_cache

        # Generation resumes from here on next iteration

    elif a_t == <terminate>:
        answer ← extract_answer(kv_cache)
        return answer
```

**Example trace** (GSM8K problem, semantic boundaries marked with `|`):

```
[prompt] |
Natalia sold 48 clips in April. |                          ← boundary 1
Half of 48 is 12, so she sold 12 in May. |                 ← boundary 2  (WRONG: 48/2 = 24)
So the total is 48 + 12 = 60. |                            ← boundary 3  (WRONG, follows from error)
<backoff> <2>                                               ← rewind 2 boundaries (delete boundary 2+3)
that division was wrong, redo carefully <continue>          ← directive
Half of 48 is 24, so she sold 24 in May. |                 ← boundary 2' (CORRECT)
The total is 48 + 24 = 72. |                               ← boundary 3'
<terminate>                                                 ← answer: 72 ✓
```

**KV cache after backoff:** `[prompt] | Natalia sold 48 clips in April. | that division was wrong, redo carefully <continue>`

The two wrong sentences are **gone**. Context reclaimed. The model continues with 2 fewer sentences of wasted context.

**Key design choices:**
- Backoff granularity is **per semantic boundary**, not per chunk — the model can undo a single sentence or several
- Depth $d_t \in \{1, \ldots, D_{\max}\}$ is a **learned** decision (the model chooses how far back to rewind)
- Maximum backoff count $B_{\max}$ prevents infinite loops
- Directive is capped at $K_{\text{dir}}$ tokens, terminated by $\langle\texttt{continue}\rangle$

---

## MDP Formulation

$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$

### State

$$s_t = (\mathbf{x},\; y_{1:n_t},\; \mathcal{C}_t)$$

where $\mathbf{x}$ is the prompt, $y_{1:n_t}$ are the tokens currently in the KV cache (this can **shrink** after backoff), and $\mathcal{C}_t$ is the remaining context capacity $= C_{\max} - n_t$.

Note: unlike standard LLM generation where $n_t$ is monotonically increasing, here $n_{t+1}$ can be $< n_t$ after a backoff.

### Transitions

**Continue:**

$$a_t = \langle\texttt{continue}\rangle: \quad y_{n_t+1:n_{t+1}} \sim p_\theta(\cdot \mid s_t), \quad n_{t+1} > n_t$$

Generate until next semantic boundary. State grows.

**Backoff:**

$$a_t = \langle\texttt{backoff}\rangle, \quad d_t \in \mathcal{D}: \quad n_b = \texttt{boundaries}[-d_t], \quad \text{dir} \sim p_\theta(\cdot \mid s_t), \quad n_{t+1} = n_b + |\text{dir}|$$

The model emits a depth token $d_t \in \{1, \ldots, D_{\max}\}$ choosing how many semantic boundaries to rewind, then generates a directive. The KV cache is truncated to boundary position $n_b$, the directive is appended, and generation resumes.

$D_{\max} = 3$ (hyperparameter): $\langle 1 \rangle$ = undo last sentence, $\langle 2 \rangle$ = undo two boundaries, $\langle 3 \rangle$ = larger rewrite. The model decides in one shot how far to rewind, rather than relying on sequential backoffs (which would require the model to remember to backoff again after truncation — unreliable).

RH: for this hyperparameter, we do need to rethink over it. use existing samples to see what is possibly the best.

**State shrinks** (net), reclaiming $n_t - n_b - |\text{dir}|$ tokens of context.

**Terminate:**

$$a_t = \langle\texttt{terminate}\rangle: \quad \hat{a} = \texttt{Extract}(s_t), \quad \text{episode ends}$$

### Policy

**Action selection:**

$$\pi_\theta(a_t \mid s_t) = \text{softmax}\!\left(\text{LM-head}(\mathbf{h}_{n_t})_{\mathcal{A}}\right)$$

**Depth selection** (only if $a_t = \langle\texttt{backoff}\rangle$):

$$\pi_\theta(d_t \mid s_t, a_t) = \text{softmax}\!\left(\text{LM-head}(\mathbf{h}_{n_t}')_{\mathcal{D}}\right)$$

where $\mathbf{h}_{n_t}'$ is the hidden state after appending the $\langle\texttt{backoff}\rangle$ token. Both are the LLM's own logits restricted to the relevant token subsets. No separate controller. $\theta = \theta_{\text{base}} + \Delta\theta_{\text{LoRA}}$.

### Reward

$$R(\tau) = r(\hat{a}, a^*) + \alpha \sum_{j=0}^{k-1} r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}_{j-1})$$

Two terms:
- $r(\hat{a}, a^*) \in \{0, 1\}$ — **outcome**: binary correctness (exact match).
- $r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}_{j-1})$ — **progress**: per-segment dense reward, measuring the change in meta-prover accuracy before vs. after segment $j$. Adapted from MRT (Qu et al., 2025).

**Progress reward.** At each segment boundary, we force the model to terminate thinking (append `</think>`) and greedily decode an answer. Progress for segment $j$ is:

$$r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}_{j-1}) = J_r(\mu(\cdot \mid \mathbf{c}_j)) - J_r(\mu(\cdot \mid \mathbf{c}_{j-1}))$$

where $\mu$ is the meta-prover (the same model forced to answer immediately), $J_r$ is the expected 0/1 accuracy of that answer, and $\mathbf{c}_j$ is the KV cache state after segment $j$. With `num_probes=1` (greedy), $J_r \in \{0, 1\}$ and progress is in $\{-1, 0, +1\}$.

**Why this replaces token penalties and exploration bonuses.** The old reward used $\lambda_{\text{tok}} \cdot T_{\text{net}}$ (penalising length) and $\lambda_{\text{explore}} \cdot \mathbb{1}[\langle\texttt{backoff}\rangle \in \tau]$ (rewarding any backoff). Both are blunt instruments:
- Token penalty punishes *length*, not *quality* — a long trace making steady progress is good.
- Exploration bonus rewards backoff regardless of whether it helped.

Progress reward subsumes both. Wasteful tokens show up as segments with zero or negative progress. Good backoffs show up as large positive progress (the meta-prover suddenly gets the right answer after context reclamation). Bad backoffs get penalised (undoing correct work). Token efficiency emerges *implicitly* — MRT found that progress optimisation reduces length without any explicit penalty.

**Hyperparameters:**
- $\alpha \geq 0$ — weight on progress signal (default: 0.5). Controls explore/exploit balance.
- `num_probes` — meta-prover samples per prefix (default: 1, greedy). Higher values give smoother signal at higher compute cost.

---

## Training via GRPO

For each prompt $x$, sample $G$ trajectories $\{\tau_1, \ldots, \tau_G\}$ under $\pi_{\theta_{\text{old}}}$:

$$\hat{A}_i = \frac{R(\tau_i) - \mu_G}{\sigma_G}$$

Update LoRA parameters:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_x\, \mathbb{E}_{i=1}^{G}\!\left[\min\!\left(\rho_i \hat{A}_i,\;\; \text{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)\, \hat{A}_i\right)\right] - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $\rho_i = \frac{\pi_\theta(\tau_i)}{\pi_{\theta_{\text{old}}}(\tau_i)}$.

**No value network.** Group-relative advantages provide variance reduction. The only trainable parameters are LoRA adapters.

### Handling backoff in the trajectory likelihood

A trajectory with backoff contains **deleted segments** that are no longer in the KV cache but were generated and influenced the policy. For the GRPO likelihood ratio, the trajectory probability includes all decisions:

$$\pi_\theta(\tau) = \prod_{t=1}^{|\tau|} \left[\underbrace{p_\theta(\text{chunk}_t \mid s_t)}_{\text{chunk tokens}} \cdot \underbrace{\pi_\theta(a_t \mid s_t)}_{\text{action}} \cdot \underbrace{\pi_\theta(d_t \mid s_t, a_t)^{\mathbb{1}[a_t = \texttt{backoff}]}}_{\text{depth (if backoff)}} \cdot \underbrace{p_\theta(\text{dir}_t \mid s_t, a_t, d_t)^{\mathbb{1}[a_t = \texttt{backoff}]}}_{\text{directive (if backoff)}}\right]$$

Tokens that were generated and then deleted still contribute to the trajectory probability (they were sampled from the policy). The depth decision $d_t$ is also included — the model gets credit for choosing the right rewind distance. The reward $R(\tau)$ only depends on the final state (which excludes deleted tokens), creating the learning signal: better rewind depth → better final answer → higher reward.

### Practical detail: KV cache truncation in PyTorch

```python
# After model emits <backoff> then <depth>:
depth = depth_token_id  # e.g., 1, 2, or 3
depth = min(depth, len(boundaries) - 1)  # can't rewind past prompt

# Pop boundaries and find truncation target
for _ in range(depth):
    boundaries.pop()
target_pos = boundaries[-1]

# Truncate past_key_values (tuple of (key, value) per layer)
past_key_values = tuple(
    (k[:, :, :target_pos, :], v[:, :, :target_pos, :])
    for k, v in past_key_values
)

# Generate directive, then resume
# Position IDs continue from target_pos
position_ids = torch.arange(target_pos, target_pos + directive_len)
```

This is supported by HuggingFace `transformers` — `past_key_values` are plain tensors that can be sliced along the sequence dimension.

---

## Semantic Boundary Detection

Action decisions occur at semantic boundaries within $[K_{\min}, K_{\max}]$ tokens.

**Boundary set:**

$$\mathcal{B} = \mathcal{B}_{\text{punct}} \cup \mathcal{B}_{\text{logic}} \cup \mathcal{B}_{\text{struct}}$$

- $\mathcal{B}_{\text{punct}} = \{$`.` `;` `?` `!` `\n\n`$\}$
- $\mathcal{B}_{\text{logic}} = \{$`therefore` `however` `so` `but` `thus` `hence` `because`$\}$
- $\mathcal{B}_{\text{struct}} = \{$`Step` `Answer:` `Therefore,`$\}$

**Trigger rule:**

$$n_t = \min\!\left(\min\{j > n_{t-1} : y_j \in \mathcal{B} \wedge j - n_{t-1} \geq K_{\min}\},\;\; n_{t-1} + K_{\max}\right)$$

Backoff targets are also aligned to semantic boundaries — the model rewinds to the nearest prior boundary, not to an arbitrary position.

---

## Training Pipeline (Three Phases)

### The Cold Start Problem

The model has never seen $\langle\texttt{backoff}\rangle$, $\langle\texttt{continue}\rangle$, or $\langle\texttt{terminate}\rangle$ in its pretraining data. These are new tokens with randomly initialized embeddings. If we start GRPO directly:

1. $\pi_\theta(\langle\texttt{backoff}\rangle \mid s_t) \approx 0$ due to random initialization
2. GRPO rollouts almost never produce backoff trajectories
3. Without backoff trajectories, GRPO cannot learn that backoff is useful
4. The model converges to a forward-only policy — $\langle\texttt{backoff}\rangle$ is never discovered

This is a classic **exploration failure**. We solve it with a three-phase curriculum.

---

### Phase 1: Warm-up SFT (teach the format)

**Goal:** Teach the model what backoff *looks like* syntactically. Not when to use it optimally — just that it exists and how the token sequence works.

**Data construction: real wrong rollouts (S²R-style, no synthetic perturbation).**

Inspired by S²R (Ma et al., 2025), we construct SFT data from the model's **own** correct and incorrect rollouts, not synthetic number perturbations. Synthetic perturbation (swapping a number in correct reasoning) creates artificial errors that don't match real failure patterns, and the model never learns to recognize its actual mistakes.

```
Script: scripts/generate_sft_real_backoff.py

For each problem x in the training set:
    1. Sample K=8 rollouts from the base model (temp=0.6)
    2. Grade each rollout against gold answer → correct/incorrect
    3. Compute pass_rate = #correct / K
    4. If pass_rate ∈ (0, 1) — problem has both correct and wrong rollouts:
        a. Pick a wrong rollout (genuine failure) and a correct rollout (genuine success)
        b. Stitch: [wrong reasoning] <backoff_3> [locally-meaningful directive] [correct reasoning] → \boxed{answer}
    5. If pass_rate = 0 or 1 — skip (no usable contrast)
```

**Critical design: locally-meaningful directives.** The directive after `<backoff_3>` must reference what's visible at the splice point, not the final wrong answer. The directive has two parts:
1. What went wrong — references the error visible in the preceding text (e.g., "Interpreting 'two times more' as 3x is wrong")
2. What to try — hints from the correct rollout's opening (e.g., "Juice = 2 × Sandwich = 2×4 = $8")

**Example real-backoff training instance:**

```
<think>
[model's actual wrong reasoning on this problem, arriving at wrong intermediate results]
...the total capacity is 2 beds × 2 students = 4, so 30/4 = 8 rooms.
<backoff_3> That gives 8, which is wrong. The pull-out couch adds 1 more student per room: 2×2+1 = 5 per room.
[model's actual correct reasoning on the same problem]
...each room fits 2×2 + 1 = 5 students, so 30/5 = 6 rooms.
</think>
\boxed{6}
```

**Dataset requirements:**
- Needs a training set where the model has **non-trivial failure rate** (10-50% of problems have both correct and wrong rollouts)
- **GSM8K is too easy** for Qwen3-1.7B (~93% pass rate at temp=0.6 → only 15% of problems are usable, yielding ~350 real backoff examples on the full 7.5k train set)
- **MATH** (Hendrycks et al., 2021) is a better fit — competition-level problems where a 1.7B model will fail much more often, producing more wrong rollouts
- All wrong reasoning is self-generated (on-policy) — no distribution shift (SCoRe's key finding)

**SFT objective:** Standard next-token prediction cross-entropy loss on the full sequence, with LoRA:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{j} \log p_\theta(y_j \mid y_{<j})$$

**What this teaches:**
- The embedding space for backoff tokens gets meaningful initialization
- The model learns the syntax: `[wrong reasoning] <backoff_3> [directive] [correct reasoning]`
- The model learns that backoff appears after its own real failure patterns, not after artificial noise

**What this does NOT teach:**
- Optimal backoff policy (when to backoff vs. continue)
- Optimal directive content
- Optimal rewind depth

These are learned in Phase 2 via RL.

---

### Phase 2: GRPO with Progress Reward (learn optimal policy)

**Goal:** The model has learned the format (Phase 1) and can produce backoff trajectories. Now train with outcome reward + dense progress signal so every segment gets credit proportional to how much it helped.

**Reward:**

$$R(\tau) = r(\hat{a}, a^*) + \alpha \sum_{j} r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}_{j-1})$$

The progress signal replaces both the old token penalty and exploration bonus. Backoff is rewarded when it *actually helps* (positive progress after context reclamation) and penalised when it doesn't (undoing correct work). No separate exploration incentive is needed — the dense reward gives fine-grained credit to every action, including the first tentative backoffs.

**GRPO update:**

$$\hat{A}_i = \frac{R(\tau_i) - \mu_G}{\sigma_G}$$

The model learns:
- **When** to backoff (on what kinds of reasoning errors)
- **How far** to rewind (optimal depth $d_t$)
- **What directive** to write (corrective vs. strategic)
- **When NOT** to backoff (on problems it can solve forward-only)

**Convergence monitoring:**
- Track backoff rate, accuracy, average progress, depth distribution
- If backoff rate drops to $< 5\%$: backoff isn't useful for this model/benchmark (H1 fails)
- If backoff rate stays $> 50\%$: model may be over-relying on backoff (check if accuracy is actually improving)
- If average progress stays near zero despite correct answers: the model solves problems in one shot (good, backoff is correctly avoided)

---

### Summary

| Phase | Method | Reward | Purpose | Outcome |
|-------|--------|--------|---------|---------|
| 1 | SFT on synthetic data | Cross-entropy | Teach format + initialize embeddings | Model knows `<backoff>` syntax |
| 2 | GRPO + progress | Correctness + $\alpha \sum r_{\text{prg}}$ | Learn optimal policy via dense signal | Model uses backoff when it helps |

**SFT teaches syntax. GRPO + progress teaches strategy.**

```
Phase 1 (SFT):          "Here's what backoff looks like"
Phase 2 (GRPO+progress): "Every segment gets credit for how much it helped"
```

---

## What Makes This Different from Self-Correction

| | Self-correction (SCoRe, S²R) | Our backoff |
|---|---|---|
| Mechanism | Append "Wait, that's wrong..." tokens | **Delete** wrong tokens from KV cache |
| Context cost | Correction consumes additional context | Context is **reclaimed** |
| Information flow | Old wrong tokens still visible via attention | Old wrong tokens are **gone** |
| Scaling behavior | Degrades under tight context budgets | **Improves** under tight context budgets |
| Training signal | Outcome reward on full (growing) trajectory | Outcome reward on final (potentially shorter) state |
