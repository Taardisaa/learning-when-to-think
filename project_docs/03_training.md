# Training Challenges: Making Backoff Work with GRPO

## The Core Problem: Non-Standard Trajectories

Standard GRPO (DeepSeek-R1 style) assumes a trajectory is a left-to-right token sequence. One forward pass computes all log-probabilities.

Backoff breaks this assumption. A trajectory with one backoff event:

```
x → [chunk1] → <continue> → [chunk2] → <backoff> → [directive] → [chunk3] → <terminate>
                              ^^^^^^^^
                              DELETED from KV cache after backoff
```

After truncation, the KV cache state is:

```
x → [chunk1] → [directive] → [chunk3] → <terminate>
```

`chunk3` was generated conditioned on `x + chunk1 + directive`, **not** on `x + chunk1 + chunk2 + ...`. A single left-to-right forward pass over the full generated token sequence would give incorrect log-probabilities for everything after the backoff.

---

## Solution: Segment-Wise Forward Passes

Split the trajectory at each backoff event. Compute log-probabilities with separate forward passes, each using the correct context.

**For a trajectory with one backoff:**

**Pass 1 (pre-backoff context):** `x + chunk1 + <continue> + chunk2 + <backoff> + directive`

Collect $\log p_\theta(y_j \mid y_{<j})$ for all tokens in this segment.

**Pass 2 (post-backoff context):** `x + chunk1 + directive + chunk3 + <terminate>`

Collect $\log p_\theta(y_j \mid y_{<j}^{\text{trunc}})$ for `chunk3` and `<terminate>` only. The prefix `x + chunk1 + directive` provides the KV cache state that the model actually had when generating `chunk3`.

**Full trajectory log-probability:**

$$\log \pi_\theta(\tau) = \underbrace{\sum_{j \in \text{seg}_1} \log p_\theta(y_j \mid y_{<j})}_{\text{pass 1: pre-backoff}} \;+\; \underbrace{\sum_{j \in \text{seg}_2} \log p_\theta(y_j \mid y_{<j}^{\text{trunc}})}_{\text{pass 2: post-backoff}}$$

**For $k$ backoff events:** $k+1$ forward passes. Each pass covers one segment between consecutive backoff points (or start/end of trajectory).

**Cost:** $\sim(k+1)\times$ the compute of a standard single-pass trajectory. In practice, most trajectories will have 0–2 backoffs, so this is 1–3x cost.

---

## GRPO Update with Backoff Trajectories

For each prompt $x$, sample $G$ trajectories $\{\tau_1, \ldots, \tau_G\}$:

**Step 1: Rollout.** Generate each trajectory using the generation loop (with actual KV cache truncation when backoff occurs). Record:
- All tokens generated (including deleted ones)
- Backoff positions and directive texts
- Final answer

**Step 2: Reward.** Compute $R(\tau_i) = r(\hat{a}_i, a^*) + \alpha \sum_j r_{\text{prg}}(\mathbf{z}_j; \mathbf{c}_{j-1})$

**Step 3: Advantages.** Group-relative:

$$\hat{A}_i = \frac{R(\tau_i) - \mu_G}{\sigma_G}$$

**Step 4: Importance ratios.** For each trajectory, compute $\log \pi_\theta(\tau_i)$ using segment-wise forward passes (as above). Then:

$$\rho_i = \exp\!\left(\log \pi_\theta(\tau_i) - \log \pi_{\theta_{\text{old}}}(\tau_i)\right)$$

**Step 5: Clipped surrogate update.**

$$\mathcal{L} = \mathbb{E}_{i}\!\left[\min\!\left(\rho_i \hat{A}_i,\; \text{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)\, \hat{A}_i\right)\right] - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$$

---

## What the Model Learns (Gradient Flow)

The gradient flows through **all** decisions in the trajectory:

| Decision | Gradient signal |
|----------|----------------|
| Generating chunk1 tokens | "Was this reasoning helpful for the final answer?" |
| Emitting $\langle\texttt{continue}\rangle$ after chunk1 | "Was it right to keep going here?" |
| Generating chunk2 tokens | "These were deleted — was generating them a waste?" |
| Emitting $\langle\texttt{backoff}\rangle$ | **"Was it right to undo at this point?"** |
| Generating directive tokens | **"Did this directive lead to better reasoning?"** |
| Generating chunk3 tokens | "Did the post-backoff reasoning reach the right answer?" |
| Emitting $\langle\texttt{terminate}\rangle$ | "Was the answer ready at this point?" |

The deleted tokens (chunk2) still contribute to the trajectory log-probability and receive gradients. If a trajectory with backoff gets higher reward than one without, the model learns:
- To detect when it's going down a bad path (emit $\langle\texttt{backoff}\rangle$ earlier)
- To write useful directives (that lead to better post-backoff reasoning)
- To avoid generating the bad tokens in the first place (over time)

---

## Practical Considerations

### Batching trajectories with different backoff counts

Trajectories in the same batch may have 0, 1, or 2+ backoff events, requiring different numbers of forward passes. Options:

1. **Pad to max segments:** Wasteful but simple. If max backoffs in batch is 2, all trajectories get 3 passes (some with dummy segments).
2. **Dynamic batching:** Group trajectories by backoff count. Run 0-backoff trajectories together in one pass, 1-backoff trajectories with two passes, etc.
3. **Sequential per-trajectory:** Just loop over trajectories. Simplest to implement, fine for small $G$.

Recommendation: start with option 3 (simplest), optimize later if needed.

### Backoff target selection

Backoff operates at **per-semantic-boundary granularity**. After emitting $\langle\texttt{backoff}\rangle$, the model emits a depth token $d_t \in \mathcal{D} = \{\langle 1 \rangle, \ldots, \langle D_{\max} \rangle\}$ specifying how many semantic boundaries to rewind.

- $\langle 1 \rangle$ — undo one sentence (fine correction)
- $\langle 3 \rangle$ — undo three sentences (moderate rewrite)
- $\langle D_{\max} \rangle$ — undo many sentences (strategic pivot)

$D_{\max} = 5$ is a reasonable default. Clamped to `len(boundaries) - 1` to prevent rewinding past the prompt.

The depth decision adds one extra token to the trajectory likelihood — the model learns optimal rewind distance through RL.

### Maximum backoff count per trajectory

Unbounded backoffs could create infinite loops (backoff → generate → backoff → generate → ...). Set a hard cap:

$$B_{\max} \in \{2, 3\}$$

After $B_{\max}$ backoffs, the $\langle\texttt{backoff}\rangle$ action is masked out. The model can only continue or terminate.

### Directive length cap

Cap directive at $K_{\text{dir}} \leq 20$ tokens. If the model hasn't emitted $\langle\texttt{continue}\rangle$ by then, force-append it and resume generation.

---

## On Custom RL Methods (RH #5)

### Decision: GRPO + MRT-style progress reward

The dense per-segment progress signal (adapted from MRT, Qu et al. 2025) is a principled addition to vanilla GRPO, not a custom RL algorithm. It replaces the ad-hoc token penalty ($\lambda_{\text{tok}} \cdot T_{\text{net}}$) and exploration bonus ($\lambda_{\text{explore}}$) with a single, well-motivated dense reward that gives each segment credit proportional to how much it helped reach the correct answer.

This is a clean contribution because:
- It uses standard GRPO mechanics (clipped surrogate + KL)
- The only addition is a reward shaping term with clear theoretical motivation (cumulative regret minimisation)
- The paper's novelty remains **backoff as a reasoning mechanism** — the progress reward is how we train it effectively

### When to reconsider

If during training we observe specific failure modes, a targeted fix becomes justified:

| Failure mode | Possible custom fix |
|-------------|-------------------|
| Backoff rate collapses to 0% (model never backs off) | Separate KL coefficient for action tokens: $\beta_{\text{action}} < \beta_{\text{reason}}$ so the model is freer to explore backoff |
| Backoff rate explodes to 80%+ (model always backs off) | Reduce $\alpha$ or add a small backoff penalty |
| Progress signal is too noisy (greedy probes disagree) | Increase `num_probes` for smoother estimates |
| Directive tokens are always identical / generic | Increase entropy bonus specifically on directive tokens |
| Post-backoff reasoning repeats deleted content | Add a similarity penalty between deleted segment and post-backoff segment |

### If we do propose something, the natural angle

The most defensible custom contribution would be **heterogeneous KL regularization**:

$$\mathrm{KL}_{\text{het}}(\pi_\theta \| \pi_{\text{ref}}) = \beta_{\text{reason}} \cdot \mathrm{KL}_{\text{reason}}(\pi_\theta \| \pi_{\text{ref}}) + \beta_{\text{action}} \cdot \mathrm{KL}_{\text{action}}(\pi_\theta \| \pi_{\text{ref}})$$

**Justification:** Action tokens ($\langle\texttt{backoff}\rangle$, $\langle\texttt{continue}\rangle$, $\langle\texttt{terminate}\rangle$) don't exist in the pretrained model — they're new tokens with randomly initialized embeddings. The pretrained reference policy $\pi_{\text{ref}}$ assigns near-uniform probability to them. A standard KL penalty therefore penalizes ANY non-uniform action distribution, even a clearly useful one (like "always continue on easy problems, backoff on hard ones"). Using a lower $\beta_{\text{action}}$ lets the model develop strong action preferences without being pulled back to the meaningless reference distribution.

But again — only propose this if vanilla GRPO shows the problem.
