# Training Challenges: Making Backoff Work with GRPO

## The Core Problem: Non-Standard Trajectories

Standard GRPO (DeepSeek-R1 style) assumes a trajectory is a left-to-right token sequence. One forward pass computes all log-probabilities.

Backoff breaks this assumption. A trajectory with one backoff event:

```
x → [chunk1] → [chunk2] → <backoff_2> → [directive] → [chunk3] → </think>
                ^^^^^^^^
                DELETED from KV cache after backoff
```

After truncation, the KV cache state is:

```
x → [chunk1] → [directive] → [chunk3] → </think>
```

`chunk3` was generated conditioned on `x + chunk1 + directive`, **not** on `x + chunk1 + chunk2 + ...`. A single left-to-right forward pass over the full generated token sequence would give incorrect log-probabilities for everything after the backoff.

---

## Solution: Segment-Wise Forward Passes

Split the trajectory at each backoff event. Compute log-probabilities with separate forward passes, each using the correct context.

**For a trajectory with one backoff:**

**Pass 1 (pre-backoff context):** `x + chunk1 + chunk2 + <backoff_N> + directive`

Collect $\log p_\theta(y_j \mid y_{<j})$ for all tokens in this segment.

**Pass 2 (post-backoff context):** `x + chunk1 + directive + chunk3 + </think>`

Collect $\log p_\theta(y_j \mid y_{<j}^{\text{trunc}})$ for `chunk3` and `</think>` only. The prefix `x + chunk1 + directive` provides the KV cache state that the model actually had when generating `chunk3`.

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
| Generating chunk2 tokens | "These were deleted — was generating them a waste?" |
| Emitting `<backoff_N>` | **"Was it right to undo at this point, and was the depth correct?"** |
| Generating directive tokens | **"Did this directive lead to better reasoning?"** |
| Generating chunk3 tokens | "Did the post-backoff reasoning reach the right answer?" |
| Emitting `</think>` | "Was the answer ready at this point?" |

The deleted tokens (chunk2) still contribute to the trajectory log-probability and receive gradients. If a trajectory with backoff gets higher reward than one without, the model learns:
- To detect when it's going down a bad path (emit `<backoff_N>` earlier)
- To choose the right rewind depth (via the N in `<backoff_N>`)
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

Backoff operates at **per-semantic-boundary granularity**. The depth is encoded directly in the backoff token (`<backoff_1>`, `<backoff_2>`, `<backoff_3>`), so a single token determines both the action and the rewind distance.

- `<backoff_1>` — undo one boundary (fine correction)
- `<backoff_2>` — undo two boundaries (moderate rewrite)
- `<backoff_3>` — undo three boundaries (strategic pivot)

$D_{\max} = 3$. Clamped to `len(boundaries) - 1` to prevent rewinding past the prompt. The model learns optimal rewind distance through RL.

### Maximum backoff count per trajectory

Unbounded backoffs could create infinite loops (backoff → generate → backoff → generate → ...). Set a hard cap:

$$B_{\max} \in \{2, 3\}$$

After $B_{\max}$ backoffs, backoff tokens are suppressed. The model can only continue generating or terminate with `</think>`.

### Directive length cap

Cap directive at $K_{\text{dir}} \leq 20$ tokens of free text. After the directive, generation resumes normally.

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
| Backoff rate collapses to 0% (model never emits `<backoff_N>`) | Separate KL coefficient for backoff tokens: $\beta_{\text{action}} < \beta_{\text{reason}}$ so the model is freer to explore backoff |
| Backoff rate explodes to 80%+ (model always backs off) | Reduce $\alpha$ or add a small backoff penalty |
| Progress signal is too noisy (greedy probes disagree) | Increase `num_probes` for smoother estimates |
| Directive tokens are always identical / generic | Increase entropy bonus specifically on directive tokens |
| Post-backoff reasoning repeats deleted content | Add a similarity penalty between deleted segment and post-backoff segment |

### If we do propose something, the natural angle

The most defensible custom contribution would be **heterogeneous KL regularization**:

$$\mathrm{KL}_{\text{het}}(\pi_\theta \| \pi_{\text{ref}}) = \beta_{\text{reason}} \cdot \mathrm{KL}_{\text{reason}}(\pi_\theta \| \pi_{\text{ref}}) + \beta_{\text{action}} \cdot \mathrm{KL}_{\text{action}}(\pi_\theta \| \pi_{\text{ref}})$$

**Justification:** Backoff tokens (`<backoff_1/2/3>`) don't exist in the pretrained model — they're new tokens with mean-initialized embeddings. The pretrained reference policy $\pi_{\text{ref}}$ assigns near-uniform probability to them. A standard KL penalty therefore penalizes ANY non-uniform backoff distribution, even a clearly useful one. Using a lower $\beta_{\text{action}}$ lets the model develop strong backoff preferences without being pulled back to the meaningless reference distribution.

But again — only propose this if vanilla GRPO shows the problem.

---

## SFT Data: Lessons Learned

### Synthetic perturbation doesn't work

The original approach (`scripts/generate_sft_from_rollouts.py`) perturbed numbers in correct rollouts to create "wrong" steps. Problems:
- The "errors" are artificial — swapping `200` to `214` in otherwise correct reasoning
- The model never sees its own real failure patterns
- Directives reference specific wrong/correct numbers but the surrounding reasoning is coherent, making the backoff feel arbitrary

### Real wrong rollouts (S²R-style) are better

`scripts/generate_sft_real_backoff.py` uses the model's own incorrect rollouts stitched with correct ones:
- Wrong reasoning reflects genuine failure modes (misinterpretation, wrong approach, arithmetic confusion)
- Directives must be **locally meaningful** — referencing what's visible at the splice point, not the final wrong answer
- No clean/synthetic examples needed — the model already knows how to reason correctly; SFT Phase 1 only needs to teach the backoff format

### Dataset difficulty matters

| Dataset | 1.7B pass rate (temp=0.6, K=8) | Usable problems | Verdict |
|---------|-------------------------------|-----------------|---------|
| GSM8K | ~93% | ~15% (most problems all-correct) | Too easy |
| MATH | Expected ~30-50% | ~60-80% (much more wrong rollouts) | Better fit |

GSM8K yields only ~350 real backoff examples from 7.5k problems. MATH should yield 3-5x more usable examples due to higher failure rate.

### Grading pitfalls

`extract_boxed_answer()` must handle diverse LaTeX formatting in `\boxed{}`:
- `\boxed{\$840}` → strip `\$` → `840`
- `\boxed{\dfrac{8}{3}}` → handle nested braces and fractions
- `\boxed{4,000,000}` vs gold `4` → units mismatch causes false negatives

These grading errors create **poisoned training data** — examples where the "wrong" rollout was actually correct, teaching the model to backoff from correct reasoning. Robust answer extraction and normalization implemented in `src/data/math.py`.
