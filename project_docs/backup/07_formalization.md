# Formal Problem Statement: Self-Reflective Adaptive Reasoning via RL

## Motivation: Why the LLM Itself Should Be the Controller

An external MLP controller π_MLP(a | h_L) conditions on a single compressed hidden vector h_L. This creates an information bottleneck: the controller cannot attend to earlier reasoning steps, detect internal contradictions, or assess trajectory coherence. The LLM's own next-token distribution, by contrast, conditions on the full sequence through self-attention and already encodes uncertainty, coherence, and reasoning progress.

**Claim:** The policy should be the LLM's own conditional distribution over a small set of action tokens, trained end-to-end via RL on the same LoRA parameters that generate reasoning.

---

## MDP Formulation

We model the reasoning process as a finite-horizon Markov Decision Process M = (S, A, P, R, γ).

### State Space

At decision step t, the state is the full token history:

    s_t = (x, y_{1:n_t})

where x ∈ V* is the input prompt and y_{1:n_t} ∈ V* are all tokens generated up to step t (across all chunks, verifications, and branches). The LLM encodes s_t implicitly through its causal attention over the concatenated sequence.

### Action Space

We extend the vocabulary V with four special tokens:

    A = { ⟨continue⟩, ⟨verify⟩, ⟨branch⟩, ⟨terminate⟩ }

After generating each chunk of K reasoning tokens, the model is prompted to emit exactly one action token a_t ∈ A.

### Transition Dynamics

The transition P(s_{t+1} | s_t, a_t) is defined by the action semantics:

    a_t = ⟨continue⟩:
        Generate next K tokens y_{n_t+1 : n_t+K} ~ p_θ(· | s_t)
        s_{t+1} = (x, y_{1:n_t+K})

    a_t = ⟨verify⟩:
        Append verification prompt v to the context
        Generate verification segment y^{ver} ~ p_θ(· | s_t, v)
        s_{t+1} = (x, y_{1:n_t}, v, y^{ver})

    a_t = ⟨branch⟩:
        Cache current trajectory as candidate c_1 = y_{1:n_t}
        Reset generation to a branch point n_b ≤ n_t
        Generate alternative continuation y^{alt} ~ p_θ(· | x, y_{1:n_b})
        s_{t+1} = (x, y_{1:n_b}, y^{alt})     [with c_1 stored for later selection]

    a_t = ⟨terminate⟩:
        Extract final answer â = Extract(s_t)
        Episode ends

The episode also terminates if t reaches a maximum horizon T_max.

### Policy

The policy is the LLM's own conditional distribution over action tokens:

    π_θ(a_t | s_t) = p_θ(a_t | x, y_{1:n_t}, ⟨decide⟩)

where ⟨decide⟩ is a fixed prompt suffix (e.g., a special token or short instruction) that cues the model to emit an action token. θ represents the base model weights plus LoRA adapters — the same parameters that generate reasoning.

This is critical: there is no separate controller network. The policy shares all parameters with the generator. The LLM's self-attention over the full trajectory IS the decision mechanism.

### Reward

The reward is assigned at episode termination:

    R(τ) = r(â, a*) − λ_tok · T(τ) − λ_br · B(τ) − λ_ver · V(τ)

where:
- r(â, a*) ∈ {0, 1} is the correctness signal (exact match for math, pass@1 for code)
- T(τ) = total reasoning tokens generated (including verifications and branches)
- B(τ) = number of ⟨branch⟩ actions taken
- V(τ) = number of ⟨verify⟩ actions taken
- λ_tok, λ_br, λ_ver ≥ 0 are cost coefficients

The reward is bounded: R(τ) ∈ [R_min, R_max] with clipping to prevent explosion.

### Discount Factor

γ = 1 (undiscounted, since episodes are short and we care about total return).

---

## Optimization Objective

The goal is to find θ* that maximizes expected return while staying close to the pretrained reference policy π_ref:

    θ* = arg max_θ  E_{x ~ D} E_{τ ~ π_θ(·|x)} [R(τ)]  −  β · E_{x ~ D} [KL(π_θ(·|x) ‖ π_ref(·|x))]

The KL term prevents catastrophic forgetting of the base model's reasoning ability.

---

## Training Algorithms

### Option A: PPO (Proximal Policy Optimization)

Collect rollouts under π_{θ_old}, compute advantages via GAE:

    δ_t = R_t + γ V_φ(s_{t+1}) − V_φ(s_t)
    Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

Update with clipped surrogate:

    L_PPO(θ) = E_t [ min( ρ_t(θ) · Â_t,  clip(ρ_t(θ), 1−ε, 1+ε) · Â_t ) ]

where ρ_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t).

Full loss:

    L(θ, φ) = −L_PPO(θ) + c_v · ‖V_φ(s_t) − R_t^{target}‖² − c_e · H(π_θ(·|s_t)) + β · KL(π_θ ‖ π_ref)

**Requires:** A value head V_φ(s_t) = Linear(h_t^{last}, 1), adding a single linear layer on the LLM's last hidden state.

### Option B: GRPO (Group Relative Policy Optimization)

For each prompt x, sample a group of G trajectories {τ_1, ..., τ_G} under π_{θ_old}. Compute group-normalized advantages:

    Â_i = (R(τ_i) − μ_G) / σ_G

where μ_G = (1/G) Σ_i R(τ_i) and σ_G = std({R(τ_i)}).

Update:

    L_GRPO(θ) = E_x E_{i=1..G} [ min( ρ_i(θ) · Â_i,  clip(ρ_i(θ), 1−ε, 1+ε) · Â_i ) ] − β · KL(π_θ ‖ π_ref)

**Advantage over PPO:** No value network needed. The group comparison provides a variance-reduced baseline automatically. More stable for sparse binary rewards (correct/incorrect).

**Trade-off:** Requires G forward passes per prompt per update. With G=4-8 and LoRA, this fits in a single-GPU budget.

---

## Architectural Comparison

### Current design (separate MLP controller):

    h = LLM_θ(x, y_{1:n_t})              # forward pass for reasoning
    z = MLP_ψ(h^{last}, stats, domain)    # separate controller
    π(a|s_t) = softmax(W_a · z)           # action from controller
    V(s_t) = w_v^T z                      # value from controller

Parameters updated: θ (LoRA) and ψ (MLP), with separate learning rates.

### Proposed design (LLM-as-controller):

    h = LLM_θ(x, y_{1:n_t}, ⟨decide⟩)    # single forward pass
    π(a|s_t) = softmax(h^{last}_{A})       # action logits = LLM's logits restricted to A ⊂ V
    V(s_t) = w_v^T h^{last}               # (PPO only) single linear layer

Parameters updated: θ (LoRA) only. The value head w_v is the only additional parameter (eliminated under GRPO).

### Why this is better:

1. **No information bottleneck.** The policy π(a|s_t) attends to ALL positions in the reasoning trajectory via self-attention, not just a compressed vector.

2. **Shared representations.** The same features that make the model good at reasoning also inform action decisions. Verification decisions benefit from the model's own uncertainty signals.

3. **Fewer parameters.** No separate MLP to train. Under GRPO, the only trainable parameters are the LoRA adapters.

4. **Theoretical grounding.** The policy is a restriction of the LLM's own next-token distribution to a 4-token subset. Since the LLM already models p(next_token | context), we are leveraging its full capacity for the decision.

---

## Compute Accounting

For fair comparison across methods, define normalized compute cost:

    C(τ) = Σ_{t=1}^{|τ|} c(a_t)

where:
    c(⟨continue⟩) = K                    # K tokens generated
    c(⟨verify⟩)   = K_ver + 1            # verification segment + decision
    c(⟨branch⟩)   = K_br + 1             # branch generation + decision
    c(⟨terminate⟩) = 1                   # just the decision token

Total model forward passes:

    F(τ) = |τ| + Σ (additional passes for verify/branch)

Report both T(τ) (total tokens) and F(τ) (total forward passes) to separate token-level and call-level efficiency.

---

## Key Differences from Original Plan

| Aspect | Original (MLP controller) | Proposed (LLM-as-controller) |
|--------|---------------------------|------------------------------|
| Decision maker | Separate 2-layer MLP | LLM's own next-token distribution |
| Information access | Last hidden state + stats | Full self-attention over trajectory |
| Extra parameters | ~200K (MLP + heads) | ~0 (just LoRA; value head optional) |
| Training signal | PPO only | PPO or GRPO |
| Action representation | MLP output logits | LLM vocabulary logits restricted to A |
| Forward passes per decision | 1 (LLM) + 1 (MLP) | 1 (LLM, action token is next token) |
