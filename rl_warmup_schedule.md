# RL Training Schedule: 3-Phase Action-Injection Warmup

## Motivation

Our SFT warmup left the model emitting action tokens in only ~18% of outputs (and `<refine>` at 0%). Vanilla GRPO with ALP reward alone would not encourage action tokens to emerge — the reward is indifferent to whether they're used. We need a way to give GRPO a dense learning signal on action decisions.

The idea: use **action-injection as an RL warmup**, then gradually relax it. The model first learns action-conditioned policies under forced choices, then learns to emit actions natively.

## Phases

### Phase A: Hard Mask (RL warmup)
- **Mechanism:** At every paragraph-break token (`\n\n`, `.\n\n`, `:\n\n`, etc.), mask all non-action logits to `-inf`. Model samples only from `{<continue>, <refine>, <terminate>}`.
- **Effect:** Every rollout contains explicit action decisions. GRPO has dense signal to learn "in which contexts does each action lead to high ALP reward?"
- **Policy shape:** The model's action-choice logits are trained under a restricted decision space.
- **Analogy:** Like teacher forcing — the structure is imposed, the model fills in the choice.

### Phase B: Soft Bias (transition)
- **Mechanism:** At paragraph breaks, add `+10` bias to action-token logits but do NOT mask. Other tokens can still win.
- **Effect:** Model usually emits actions (biased) but sometimes continues reasoning freely. Tests whether Phase A generalized.
- **Purpose:** Bridge between forced and free generation. Gradually reduces the structural prior.

### Phase C: No Injection (free generation)
- **Mechanism:** No logits modification. Model samples freely from the full vocabulary.
- **Effect:** The "real" test — did Phase A + B shape the model's intrinsic policy, or was it just mimicking the mask?
- **This is where H3 (action distribution by difficulty) becomes meaningful.** Under Phase A, distributions are largely imposed, not learned.

## Analogies

- **Teacher forcing → free generation** (seq2seq training)
- **Exploration schedule decay** (RL)
- **Curriculum learning** — restricted → full action space
- **Behavioral cloning → environment interaction** (imitation learning → RL)

## Experimental Value

- The 3-phase schedule is itself a contribution worth documenting.
- Enables a clean ablation: "no warmup" (Phase C directly) vs "warmup → release" (A → C).
- H3 only meaningfully answerable in Phase C.
- H1 (Pareto) and H2 (token-difficulty) can be measured at the end of each phase to show progression.

## Implementation Notes

- All three phases are already implemented via the pluggable `ActionStrategy` interface:
  - `HardMask` → Phase A
  - `SoftBias` → Phase B
  - `NoInjection` → Phase C
- Switching between phases = change one config field (`action_strategy`).
- Same GRPO trainer, same ALP reward, same checkpoint chain.

## Proposed Minimum Schedule (class project)

If full 3-phase is too expensive, the minimum viable experiment:

1. **Run Phase A** for N steps → checkpoint A
2. **Run Phase C from checkpoint A** for M steps → checkpoint C
3. Evaluate both checkpoints on MATH-500 + compare to base SFT model

This gives us: baseline → SFT → GRPO-A → GRPO-C progression, with the A→C transition being the headline result (did the model retain action behaviors when mask released?).

## Open Questions

- How long should Phase A run? Proposal: until reward plateaus, or fixed budget.
- Should Phase B be skipped entirely (binary hard→free) or kept as a gradual release?
- How to measure "did the model retain actions in Phase C?" — report action emission rate % over training.
- Should `<refine>` be disabled initially (only 2-action hard mask) and reintroduced later? Could help since `<refine>` is 0% from SFT.
