# Hypotheses

## H1: Termination-First Control Captures Most Gains (Mandatory Baseline)

A minimal binary controller (continue/terminate) should capture most practical efficiency gains.

**Success criteria:**
- At matched GSM8K accuracy, reasoning tokens decrease by >=15% vs budget-forced fixed CoT
- On HumanEval, compute-per-correct-answer improves by >=8%
- A simple 3-bin difficulty router recovers <90% of the gain (proving online RL is needed)

**Failure:** Token savings <10%, accuracy drops >1.5 points, or difficulty router matches 90%+ of gain

**Why this matters:** If termination alone explains everything, richer action spaces aren't needed. This must be beaten before claiming value for verify/branch.

## H2: Phase-Based Probing Beats Stepwise Micromanagement

A short "diagnostic probe" at the start should decide a coarse reasoning mode (early_terminate / continue_linear / branch_once / late_verify), rather than making action decisions at every step.

**Success criteria:**
- Probe-based controller matches or exceeds stepwise 4-action controller accuracy
- Uses >=15% fewer reasoning tokens across GSM8K+HumanEval combined
- Beats a purely upfront difficulty router by >=5% on compute-per-correct-answer

**Failure:** Token savings <10% vs stepwise, accuracy falls >2 points, or simple router matches within 3%

## H3: Verification Should Be Sparse, Late, and Conflict-Triggered

Verification should fire on <20% of trajectories, mostly near the end, and only when the controller's preferred action disagrees with the model's apparent textual certainty.

**Success criteria:**
- Verification rate <20%
- Conflict-triggering beats always-verify by >=8% and confidence-triggering by >=5% on compute-per-correct-answer
- No accuracy loss

**Failure:** Verification stays frequent without benefit, removing verify changes compute-per-correct by <3%

## H4: Cross-Domain Transfer Is a Stress Test, Not an Assumption

A shared controller (math + code) should stay within 1 accuracy point of per-domain specialists while reducing cross-domain variance in compute-per-correct by >=20%.

**Failure:** Shared controller underperforms specialists by >8% on compute-per-correct, or accuracy drops >2 points

## The Sharp Proposal (Summary)

1. Start with termination-first control as the mandatory baseline to beat
2. Test phase-based probing as the main novel extension
3. Treat verification as guilty until proven useful
4. Frame cross-domain transfer as an adversarial test
