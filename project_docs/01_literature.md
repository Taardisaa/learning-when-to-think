# Key Literature

## Primary Sources (Verified)

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Snell et al., "Scaling LLM Test-Time Compute Optimally..." | ICLR 2025 Oral | Shows adaptive per-prompt compute allocation matters; mainly allocates amount, not action type |
| s1: Simple test-time scaling | 2025 preprint | Makes controllable reasoning-budget scaling practical; only scales length |
| Yang et al., "Towards Thinking-Optimal Scaling..." | NeurIPS 2025 | Optimal reasoning-length distributions; still length-allocation, not action-level |
| MRT: "Optimizing Test-Time Compute via Meta RL Finetuning" | ICML 2025 | Meta-RL for test-time compute; related but different formulation |

## Novelty Threats

| Paper | Why It's a Threat |
|-------|-------------------|
| SCoRe: "Training LMs to Self-Correct via RL" (Kumar et al., ICLR 2025 Oral) | Two-stage RL for self-correction with reward shaping bonus α(r₂-r₁). Key findings: SFT fails due to distribution shift + behavior collapse; on-policy RL essential. Fixed 2-turn protocol, no KV cache reclamation. |
| S²R: "Teaching LLMs to Self-verify and Self-correct via RL" (Ma et al., 2025) | Interleaved solve→verify→correct loop with only 3.1k SFT samples. Key for our data construction: uses model's own wrong rollouts + confirmative verification. Outcome-level RL for strong models, process-level for weak. |
| "Rewarding Progress: Scaling Automated Process Verifiers" (ICLR 2025 Spotlight) | Best modern verifier/search comparison for step-level reward shaping |

## Key Related Work by Theme

### Test-Time Scaling
- Huang et al. (2025) — per-query dynamic compute with latency awareness
- Liu et al. (2025) — prompting strategies change rank as sampling budgets grow
- Chen et al. (2025) — systems innovations for TTS latency on edge hardware

### Verification & Multi-Path
- Forest of Thought (Bi et al.) — multiple reasoning trees with self-correction
- Reflexion (Shinn et al.) — verbal feedback + episodic memory across attempts
- Singhi et al. (2025) — solution generation usually more efficient than verification
- ParaThinker (Wen et al.) — width scaling via parallel paths

### Self-Correction & Error Detection
- Huang et al. (2024), "Large Language Models Cannot Self-Correct Reasoning Yet" — shows prompting-based self-correction is unreliable; models change correct answers to wrong as often as they fix mistakes. Key pessimistic baseline: intrinsic self-correction via prompting ≈ noise
- STaR / Self-Taught Reasoner (Zelikman et al., 2022) — iteratively trains on own correct traces; self-improvement via filtering, not online correction
- RISE (Qu et al., 2024) — multi-turn RL for iterative refinement; shows improvement when model gets multiple attempts, but appends rather than rewinding
- GLoRE (Havrilla et al., 2024) — trains separate verifier to score intermediate steps; external signal enables correction but requires a second model
- DeepSeek-R1 (2025) — self-correction behavior ("wait, that's wrong") emerges from GRPO training on outcome reward alone, without explicit correction tokens. Key positive evidence that RL can discover self-correction without teaching it explicitly

**Critical distinction for our work**: prompting-based self-correction fails (Huang et al.), but RL-trained self-correction can work (DeepSeek-R1, RISE, SCoRe, S²R). Our approach differs from all of these: we make backoff a discrete, learned action with KV cache truncation (reclaiming context), rather than appending corrections that consume additional context. Our SFT data construction borrows from S²R: stitching real wrong rollouts with correct ones (on-policy, no synthetic perturbation), with locally-meaningful directives at the splice point.

**Key lessons adopted from SCoRe/S²R:**
- On-policy data is non-negotiable — off-policy correction traces cause distribution shift (SCoRe §4)
- Synthetic perturbation (number swaps) doesn't match real failure patterns — use the model's own wrong rollouts instead (S²R §2.2.2)
- Reward shaping prevents behavior collapse toward "just get it right the first time" (SCoRe §5.2)
- Need a dataset where the model actually fails: GSM8K is too easy for 1.7B models (~93% pass rate), MATH is better

### RL for Reasoning
- PPO (Schulman et al., 2017) — stable policy gradient method
- GRPO (Shao et al., 2024) — group relative policy optimization, no value network needed; used in DeepSeek-R1
- Setlur et al. (2025) — RL with negative gradients for exploratory search

## Identified Gaps (From Synthesis)

1. No policy that learns non-monotonic reasoning (backoff / undo) as a first-class action
2. Self-correction work appends corrections but never reclaims context (no KV cache truncation)
3. RL-for-reasoning papers don't tightly connect to correctness-cost tradeoffs on standard benchmarks
4. Termination and backoff decisions are underexplored as learned actions
5. Cross-domain evidence for adaptive compute policies is limited
6. Efficiency metrics are fragmented across papers
