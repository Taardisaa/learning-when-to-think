# Semantic Boundary Probing: Learning Step Boundaries from Hidden States

## Motivation

Our `BoundaryTracker` (`src/boundary.py`) uses heuristic rules — punctuation, logic connectives ("therefore", "however"), structural markers ("Step", "Answer:") — to detect where one reasoning step ends and the next begins. These boundaries determine where `<backoff_N>` tokens can be placed and how chunks are defined for the progress reward.

The heuristics work but have fundamental limits:
- **Brittle**: miss valid boundaries that don't match keyword patterns (e.g., "I see that..." or "The key insight is...")
- **Language-dependent**: hardcoded English connectives
- **Not principled**: the model likely already "knows" where its reasoning steps begin and end — this information should be extractable from its hidden states

## Boundary Definition

A **semantic boundary** is a transition point between two distinct **reasoning steps** in a CoT trace. It is NOT a sentence boundary — it operates at a coarser granularity. A single reasoning step (chunk) typically spans 1-4 sentences and corresponds to one atomic cognitive operation.

### What constitutes a reasoning step (chunk)

Each chunk is a self-contained unit of reasoning that accomplishes one thing:

| Chunk type | Example | Typical size |
|------------|---------|-------------|
| Problem interpretation | "So I need to find how many integers satisfy..." | 1-2 sentences |
| Equation setup | "Let x be the number of apples. Then 3x + 5 = 20." | 1-3 sentences |
| Algebraic manipulation | "Subtracting 5 from both sides: 3x = 15. Dividing by 3: x = 5." | 1-4 sentences |
| Case analysis (one case) | "If n is even, then n² is divisible by 4, so..." | 2-4 sentences |
| Sub-computation | "First compute C(8,2) = 28." | 1-2 sentences |
| Verification / sanity check | "Let me verify: plugging x=5 back in, 3(5)+5 = 20. ✓" | 1-2 sentences |
| Course correction | "Wait, that can't be right because... Let me reconsider." | 1-3 sentences |
| Answer extraction | "Therefore the answer is 5." | 1 sentence |

### Where boundaries go

Boundaries are placed **between** chunks — at the point where the model transitions from one reasoning operation to the next.

**Valid boundary points:**
- Between problem interpretation and equation formulation
- Between setting up an equation and solving it
- Between computing an intermediate result and applying it
- Between analyzing one case and moving to another
- Before a course correction ("Wait...", "Actually...", "Hmm, but...")
- Between solving and verifying
- Between verification and final answer extraction
- After a paragraph break (`\n\n`) that marks a topic shift

**NOT boundary points:**
- Between consecutive lines of the **same** algebraic simplification (those are one chunk)
- Mid-sentence or mid-formula
- Between a connective and its clause ("Therefore" and "x = 5" are part of the same chunk)
- Between two sentences that elaborate the same point without advancing the reasoning

### Granularity calibration

The right granularity is: **a boundary marks where you could meaningfully "rewind to" during backoff.** If rewinding to a point wouldn't let you restart a distinct reasoning step, it's not a real boundary — it's just a sentence break within a step.

Concretely, chunks should average **15-80 tokens** (matching `BoundaryTracker`'s `k_min=15`, `k_max=80`). A rollout of ~300 tokens should have roughly 5-15 boundaries. The number of boundaries scales roughly linearly with rollout length.

### Relationship to discourse segmentation

Our semantic boundaries are related to but distinct from EDUs (Elementary Discourse Units) in discourse analysis:

| | EDU (discourse) | Our semantic boundary |
|---|---|---|
| Granularity | Clause-level (~5-15 tokens) | Reasoning-step-level (~15-80 tokens) |
| What it segments | Rhetorical structure of text | Logical structure of reasoning |
| One chunk contains | 1 EDU | 1-4 EDUs |
| Defined by | Syntactic/rhetorical transitions | Cognitive operation transitions |

Our boundaries are coarser — each chunk may contain multiple EDUs, but they all serve a single reasoning purpose.

## Related Work

Several recent papers establish that LLM hidden states encode rich structural information:

| Paper | Finding | Relevance |
|-------|---------|-----------|
| Hewitt 2019 (structural probes) | BERT's hidden states encode full parse trees, recoverable via linear transformation | Proves syntax structure is encoded; reasoning step structure should be too |
| Zhang 2025 (self-verification) | Reasoning model hidden states encode intermediate answer correctness (ROC-AUC > 0.8, often linear) | Step-level correctness is encoded; step *boundaries* should be as well |
| Afzal 2025 (knowing before saying) | CoT success is predictable from hidden states before generation starts | Hidden states encode reasoning trajectory info early |
| Damirchi 2025 (TaT) | Layer-wise displacement ($h_{\ell+1} - h_\ell$) captures reasoning validity better than raw activations, generalizes across tasks | Displacement may be a better input representation for our probe |
| Fan 2025 (DISRPT) | Discourse segmentation achieves 90% F1 with token-level classification on encoder models | Validates that text can be segmented into semantic units with high accuracy |

Key insight from Zhang 2025: they segment CoT into chunks using the **same keyword heuristics** we use ("wait", "alternatively", "let me reconsider") — then prove that correctness at those boundaries is encoded in hidden states. If correctness is encoded, the boundaries themselves almost certainly are too. Nobody has probed for boundary structure directly yet.

## Approach

Three-step pipeline:

### Step 1: Boundary Annotation with Qwen3-32B

Use a large model (Qwen3-32B, non-thinking mode) to annotate semantic boundaries in our existing rollout data. The model receives a CoT reasoning trace and outputs character offsets where reasoning steps transition.

- **Input**: rollout text from `data/rollouts_grouped_math_Qwen3-1.7B.jsonl`
- **Output**: `data/boundary_annotations.jsonl` with char offsets per rollout
- **Scale**: ~2000 rollouts (mix of correct and incorrect)
- **Quality control**: validate against heuristic (expect high agreement on obvious boundaries + additional valid ones the heuristic misses)

### Step 2: Hidden State Extraction from Qwen3-1.7B

Run frozen Qwen3-1.7B forward passes on the annotated rollouts. Extract hidden states at all layers and align boundary annotations to token positions via character offset mapping.

- **Output**: `data/probe_features/` — per-layer hidden state tensors + binary boundary labels
- **Pilot**: 200 rollouts, all layers (for layer sweep)
- **Full**: 2000 rollouts, top-3 layers (identified from pilot)

Also compute **displacement** features: $d_{t,\ell} = h_{t,\ell+1} - h_{t,\ell}$ (per TaT's finding that displacement isolates active computation from static token identity).

### Step 3: Train Boundary Probe

Sweep over configurations to find what works:

| Dimension | Options |
|-----------|---------|
| Probe architecture | Linear (`Linear(2048, 1)`) vs. MLP (`Linear(2048, 256) → ReLU → Linear(256, 1)`) |
| Input representation | Raw hidden state vs. displacement |
| Layer | All layers (pilot sweep to identify peak) |

Training: weighted BCE loss (boundaries are ~5-10% of tokens), early stopping on validation F1, split by rollout (not token) to avoid leakage.

## Integration

### Route A: Side Module at Inference (immediate)

The trained probe runs alongside generation. Each time a token is generated:
1. Read its hidden state from the target layer (already computed)
2. Feed to probe → boundary probability
3. Replace `is_boundary()` in `BoundaryTracker.step()` with probe prediction

No changes to model weights. The probe is ~500K params — a single matrix multiply per token.

### Route B: Auxiliary Loss in Training (future)

Integrate the probe's gradient into Phase 2 GRPO:
1. Probe identifies boundaries during rollout generation
2. Auxiliary loss encourages the model to develop clear internal boundary signals
3. Model internalizes boundary detection — the probe becomes redundant

## Research Questions

1. **Is semantic boundary structure linearly encoded in Qwen3-1.7B's hidden states?** (Linear probe AUC > 0.85 would confirm)
2. **Which layers encode this best?** (Expect mid-to-late, per Zhang & Afzal)
3. **Does displacement outperform raw activation?** (Test TaT's hypothesis on our model)
4. **Does the probe generalize beyond the annotation patterns?** (Qualitative error analysis)
5. **Does boundary detection transfer across reasoning domains?** (Train on math, test on logic)

## Implementation

```
scripts/annotate_boundaries.py     # Step 1: Qwen3-32B annotation
scripts/extract_hidden_states.py   # Step 2: hidden state extraction
scripts/train_boundary_probe.py    # Step 3: probe training + sweep
src/probe/__init__.py
src/probe/boundary_probe.py        # Probe models + training loop
```

## Connection to Other Components

- **`src/boundary.py`**: Current heuristic tracker. Route A replaces `is_boundary()` with probe call.
- **`src/train/reward.py`**: Progress reward probes at segment boundaries. Better boundaries → better progress signal.
- **`04_future_compression.md`**: Hidden-state compression segments the trajectory at boundaries. Learned boundaries → better compression points.
- **SFT data generation**: Probe can automate error-point detection in rollouts, replacing manual subagent work. Combined with a Zhang-style correctness probe, this fully automates `<backoff_N>` placement.
