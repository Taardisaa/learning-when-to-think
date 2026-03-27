# Future Direction: Hidden-State Compression for Backoff

## Motivation

The current text-directive approach has a tension:
- **Deleting wrong tokens** reclaims context space (the core novelty) but loses information
- **Text directives** bridge the truncation boundary but have limited bandwidth (~20 tokens) and must encode the error signal in natural language
- The model generates the directive while seeing the wrong tokens, but after truncation only the directive text survives — a lossy text bottleneck

## Proposed Extension: Compressed Hidden-State Injection

Instead of encoding the error signal as text, **compress the wrong segment's hidden states into a dense representation** and inject it directly into the post-truncation KV cache.

### Architecture

```
Step 1: Generate wrong tokens (as now)
        hidden_states = last_layer_outputs[wrong_segment]  # [N, hidden_dim]

Step 2: Compress via learned module
        compressed = compressor(hidden_states)  # [K, hidden_dim], K << N
        # e.g., K=2-4 vectors summarizing an N=50 token segment

Step 3: Truncate KV cache to rewind point (as now)

Step 4: Inject compressed vectors into clean cache
        Option A: append as additional KV entries (like gist tokens)
        Option B: add to hidden states of first continuation tokens
        Option C: cross-attend from continuation tokens to compressed vectors

Step 5: Resume generation conditioned on compressed context
```

### Compressor Design Options

**Cross-attention pooler** (most flexible):
```
K learned query vectors attend over N wrong-segment hidden states
→ produces K context-aware summary vectors
```

**Linear projection** (simplest):
```
Mean-pool N hidden states → MLP → K vectors
```

**Gating mechanism** (selective):
```
Per-position importance scores → weighted sum → K vectors
Only the "informative" parts of the wrong segment survive
```

### Training

The compressor is trained end-to-end with GRPO:
- If compressed representation helps continuation reach correct answer → high reward
- If it's uninformative (continuation makes same mistake) → low reward
- Gradient flows through: continuation → injected vectors → compressor → original hidden states

### Information Density Comparison

| Method | Tokens used | Information type | Bandwidth |
|--------|------------|-----------------|-----------|
| No backoff (keep wrong tokens) | N (~50) | Full hidden states | High but wastes context |
| Text directive (current) | ~20 | Natural language | Low, interpretable |
| Compressed hidden states | K (2-4) | Dense vectors | High, compact |

The compressed approach uses ~4 token slots to encode information from ~50 wrong tokens — a ~12x compression ratio while preserving information density comparable to keeping the full segment.

### Prior Work

- **Gisting** (Mu et al., 2023) — compresses prompts into learned "gist tokens" in KV cache. Same mechanism but applied to prompts, not failed reasoning traces.
- **AutoCompressors** (Chevalier et al., 2023) — recursively compresses long contexts into summary vectors via soft prompts.
- **Landmark Attention** (Mohtashami & Jaggi, 2023) — inserts special tokens as compressed pointers to earlier context segments.
- **ICAE** (Ge et al., 2024) — in-context autoencoder that compresses context into memory slots for later retrieval.

None of these apply compression specifically to **failed reasoning** for self-correction. The novel angle: the compression target is error-informative — what went wrong and what to do instead — rather than general context summarization.

### Research Questions

1. **Can a few compressed vectors carry enough steering signal?** Text directives like "36 is wrong, it should be 35" are specific. Can 2-4 dense vectors encode equivalent information?
2. **Does the compressor learn error-specific representations?** Or does it just learn a generic summary that's no better than the text directive?
3. **How does compression ratio K affect correction quality?** Is there a sweet spot between K=1 (too lossy) and K=10 (not much compression)?
4. **Does this compose with deeper backoffs?** For depth_3 (rewinding 3 chunks), the wrong segment is longer — does compression scale?

### Key Insight: The LLM Is Already Its Own Compressor

Adding a separate compression module may be unnecessary. Autoregressive transformers already compress all prior context into the hidden state at each position. The hidden state at the `<depth>` token position already "sees" all the wrong tokens via causal attention — it IS a compressed summary of the failed reasoning.

This reframes the three levels of compression:

**Level 0 (current text directive):** The model writes text like `"36 is wrong, should be 35"`. These tokens, when re-encoded into the clean cache, produce hidden states that steer the continuation. The model is compressing its error understanding into natural language, then decompressing it through re-encoding. GRPO optimizes the text to be maximally informative for correction.

**Level 1 (hidden state injection):** Instead of going through the text bottleneck, directly inject the hidden state from the `<backoff>`/`<depth>` position into the clean cache as a phantom KV entry. No separate module — the LLM's own representations are the compressed format. The model learns to pack error-relevant information into the `<backoff>` token's hidden state because GRPO rewards continuations that use it well. Implementation:
```
1. Run wrong tokens → <backoff> → <depth>
2. Extract hidden state h at the <depth> position  (already computed)
3. Truncate KV cache
4. Insert h as an extra key-value entry in the clean cache
5. All continuation tokens attend to h via normal attention
6. Resume generation
```
No extra parameters. The special tokens act as learned "compression prompts."

**Level 2 (learned compressor):** Add a small cross-attention module that attends over ALL wrong-segment hidden states (not just the last position). Produces K summary vectors that capture different aspects of the error. Higher bandwidth than a single hidden state, but requires extra trainable parameters.

The practical progression: Level 0 (text) → validate the hypothesis → Level 1 (hidden state) → compare → Level 2 (learned compressor) if needed.

Level 0 has a key advantage: **interpretability**. We can read the directives and understand what the model thinks went wrong. Level 1 and 2 are opaque. For a research project, starting with interpretable compression (text) and measuring its limits before going dense is the right order.

### Implementation Path

1. Get text-directive backoff working end-to-end with GRPO (current approach — Level 0)
2. Measure: how often does the text directive actually lead to correction? What does the model write in directives after GRPO training?
3. If text directives hit a bandwidth bottleneck (corrections fail because the directive can't encode enough information), try Level 1 (hidden state injection)
4. Compare: text directive vs hidden state injection vs no-directive (just truncate) vs no-backoff (baseline)
5. Level 2 (learned compressor) only if Level 1 shows that single-vector compression is insufficient

This is a natural follow-up once the text-directive approach validates the core hypothesis that learned backoff improves reasoning.
