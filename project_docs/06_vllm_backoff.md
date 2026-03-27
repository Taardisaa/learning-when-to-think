# vLLM Integration for Backoff Generation

## Why vLLM

Qwen3 is pure transformer (GQA) — fully supported by vLLM. Our current `BackoffGenerator` does sequential token-by-token generation on HuggingFace, which is slow. vLLM gives us:

- Batched inference (multiple problems simultaneously)
- PagedAttention (efficient KV memory)
- Prefix caching (shared prompt KV reuse across rollouts)
- Native multi-rollout (`SamplingParams.n`) for GRPO

## Backoff Strategy Options

At a backoff event, the model has generated some "wrong" tokens that need to be undone. Three options for handling them, in order of implementation complexity:

### Option 1: Hard KV truncation

Delete the wrong tokens from the KV cache entirely.

- **Mechanism**: Free the paged blocks beyond the rewind point, directive tokens fill the recovered positions.
- **Pros**: Context fully reclaimed. Wrong tokens are invisible to future attention — cleanest signal.
- **Cons**: Requires modifying vLLM's `BlockManager` to free/truncate blocks mid-generation. Discontinuity at the truncation boundary.
- **vLLM approach**: Stop generation on `<backoff_N>`, collect directive, then submit a new request with `prompt = original_prefix[:rewind_pos] + directive`. Prefix caching automatically reuses the shared prefix blocks — no `BlockManager` changes needed (stop-and-restart).

### Option 2: Attention masking

Keep wrong tokens in the KV cache but mask them out (attention weight = 0).

- **Mechanism**: Set attention mask to 0 for the "wrong" positions. New tokens cannot attend to them, but they still occupy KV memory.
- **Pros**: No cache surgery. Simple to implement — just modify the attention mask. No discontinuity.
- **Cons**: Wrong tokens still consume KV memory (no context reclamation). vLLM doesn't natively support per-position attention masking during cached generation — would need to inject a custom mask into the attention kernel.
- **vLLM approach**: Would require modifying the attention backend to accept a per-sequence position mask. More invasive than stop-and-restart.

### Option 3: Soft decay

Multiply attention scores to wrong tokens by α ∈ (0, 1) instead of binary 0/1.

- **Mechanism**: Scale attention logits for wrong positions by α before softmax. Gradually reduces their influence without fully removing them.
- **Pros**: Smoothest transition — no hard discontinuity. Model still "sees" what went wrong (useful context for learning).
- **Cons**: Same memory cost as masking. Requires custom attention kernel changes. What value of α? It's an extra hyperparameter. Model wasn't pretrained with fractional attention, so behavior is unpredictable.
- **vLLM approach**: Same as masking but with a weighted mask instead of binary. Equally invasive.

### Recommendation

**Start with stop-and-restart (Option 1, no vLLM modifications):**

1. Generate with vLLM using `stop=["<backoff_1>", "<backoff_2>", "<backoff_3>"]`
2. When generation stops on a backoff token, parse the depth
3. Collect directive tokens (short secondary generation)
4. Submit a new request: `prompt[:rewind_pos] + directive`
5. vLLM prefix caching reuses the shared prefix KV blocks automatically
6. Resume generation with the same stop tokens

This gets us batched inference and prefix caching with zero vLLM modifications. The "hard truncation" happens implicitly — the new request simply doesn't include the wrong tokens.

**Later, experiment with masking/decay** as an ablation to test whether context reclamation (Option 1) actually matters vs. just reducing attention (Options 2/3). This is a key ablation mentioned in `02_method.md`.

## Implementation Sketch (Stop-and-Restart)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...", enable_prefix_caching=True)
backoff_tokens = ["<backoff_1>", "<backoff_2>", "<backoff_3>"]

params = SamplingParams(
    temperature=0.6,
    max_tokens=2048,
    stop=backoff_tokens + ["</think>"],  # stop on any action token
)

# Phase 1: generate until action
output = llm.generate([prompt], params)[0]
text = output.outputs[0].text
stop_reason = output.outputs[0].stop_reason

if stop_reason in backoff_tokens:
    depth = int(stop_reason[-1])  # 1, 2, or 3
    # Find rewind point (boundary positions tracked externally)
    rewind_pos = boundaries[-depth]
    # Collect directive (short generation from current context)
    directive = collect_directive(llm, prompt + text, max_tokens=20)
    # Resume with truncated prefix + directive
    new_prompt = prompt[:rewind_pos] + directive
    output = llm.generate([new_prompt], params)[0]

elif stop_reason == "</think>":
    # Generate answer
    ...
```

## Comparison: HF Sequential vs vLLM Stop-and-Restart

| | HF BackoffGenerator | vLLM stop-and-restart |
|---|---|---|
| Throughput | ~1 problem/sec | ~10-50 problems/sec (batched) |
| KV memory | Manual tensor slicing | PagedAttention (automatic) |
| Prefix sharing | Manual `prefill()` + `_clone_cache()` | Built-in prefix caching |
| Multi-rollout | Sequential or custom batching | Native `n=G` parameter |
| Backoff cost | Cheap (slice tensors) | Re-prefill from cache (prefix caching helps) |
| Code complexity | Custom generation loop | Mostly vLLM API calls |
