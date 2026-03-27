## Stabilizing SFT with Backoff Tokens on Qwen3

References:
- https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
- https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-2507
- https://qwenlm.github.io/blog/qwen3/

### Root causes of degraded SFT performance (Qwen3.5 era)

1. **Hybrid architecture complexity.** Qwen3.5 has linear attention (Mamba) layers with recurrent states — snapshot/restore on backoff caused discontinuities between full-attention and linear-attention layers.
2. **`<continue>` token pollution.** Appeared at every boundary in every example, massively distorting the training distribution. Carried no semantic meaning.
3. **LoRA alpha too high.** `alpha=32` with `r=16` gave 2x scaling, accelerating forgetting.
4. **100% backoff-format data.** Even "clean" examples used `<continue>` tokens, so the model never saw natural CoT.

### What we changed (Qwen3 migration)

1. **Switched to Qwen3-4B-Thinking-2507** — pure transformer (GQA), standard `DynamicCache`. Cache ops are simple KV tensor slices, no recurrent state management.
2. **Removed `<continue>` token.** Boundaries detected heuristically by `BoundaryTracker`. Only 3 new tokens: `<backoff_1>`, `<backoff_2>`, `<backoff_3>`.
3. **LoRA alpha = 16** (1:1 with r=16), per Unsloth recommendation.
4. **SFT data looks like natural CoT** — backoff events are the only foreign element.

### Qwen3 fine-tuning guidance (from Unsloth + Qwen)

- **75/25 reasoning split**: mix ≥75% reasoning examples with ≤25% non-reasoning/direct answers to preserve reasoning ability. For us: 75% backoff examples, 25% clean CoT (no backoff tokens at all).
- **Temperature 0.6** for thinking models (not 0.7). Use `top_p=0.95, top_k=20`.
- **`enable_thinking=True`** is the default for Thinking models — `<think>`/`</think>` are handled by the chat template.
- **`</think>` token ID is 151668** in Qwen3 (was 248069 in Qwen3.5). Our code detects it from the tokenizer, not hardcoded.
- **QLoRA (4-bit) works for Qwen3** (unlike Qwen3.5 where it was not recommended due to quantization differences). But we use bf16 for now.
- **No router layer concerns** — Qwen3-4B is dense, not MoE.

### Current SFT data format

Prompt uses Qwen3's native format: `"Please reason step by step, and put your final answer within \boxed{}."` — not the GSM8K `####` convention, which fights the model's pretrained behavior.

Clean example (25%):
```
<think>
Janet has 16 eggs per day.
She eats 3 and bakes 4, leaving 16-3-4=9 eggs.
She sells them at $2 each, making 9*2=$18.
</think>
\boxed{18}
```

Backoff example (75%):
```
<think>
Janet has 32 eggs per day.
<backoff_1> 32 is wrong, it should be 16
Janet has 16 eggs per day.
She eats 3 and bakes 4, leaving 16-3-4=9 eggs.
She sells them at $2 each, making 9*2=$18.
</think>
\boxed{18}
```

### TODO

- [ ] Re-run baselines on Qwen3-4B-Thinking-2507 (replace Qwen3.5 numbers)
- [ ] Regenerate SFT data with 75/25 backoff/clean split (`--backoff-ratio 0.75`)
- [ ] Train Phase 1 SFT: 1 epoch first, check if reasoning is preserved
- [ ] Compare: 1 epoch vs 3 epochs on eval accuracy
- [ ] Explore soft-decay attention masking as alternative to hard KV truncation
