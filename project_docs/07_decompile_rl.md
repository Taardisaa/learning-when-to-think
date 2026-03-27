# RL for Semantic Correctness in LLM4Decompile

## Context

**Existing work:**
- **D-LiFT** (2506.10125): Post-processes Ghidra output. D-SCORE = compilation → symbolic exec (angr+Z3, checks return value + call count only) → readability. Trains on 300 functions. Limitation: symbolic check is partial (ignores side effects), slow, 80% applicability.
- **SK2Decompile** (2509.22114): Two-phase skeleton→skin. RL for structure + naming. Not focused on semantic correctness.
- **ACECODER** (2502.01718): For code generation, not decompilation. Auto-synthesizes test cases with GPT-4o-mini → trains reward model → RL. Key idea: use LLMs to generate verification data at scale.
- **DeepSeek-Coder-V2**: Trains a reward model on compiler+test feedback rather than raw signals. RM generalizes better than raw compiler output, especially when test coverage is weak.

**Our goal:** Apply RL to `LLM4Decompile-1.3b-v1.5` (end-to-end x64 asm → C) to improve semantic correctness. No traditional decompilers involved.

**Dataset:** `LLM4Binary/decompile-eval` (75K samples). Columns: `asm`, `func` (gold C), `func_dep` (dependencies), `test` (mostly empty except 656 HumanEval-adapted samples), `opt` (O0-O3). The 656 HumanEval samples are Python coding problems ported to C — functional but not representative of real decompilation.

## Reward Design: Differential Testing

### Why differential testing over alternatives

| Method | What it checks | Limitation |
|--------|---------------|------------|
| Symbolic exec (D-LiFT) | Return value + call count via angr+Z3 | Ignores side effects, memory mutations. Slow (~sec/sample). Only 80% applicability. |
| LLM-generated tests | Whatever the LLM thinks is important | Tests check "obvious" behavior, miss edge cases → reward-hackable. Quality varies. |
| Compilation-only | Syntactic validity | Way too weak (compiles ≠ correct). |
| HumanEval test execution | Assertion pass/fail | Only 656 samples. Python coding problems ported to C, not representative. |

**Differential testing directly measures functional equivalence: same inputs → same outputs.** With N=100+ diverse random inputs, it's hard to hack because the model can't predict which inputs will be tested. It gives a continuous reward signal (fraction matching) rather than binary.

### Reward function

```
R(decompiled_code) =
  -1.0                          if gcc compilation fails
  match_count / N  (in [0, 1])  if compiles; where N = number of random test inputs
                                 and match_count = inputs where output matches gold
```

### How it works

1. **Preprocessing (one-time):** Parse each gold C function signature → extract return type + argument types
2. **Filter:** Keep functions with "fuzzable" signatures (args are int, long, float, double, char, small fixed-size arrays). Exclude functions with opaque struct pointers, variadic args, FILE*, etc.
3. **Build test harness:** For each function, generate a C harness that:
   - Includes `func_dep` + gold function (renamed `_gold`) + decompiled function
   - Generates N random inputs matching the signature
   - Calls both functions with same inputs
   - Compares return values (and output params if applicable)
   - Prints match count
4. **Compile and run:** `gcc -O0 -lm`, timeout 5s, parse match count from stdout
5. **Reward:** `match_count / N`

### Handling limitations

- **Complex signatures:** Initially excluded. Can expand later with LLM-generated test harnesses or manual curation.
- **Side effects / global state:** Start with pure functions (no globals, no I/O). Filter based on gold source analysis.
- **Non-deterministic functions:** Exclude (detect by running gold function twice with same input, checking consistency).
- **Expected coverage:** Estimate 30-50% of 75K functions are "fuzzable." That's 22K-37K training samples — plenty for RL.

## Model & Training

- **Model:** `LLM4Binary/llm4decompile-1.3b-v1.5` (DeepSeek-Coder-1.3B, end-to-end asm→C)
- **Prompt:** `# This is the assembly code:\n{asm}\n# What is the source code?\n`
- **RL:** GRPO, LoRA (r=16, α=16), G=4-8 rollouts, PPO clipping ε=0.2, KL β=0.01
- **Curriculum:** Start with O0 (easiest optimization level), expand to O1→O2→O3

## Proposed File Structure

```
decompile_rl/
├── __init__.py
├── config.py              # Hyperparameters dataclass
├── data.py                # Load decompile-eval, parse signatures, filter fuzzable functions
├── signature_parser.py    # Parse C function signatures → (return_type, [(arg_name, arg_type)])
├── harness_generator.py   # Generate differential test harness C code for a given function pair
├── sandbox.py             # Sandboxed gcc compilation + execution (timeouts, resource limits)
├── reward.py              # Orchestrates: generate harness → compile → run → parse match fraction
├── rollout.py             # vLLM-based batched rollout generation
├── grpo.py                # GRPO training loop with LoRA
└── eval.py                # Eval on HumanEval subset (with real test assertions) + fuzzable set
scripts/
├── preprocess_dataset.py  # One-time: parse signatures, classify fuzzable, build harness templates
└── train_decompile_grpo.py
```

## Implementation Steps

### Step 1: Signature parser + dataset filter
- Parse C function declarations from `func` column using regex / pycparser
- Classify each function: fuzzable (simple numeric args) vs complex
- Output: filtered dataset with signature metadata
- Expected: 30-50% of functions are fuzzable

### Step 2: Differential test harness generator
- Template-based: takes gold function (renamed `_gold`), decompiled function, signature info
- Generates random inputs using `rand()` / type-specific generators
- Compares outputs, prints match count
- Must handle `func_dep` correctly (shared includes/types)

### Step 3: Sandbox + reward
- `sandbox.py`: Write harness to tempfile, `gcc -O0 -lm -o /tmp/test`, run with timeout
- `reward.py`: -1 for compile fail, match_fraction for success
- Validate: gold C through pipeline → should get 1.0

### Step 4: Baseline eval
- Run LLM4Decompile-1.3b-v1.5 on fuzzable subset
- Record: compile rate, mean match fraction, perfect match (1.0) rate
- Also run on HumanEval 656 samples with real tests for comparison with published numbers

### Step 5: GRPO training
- vLLM rollout generation (G=8, temp=0.8)
- Differential testing reward on each rollout
- Group-relative advantage, PPO update on LoRA weights
- Curriculum: O0 first, then mixed

### Step 6: Eval + iterate
- Compare re-executability: baseline vs RL-finetuned
- On both fuzzable set (differential) and HumanEval subset (real tests)
- Analyze: which functions improved? Which optimization levels benefit most?

## Verification

1. **Reward sanity:** Gold C source → reward should be 1.0. Random C code → should be ~0 or -1.
2. **Baseline numbers** before RL (published: ~21% re-executability for 1.3B-v1.5).
3. **Training curve:** Mean reward should increase monotonically.
4. **Held-out eval:** HumanEval subset with real test assertions (independent of training reward).
5. **Manual inspection:** Sample 20 improved functions, verify the improvements are real semantic fixes, not reward hacking.

## Risks

- **Fuzzable fraction too small:** If <20% of functions have simple signatures, may need to add LLM-generated harnesses for complex ones.
- **Reward sparsity:** If baseline model rarely compiles output, most rewards are -1. Mitigation: O0 curriculum, or add small intermediate reward for "fewer compile errors."
- **Sequence length:** Some asm inputs are very long. Filter to ≤4096 tokens.
- **Harness bugs:** Auto-generated harnesses could have bugs (wrong type casting, integer overflow in random generation). Need careful testing of harness generator.

## References

- D-LiFT: <https://arxiv.org/abs/2506.10125>
- SK2Decompile: <https://arxiv.org/abs/2509.22114>
- ACECODER: <https://arxiv.org/abs/2502.01718>
- LLM4Decompile: <https://arxiv.org/abs/2403.05286> / <https://github.com/albertan017/LLM4Decompile>
- DeepSeek-Coder-V2: <https://arxiv.org/abs/2406.11931>
- Eq@DFuzz (differential fuzzing for equivalence): <https://arxiv.org/abs/2602.15761>
- Blog post survey: <https://taardisaa.github.io/posts/decompiler-rl/>
