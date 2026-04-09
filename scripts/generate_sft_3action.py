"""Build SFT warmup dataset by using Qwen3-32B to rewrite rollouts with action tokens.

Instead of regex-based splitting, prompts a stronger model to intelligently place
<continue>, <refine>, and <terminate> tokens at semantic boundaries.

Usage:
    # Full run
    python -m scripts.generate_sft_3action \
        --rollouts data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl \
        --rewriter Qwen/Qwen3-32B

    # Dry run on small subset
    python -m scripts.generate_sft_3action \
        --rollouts data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl \
        --rewriter Qwen/Qwen3-32B --limit 10

    # Use pre-generated rewrite prompts (skip vLLM)
    python -m scripts.generate_sft_3action --from-rewrites data/rewrites_raw.jsonl
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from src.data.math import extract_boxed_answer, grade_math_answer


# ── Rewrite prompt ──

REWRITE_SYSTEM = """\
You are a math reasoning annotator. Your job is to rewrite a model's reasoning \
trajectory by inserting special action tokens at the right places.

## Action Tokens

- `<continue>` — placed after each intermediate reasoning step, at a semantic \
boundary where the model could decide "should I keep going?" Place it between \
distinct reasoning steps (e.g., between setting up an equation and solving it, \
between computing a sub-result and applying it, between considering different cases).

- `<refine>` — placed at the point where reasoning goes wrong. Followed by a \
corrective directive that names the specific error (without revealing the answer), \
then correct reasoning continues. Use ONLY when the trajectory contains an error.

- `<terminate>` — placed exactly once, after the final reasoning step, immediately \
before \\boxed{answer}. Signals "stop reasoning, output the answer."

## Rules

1. Place `<continue>` at semantic boundaries — NOT mid-sentence, NOT mid-formula, \
NOT between lines of the same algebraic simplification.
2. Every output must have exactly one `<terminate>` followed by `\\boxed{answer}`.
3. For CLEAN rewrites (correct trajectories): only use `<continue>` and `<terminate>`. No `<refine>`.
4. For REFINE rewrites (wrong trajectories): identify where the error happens, \
place `<refine>` there with a directive that describes the specific mistake, then \
write correct reasoning to the gold answer. The directive must NOT reveal the gold answer.
5. Keep the original reasoning style and verbosity. Do not shorten or paraphrase — \
just insert tokens at the right boundaries.
6. The final `\\boxed{}` must contain the correct answer.
7. Do NOT wrap output in ```code blocks```, <think> tags, or any other markup. \
Output the rewritten trajectory directly."""

CLEAN_FEWSHOT = """\
## Example: Clean rewrite

Question: What is the slope of the line passing through (-3,5) and (2,-5)?
Gold answer: -2
Trajectory (correct):
To find the slope of the line passing through the points (-3,5) and (2,-5), we use the slope formula:
m = (y2 - y1) / (x2 - x1)
where (x1, y1) and (x2, y2) are the coordinates of the two points. Here, (x1, y1) = (-3, 5) and (x2, y2) = (2, -5).
First, we calculate the difference in the y-coordinates:
y2 - y1 = -5 - 5 = -10
Next, we calculate the difference in the x-coordinates:
x2 - x1 = 2 - (-3) = 2 + 3 = 5
Now, we substitute these values into the slope formula:
m = -10 / 5 = -2
Therefore, the slope of the line is -2.

Rewrite:
To find the slope of the line passing through the points (-3,5) and (2,-5), we use the slope formula:
m = (y2 - y1) / (x2 - x1)
where (x1, y1) and (x2, y2) are the coordinates of the two points. Here, (x1, y1) = (-3, 5) and (x2, y2) = (2, -5).
<continue>
First, we calculate the difference in the y-coordinates:
y2 - y1 = -5 - 5 = -10
<continue>
Next, we calculate the difference in the x-coordinates:
x2 - x1 = 2 - (-3) = 2 + 3 = 5
<continue>
Now, we substitute these values into the slope formula:
m = -10 / 5 = -2
<continue>
Therefore, the slope of the line is -2.
<terminate>
\\boxed{-2}"""

REFINE_FEWSHOT = """\
## Example: Refine rewrite

Question: A cylinder has a radius of 3 cm and a height of 8 cm. What is the longest segment, in centimeters, that would fit inside the cylinder?
Gold answer: 10
Trajectory (WRONG — predicted \\sqrt{82}):
To find the longest segment that would fit inside a cylinder, we need to determine the length of the cylinder's space diagonal. The formula for the space diagonal d of a cylinder is:
d = sqrt(2r^2 + h^2)
where r is the radius and h is the height. Given r = 3 cm and h = 8 cm:
d = sqrt(2(9) + 64) = sqrt(18 + 64) = sqrt(82)
Therefore, the longest segment is sqrt(82) cm.

Rewrite:
To find the longest segment that would fit inside a cylinder, we need to determine the length of the cylinder's space diagonal.
<continue>
The formula for the space diagonal d of a cylinder is:
d = sqrt(2r^2 + h^2)
where r is the radius and h is the height.
<refine> Wait, I used the wrong formula. The longest segment inside a cylinder is the diagonal from one edge of the base to the opposite edge of the top, so the base of the right triangle is the diameter (2r), not sqrt(2)*r. Let me redo this with the correct formula d = sqrt((2r)^2 + h^2).
<continue>
The correct formula uses the diameter, not r*sqrt(2):
d = sqrt((2r)^2 + h^2)
Given r = 3 cm, so diameter = 6 cm, and h = 8 cm:
d = sqrt(6^2 + 8^2) = sqrt(36 + 64) = sqrt(100) = 10
<terminate>
\\boxed{10}"""


def build_clean_prompt(question: str, trajectory: str, gold_answer: str) -> str:
    return f"""{REWRITE_SYSTEM}

{CLEAN_FEWSHOT}

---

Now rewrite the following trajectory. This trajectory is CORRECT, so use only <continue> and <terminate> (no <refine>).

Question: {question}
Gold answer: {gold_answer}
Trajectory (correct):
{trajectory}

Rewrite:"""


def build_refine_prompt(
    question: str, wrong_trajectory: str, wrong_predicted: str | None, gold_answer: str
) -> str:
    pred_note = f" — predicted {wrong_predicted}" if wrong_predicted else ""
    return f"""{REWRITE_SYSTEM}

{REFINE_FEWSHOT}

---

Now rewrite the following trajectory. This trajectory is WRONG{pred_note}, so you must use <refine> at the error point, write a corrective directive and correct reasoning, then <terminate> before \\boxed{{{gold_answer}}}.

Question: {question}
Gold answer: {gold_answer}
Trajectory (WRONG{pred_note}):
{wrong_trajectory}

Rewrite:"""


# ── Validation ──


def validate_rewrite(text: str, gold_answer: str, expect_refine: bool) -> str | None:
    """Validate a rewritten trajectory. Returns error string or None if valid."""
    if text.count("<terminate>") != 1:
        return f"expected 1 <terminate>, got {text.count('<terminate>')}"
    if "<continue>" not in text:
        return "no <continue> found"
    if "\\boxed{" not in text and "\\boxed " not in text:
        return "no \\boxed{} found"

    # Check <terminate> before \boxed
    t_pos = text.find("<terminate>")
    b_pos = text.find("\\boxed{")
    if b_pos >= 0 and t_pos > b_pos:
        return "<terminate> after \\boxed{}"

    # Check answer correctness
    predicted = extract_boxed_answer(text)
    if not grade_math_answer(predicted, gold_answer):
        return f"wrong answer: predicted={predicted}, gold={gold_answer}"

    # Check refine token
    if expect_refine and "<refine>" not in text:
        return "missing <refine> in refine example"
    if not expect_refine and "<refine>" in text:
        return "unexpected <refine> in clean example"

    return None


# ── Main pipeline ──


def prepare_rewrite_inputs(
    grouped_rollouts: list[dict],
    clean_ratio: float = 0.6,
    seed: int = 42,
) -> list[dict]:
    """Prepare (prompt, metadata) pairs for LLM rewriting."""
    rng = random.Random(seed)
    inputs = []

    trivial = [g for g in grouped_rollouts if g["pass_rate"] == 1.0]
    mixed = [g for g in grouped_rollouts if 0 < g["pass_rate"] < 1.0]

    # Clean examples from trivial problems
    for group in trivial:
        r = rng.choice(group["rollouts"])
        inputs.append({
            "prompt": build_clean_prompt(group["question"], r["text"], group["answer"]),
            "question": group["question"],
            "answer": group["answer"],
            "expect_refine": False,
        })

    # Mixed problems: clean or refine
    for group in mixed:
        correct = [r for r in group["rollouts"] if r["correct"]]
        wrong = [r for r in group["rollouts"] if not r["correct"]]

        if rng.random() < clean_ratio:
            r = rng.choice(correct)
            inputs.append({
                "prompt": build_clean_prompt(group["question"], r["text"], group["answer"]),
                "question": group["question"],
                "answer": group["answer"],
                "expect_refine": False,
            })
        else:
            w = rng.choice(wrong)
            inputs.append({
                "prompt": build_refine_prompt(
                    group["question"], w["text"], w["predicted"], group["answer"]
                ),
                "question": group["question"],
                "answer": group["answer"],
                "expect_refine": True,
            })

    # Extra refine examples to hit ~40% target
    n_clean = sum(1 for i in inputs if not i["expect_refine"])
    n_refine = sum(1 for i in inputs if i["expect_refine"])
    target_refine = int(n_clean * 0.4 / 0.6) - n_refine

    if target_refine > 0 and mixed:
        rng.shuffle(mixed)
        for group in mixed * 3:
            if target_refine <= 0:
                break
            correct = [r for r in group["rollouts"] if r["correct"]]
            wrong = [r for r in group["rollouts"] if not r["correct"]]
            w = rng.choice(wrong)
            inputs.append({
                "prompt": build_refine_prompt(
                    group["question"], w["text"], w["predicted"], group["answer"]
                ),
                "question": group["question"],
                "answer": group["answer"],
                "expect_refine": True,
            })
            target_refine -= 1

    rng.shuffle(inputs)
    return inputs


MATH_SYSTEM_PROMPT = (
    "Solve the following math problem. "
    r"Please reason step by step, and put your final answer within \boxed{}."
)


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT warmup dataset via LLM rewriting with Qwen3-32B"
    )
    parser.add_argument("--rollouts", default=None,
                        help="Grouped rollouts JSONL")
    parser.add_argument("--rewriter", default="Qwen/Qwen3-32B",
                        help="Model for rewriting (default: Qwen3-32B)")
    parser.add_argument("--output", default="data/sft_3action_math_train.jsonl")
    parser.add_argument("--clean-ratio", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs (e.g. '1,2'). Default: auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load rollouts
    print(f"Loading rollouts from {args.rollouts}...")
    with open(args.rollouts) as f:
        grouped = [json.loads(line) for line in f]
    if args.limit:
        grouped = grouped[:args.limit]
    print(f"  {len(grouped)} problems")

    # Prepare rewrite inputs
    inputs = prepare_rewrite_inputs(grouped, clean_ratio=args.clean_ratio, seed=args.seed)
    n_clean = sum(1 for i in inputs if not i["expect_refine"])
    n_refine = sum(1 for i in inputs if i["expect_refine"])
    print(f"  Prepared {len(inputs)} rewrite prompts (clean={n_clean}, refine={n_refine})")

    # Set visible GPUs before loading model
    if args.gpus:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")

    # Load rewriter model
    print(f"\nLoading rewriter: {args.rewriter}...")
    llm = LLM(
        model=args.rewriter,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
    )

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.3,  # low temp for faithful rewriting
        top_p=0.95,
    )

    # Generate rewrites
    prompts = [i["prompt"] for i in inputs]
    print(f"Generating {len(prompts)} rewrites...")
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(prompts):.2f}s/prompt)")

    # Free GPU
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # Validate and build examples
    examples = []
    stats = {"clean": 0, "refine": 0, "rejected": 0}
    rejection_reasons = {}

    for inp, output in zip(inputs, outputs):
        text = output.outputs[0].text.strip()

        # Strip <think>...</think> wrapper if the rewriter added one
        think_end = text.find("</think>")
        if think_end >= 0:
            text = text[think_end + len("</think>"):].strip()
        # Also strip leading <think>
        if text.startswith("<think>"):
            text = text[len("<think>"):].strip()

        error = validate_rewrite(text, inp["answer"], inp["expect_refine"])
        if error:
            stats["rejected"] += 1
            rejection_reasons[error] = rejection_reasons.get(error, 0) + 1
            continue

        has_refine = "<refine>" in text
        if has_refine:
            stats["refine"] += 1
        else:
            stats["clean"] += 1

        examples.append({
            "question": inp["question"],
            "answer": inp["answer"],
            "messages": [
                {"role": "user", "content": f"{MATH_SYSTEM_PROMPT}\n\n{inp['question']}"},
                {"role": "assistant", "content": text},
            ],
            "has_refine": has_refine,
        })

    total = len(examples)
    print(f"\nDataset built: {total} examples")
    print(f"  Clean:    {stats['clean']:5d} ({100*stats['clean']/max(total,1):.1f}%)")
    print(f"  Refine:   {stats['refine']:5d} ({100*stats['refine']/max(total,1):.1f}%)")
    print(f"  Rejected: {stats['rejected']:5d}")
    if rejection_reasons:
        print(f"  Rejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Token length stats
    if examples:
        lengths = [len(ex["messages"][1]["content"]) // 4 for ex in examples]
        arr = np.array(lengths)
        print(f"\n  Token estimate: mean={arr.mean():.0f}, median={np.median(arr):.0f}, "
              f"p95={np.percentile(arr, 95):.0f}, max={arr.max()}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nSaved to {args.output}")

    # Show samples
    for label, filter_fn in [("CLEAN", lambda e: not e["has_refine"]),
                              ("REFINE", lambda e: e["has_refine"])]:
        print(f"\n{'='*60}")
        print(f"SAMPLE {label} EXAMPLE")
        print(f"{'='*60}")
        matches = [e for e in examples if filter_fn(e)]
        if matches:
            content = matches[0]["messages"][1]["content"]
            print(content[:800])
            if len(content) > 800:
                print(f"  ... ({len(content)} chars total)")


if __name__ == "__main__":
    main()
