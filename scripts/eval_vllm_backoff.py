"""Eval: vLLM-based backoff evaluation on GSM8K.

Uses stop-and-restart approach: vLLM generates until a backoff token,
we handle backoff externally (directive collection + prompt rebuild),
then resubmit. Prefix caching reuses shared KV blocks automatically.

No vLLM modifications needed — purely API-level.

Usage:
    # Base model (no backoff)
    python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-4B-Thinking-2507 --n 500

    # SFT'd model with backoff
    python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500
"""

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from src.boundary import BoundaryTracker
from src.data.gsm8k import load_gsm8k, extract_predicted_number
from src.prompt import build_prompt
from src.tokens import BACKOFF_TOKENS, NEW_SPECIAL_TOKENS, TERMINATE_TOKEN


@dataclass
class ProblemState:
    """Tracks state for a single problem through the backoff loop."""

    index: int
    question: str
    gold: str
    prompt_token_ids: list[int]
    gen_token_ids: list[int] = field(default_factory=list)
    boundary_token_offsets: list[int] = field(default_factory=list)
    backoff_count: int = 0
    segments: list[dict] = field(default_factory=list)
    done: bool = False
    predicted: str | None = None
    full_output: str = ""


def merge_lora(base_model: str, checkpoint: str) -> str:
    """Merge LoRA adapter into base model, save to temp dir, return path."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Merging LoRA: {base_model} + {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu"
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, checkpoint)
    model = model.merge_and_unload()

    merged_dir = tempfile.mkdtemp(prefix="merged_lora_")
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    del model, base
    torch.cuda.empty_cache()
    return merged_dir


def find_boundaries(token_ids: list[int], tokenizer, k_min: int = 15, k_max: int = 80) -> list[int]:
    """Find boundary positions (as token indices) by replaying through BoundaryTracker."""
    tracker = BoundaryTracker(k_min, k_max)
    boundaries = []
    for i, tid in enumerate(token_ids):
        token_text = tokenizer.decode([tid])
        if tracker.step(token_text):
            boundaries.append(i + 1)  # position AFTER boundary token
    return boundaries


def process_output(state: ProblemState, output, tokenizer, backoff_token_ids: set):
    """Process a vLLM generation output, updating state."""
    comp = output.outputs[0]
    new_token_ids = list(comp.token_ids)
    new_text = comp.text

    if comp.stop_reason is not None and comp.stop_reason in backoff_token_ids:
        # Backoff token emitted — don't include it in gen_token_ids
        # (it was consumed as a stop signal)
        state.gen_token_ids.extend(new_token_ids)
        state.segments.append({
            "chunk_text": new_text,
            "action": f"<backoff_{backoff_token_ids[comp.stop_reason]}>",
            "n_tokens": len(new_token_ids),
        })
        state.backoff_count += 1
        state.done = False
    else:
        # No backoff — generation completed naturally (EOS or max_tokens)
        state.gen_token_ids.extend(new_token_ids)
        state.full_output = tokenizer.decode(
            state.gen_token_ids, skip_special_tokens=True
        )
        state.predicted = extract_predicted_number(state.full_output)
        state.segments.append({
            "chunk_text": new_text,
            "action": "complete",
            "n_tokens": len(new_token_ids),
        })
        state.done = True


def main():
    parser = argparse.ArgumentParser(description="vLLM backoff evaluation on GSM8K")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA checkpoint path (omit for base model eval)")
    parser.add_argument("--n", type=int, default=500, help="Number of test problems")
    parser.add_argument("--t-max", type=int, default=8192, help="Max tokens per generation")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--k-min", type=int, default=15, help="Min tokens before boundary")
    parser.add_argument("--k-max", type=int, default=80, help="Max tokens before forced boundary")
    parser.add_argument("--k-dir", type=int, default=20, help="Max directive tokens")
    parser.add_argument("--b-max", type=int, default=2, help="Max backoffs per problem")
    args = parser.parse_args()

    # ── Load model ──
    if args.checkpoint:
        model_path = merge_lora(args.base_model, args.checkpoint)
    else:
        model_path = args.base_model

    print(f"Loading model into vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=16384,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # ── Resolve backoff token IDs ──
    # Map: token_id -> depth (1, 2, 3)
    backoff_token_id_to_depth = {}
    backoff_stop_ids = []
    for tok in BACKOFF_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            depth = int(tok[-2])  # "<backoff_1>" -> 1
            backoff_token_id_to_depth[tid] = depth
            backoff_stop_ids.append(tid)
            print(f"  {tok} -> id {tid}")

    # ── Sampling params ──
    main_params = SamplingParams(
        max_tokens=args.t_max,
        temperature=0,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=backoff_stop_ids,
    )

    directive_params = SamplingParams(
        max_tokens=args.k_dir,
        temperature=0,
        stop=["\n"],
        include_stop_str_in_output=True,
    )

    # ── Load data and build prompts ──
    data = load_gsm8k("test", subset_size=args.n)
    states: list[ProblemState] = []

    for i, ex in enumerate(data):
        prompt_text = build_prompt(tokenizer, ex["question"])
        prompt_ids = tokenizer.encode(prompt_text)

        states.append(ProblemState(
            index=i,
            question=ex["question"],
            gold=ex["answer_number"],
            prompt_token_ids=prompt_ids,
        ))

    # ── Phase 1: Batched initial generation ──
    print(f"\nPhase 1: Generating {len(states)} problems...")
    t0 = time.time()

    prompts = [{"prompt_token_ids": s.prompt_token_ids} for s in states]
    outputs = llm.generate(prompts, main_params)

    pending = []
    for state, output in zip(states, outputs):
        process_output(state, output, tokenizer, backoff_token_id_to_depth)
        if not state.done:
            pending.append(state)

    t1 = time.time()
    print(f"  Done in {t1 - t0:.1f}s. "
          f"Completed: {len(states) - len(pending)}, "
          f"Need backoff: {len(pending)}")

    # ── Phase 2: Backoff loop ──
    backoff_round = 0
    while pending and backoff_round < args.b_max:
        backoff_round += 1
        print(f"\nBackoff round {backoff_round}: {len(pending)} problems...")

        # Step A: Find boundaries in accumulated generated tokens
        for state in pending:
            state.boundary_token_offsets = find_boundaries(
                state.gen_token_ids, tokenizer, args.k_min, args.k_max
            )

        # Step B: Batched directive collection
        dir_prompts = []
        for state in pending:
            # Directive prompt = original prompt + generated tokens + backoff token
            backoff_tok_name = state.segments[-1]["action"]  # e.g. "<backoff_1>"
            backoff_tid = tokenizer.convert_tokens_to_ids(backoff_tok_name)
            dir_prompt_ids = state.prompt_token_ids + state.gen_token_ids + [backoff_tid]
            dir_prompts.append({"prompt_token_ids": dir_prompt_ids})

        dir_outputs = llm.generate(dir_prompts, directive_params)

        # Step C: Apply backoff — rewind and rebuild prompts
        rewind_prompts = []
        for state, d_out in zip(pending, dir_outputs):
            directive_text = d_out.outputs[0].text
            directive_ids = list(d_out.outputs[0].token_ids)

            # Determine rewind depth from backoff token
            backoff_tok_name = state.segments[-1]["action"]
            depth = int(backoff_tok_name[-2])  # "<backoff_1>" -> 1
            depth = min(depth, len(state.boundary_token_offsets))

            # Pop boundaries
            for _ in range(depth):
                if state.boundary_token_offsets:
                    state.boundary_token_offsets.pop()

            if state.boundary_token_offsets:
                rewind_pos = state.boundary_token_offsets[-1]
            else:
                rewind_pos = 0

            # Record directive in segment
            state.segments[-1]["directive"] = directive_text
            state.segments[-1]["rewind_token_pos"] = rewind_pos

            # Truncate gen_token_ids to rewind point + append directive
            state.gen_token_ids = state.gen_token_ids[:rewind_pos] + directive_ids

            # Build rewind prompt
            rewind_prompt_ids = state.prompt_token_ids + state.gen_token_ids
            rewind_prompts.append({"prompt_token_ids": rewind_prompt_ids})

        # Step D: Batched re-generation
        rewind_outputs = llm.generate(rewind_prompts, main_params)

        next_pending = []
        for state, output in zip(pending, rewind_outputs):
            process_output(state, output, tokenizer, backoff_token_id_to_depth)
            if not state.done and state.backoff_count < args.b_max:
                next_pending.append(state)
            elif not state.done:
                # Exceeded b_max — force done
                state.full_output = tokenizer.decode(
                    state.gen_token_ids, skip_special_tokens=True
                )
                state.predicted = extract_predicted_number(state.full_output)
                state.done = True

        pending = next_pending
        print(f"  Completed: {len(states) - len(pending) - sum(1 for s in states if not s.done)}, "
              f"Still pending: {len(pending)}")

    # Force-finish any remaining
    for state in pending:
        state.full_output = tokenizer.decode(
            state.gen_token_ids, skip_special_tokens=True
        )
        state.predicted = extract_predicted_number(state.full_output)
        state.done = True

    total_time = time.time() - t0

    # ── Score and print results ──
    correct = 0
    total_backoffs = 0
    for state in states:
        is_correct = state.predicted is not None and state.predicted == state.gold
        if is_correct:
            correct += 1
        total_backoffs += state.backoff_count

    print(f"\n{'='*70}")
    print(f"SUMMARY ({args.n} problems)")
    print(f"{'='*70}")
    print(f"Accuracy:     {correct}/{args.n} ({100*correct/args.n:.1f}%)")
    print(f"Backoffs:     {total_backoffs} total ({total_backoffs/args.n:.2f}/problem)")
    print(f"Time:         {total_time:.1f}s ({total_time/args.n:.2f}s/problem)")

    # ── Save results ──
    if args.output:
        output_path = Path(args.output)
    else:
        if args.checkpoint:
            short_name = Path(args.checkpoint).name + "_sft"
        else:
            short_name = args.base_model.split("/")[-1]
        output_path = Path("eval_results") / f"vllm_{short_name}_n{args.n}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for state in states:
        is_correct = state.predicted is not None and state.predicted == state.gold
        results.append({
            "index": state.index,
            "question": state.question,
            "gold": state.gold,
            "predicted": state.predicted,
            "correct": is_correct,
            "output": state.full_output,
            "segments": state.segments,
            "backoff_count": state.backoff_count,
        })

    summary = {
        "model": args.base_model,
        "checkpoint": args.checkpoint,
        "mode": "vllm_backoff",
        "n": args.n,
        "accuracy": round(100 * correct / args.n, 2),
        "correct": correct,
        "total_backoffs": total_backoffs,
        "inference_time_sec": round(total_time, 1),
    }

    output_data = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
