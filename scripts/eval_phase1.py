"""Eval: generate on GSM8K test problems.

With --no-adapter: uses standard model.generate() (no backoff tokens).
With --checkpoint:  uses BackoffGenerator (boundary + backoff tokens).

Usage:
    python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500
    python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.data.gsm8k import load_gsm8k, extract_predicted_number
from src.tokens import setup_tokenizer_and_model, TERMINATE_TOKEN
from src.generation import BackoffGenerator


def _generate_raw(model, tokenizer, prompt_ids, t_max):
    """Standard model.generate() — for base models without boundary tokens."""
    attention_mask = torch.ones_like(prompt_ids)
    with torch.no_grad():
        output = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=t_max,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output[0, prompt_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Evaluate base model with standard generate (no boundary tokens)")
    parser.add_argument("--n", type=int, default=10, help="Number of test problems")
    parser.add_argument("--t-max", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto-generated from model name)")
    args = parser.parse_args()

    use_raw = args.no_adapter or args.checkpoint is None

    if use_raw:
        print(f"Loading base model only: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
    else:
        print(f"Loading base model: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="cuda"
        )
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Loading LoRA adapter: {args.checkpoint}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model.eval()

    # Set up BackoffGenerator only for SFT'd models
    gen = None
    token_ids = None
    id2name = {}
    if not use_raw:
        from src.tokens import NEW_SPECIAL_TOKENS, BACKOFF_TOKENS
        all_special = NEW_SPECIAL_TOKENS + [TERMINATE_TOKEN]
        token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in all_special}
        print(f"Token IDs: {token_ids}")
        id2name = {v: k for k, v in token_ids.items()}

        config = BackoffConfig(
            t_max=args.t_max, temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        gen = BackoffGenerator(model, tokenizer, token_ids, config)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if use_raw:
            short_name = args.base_model.split("/")[-1]
        else:
            short_name = Path(args.checkpoint).name + "_sft"
        output_path = Path("eval_results") / f"{short_name}_n{args.n}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_gsm8k("test", subset_size=args.n)

    correct = 0
    total_backoffs = 0
    total_segments = 0
    total_terminates = 0
    results = []

    for i, ex in enumerate(data):
        messages = [{"role": "user", "content": (
            f"Solve the following math problem. "
            f"Please reason step by step, and put your final answer "
            f"within \\boxed{{}}.\n\n{ex['question']}"
        )}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("cuda")

        if use_raw:
            # Standard generation for base model
            output_text = _generate_raw(model, tokenizer, prompt_ids, args.t_max)
            predicted = extract_predicted_number(output_text)
            is_correct = predicted == ex["answer_number"]
            if is_correct:
                correct += 1

            results.append({
                "index": i,
                "question": ex["question"],
                "gold": ex["answer_number"],
                "predicted": predicted,
                "correct": is_correct,
                "output": output_text,
            })

            print(f"\n{'='*70}")
            print(f"Problem {i}: {ex['question']}...")
            print(f"Gold: {ex['answer_number']}  Predicted: {predicted}  "
                  f"{'OK' if is_correct else 'WRONG'}")
            print(f"  Output: {output_text}")
        else:
            # BackoffGenerator for SFT'd model
            traj = gen.generate(prompt_ids)

            n_backoff = traj.backoff_count
            n_terminate = 1 if traj.terminated else 0
            total_segments += len(traj.segments)
            total_backoffs += n_backoff
            total_terminates += n_terminate

            is_correct = traj.answer_number == ex["answer_number"]
            if is_correct:
                correct += 1

            # Build segment details for JSON
            seg_details = []
            for j, seg in enumerate(traj.segments):
                sd = {
                    "chunk_text": tokenizer.decode(seg.chunk_ids),
                    "action": id2name.get(seg.action, str(seg.action)) if seg.action is not None else "boundary",
                    "n_tokens": len(seg.chunk_ids),
                }
                if seg.directive_ids:
                    sd["directive"] = tokenizer.decode(seg.directive_ids)
                if seg.rewind_pos is not None:
                    sd["rewind_pos"] = seg.rewind_pos
                seg_details.append(sd)

            results.append({
                "index": i,
                "question": ex["question"],
                "gold": ex["answer_number"],
                "predicted": traj.answer_number,
                "correct": is_correct,
                "answer_text": traj.answer_text,
                "segments": seg_details,
                "backoff_count": n_backoff,
                "terminated": traj.terminated,
                "gen_tokens": traj.total_generated_tokens,
            })

            print(f"\n{'='*70}")
            print(f"Problem {i}: {ex['question']}...")
            print(f"Gold: {ex['answer_number']}  Predicted: {traj.answer_number}  "
                  f"{'OK' if is_correct else 'WRONG'}")
            print(f"Segments: {len(traj.segments)}  "
                  f"<backoff>: {n_backoff}  "
                  f"terminated: {traj.terminated}  gen_tokens: {traj.total_generated_tokens}")

            for j, seg in enumerate(traj.segments):
                if seg.action is None:
                    action_name = "boundary"
                else:
                    action_name = id2name.get(seg.action, str(seg.action))
                chunk_text = tokenizer.decode(seg.chunk_ids)
                print(f"  Seg {j}: [{action_name}] {chunk_text}")
                if seg.directive_ids:
                    dir_text = tokenizer.decode(seg.directive_ids)
                    print(f"         directive: {dir_text}")

            if traj.answer_text:
                print(f"  Answer: {traj.answer_text}")

    print(f"\n{'='*70}")
    print(f"SUMMARY ({args.n} problems)")
    print(f"{'='*70}")
    print(f"Accuracy:     {correct}/{args.n} ({100*correct/args.n:.0f}%)")
    if not use_raw:
        print(f"Segments:     {total_segments} total ({total_segments/args.n:.1f}/problem)")
        print(f"<backoff>:    {total_backoffs} total ({total_backoffs/args.n:.1f}/problem)")
        print(f"Terminated:   {total_terminates}/{args.n}")

    # Save results
    summary = {
        "model": args.base_model,
        "checkpoint": args.checkpoint,
        "mode": "raw" if use_raw else "backoff",
        "n": args.n,
        "accuracy": round(100 * correct / args.n, 2),
        "correct": correct,
    }
    if not use_raw:
        summary["total_segments"] = total_segments
        summary["total_backoffs"] = total_backoffs
        summary["terminated"] = total_terminates

    output_data = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
