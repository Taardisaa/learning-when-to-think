"""Quick eval: load Phase 1 checkpoint and generate on a few GSM8K test problems.

Check whether the model produces <continue>, <backoff>, </think> tokens.

Usage:
    python -m scripts.eval_phase1
    python -m scripts.eval_phase1 --checkpoint checkpoints/phase1/final --n 10
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BackoffConfig
from src.data.gsm8k import load_gsm8k
from src.tokens import setup_tokenizer_and_model, TERMINATE_TOKEN
from src.generation import BackoffGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/phase1/test")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n", type=int, default=10, help="Number of test problems")
    parser.add_argument("--t-max", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda"
    )

    # The checkpoint saved the tokenizer with special tokens already added,
    # but we still need to resize the base model's embeddings to match
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    # Build token_ids map from the already-configured tokenizer
    from src.tokens import NEW_SPECIAL_TOKENS, ACTION_TOKENS, DEPTH_TOKENS
    all_special = NEW_SPECIAL_TOKENS + [TERMINATE_TOKEN]
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in all_special}
    print(f"Token IDs: {token_ids}")

    config = BackoffConfig(t_max=args.t_max, temperature=args.temperature)
    gen = BackoffGenerator(model, tokenizer, token_ids, config)

    data = load_gsm8k("test", subset_size=args.n)
    id2name = {v: k for k, v in token_ids.items()}

    correct = 0
    total_backoffs = 0
    total_continues = 0
    total_terminates = 0

    for i, ex in enumerate(data):
        messages = [{"role": "user", "content": (
            f"Solve the following math problem. "
            f"Give the final answer after ####.\n\n{ex['question']}"
        )}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("cuda")

        traj = gen.generate(prompt_ids)

        # Count tokens
        n_continue = sum(1 for s in traj.segments if s.action == token_ids["<continue>"])
        n_backoff = traj.backoff_count
        n_terminate = 1 if traj.terminated else 0
        total_continues += n_continue
        total_backoffs += n_backoff
        total_terminates += n_terminate

        is_correct = traj.answer_number == ex["answer_number"]
        if is_correct:
            correct += 1

        print(f"\n{'='*70}")
        print(f"Problem {i}: {ex['question'][:80]}...")
        print(f"Gold: {ex['answer_number']}  Predicted: {traj.answer_number}  {'OK' if is_correct else 'WRONG'}")
        print(f"Segments: {len(traj.segments)}  "
              f"<continue>: {n_continue}  <backoff>: {n_backoff}  "
              f"terminated: {traj.terminated}  gen_tokens: {traj.total_generated_tokens}")

        for j, seg in enumerate(traj.segments[:8]):
            action_name = id2name.get(seg.action, str(seg.action))
            chunk_text = tokenizer.decode(seg.chunk_ids)[:100]
            print(f"  Seg {j}: [{action_name}] {chunk_text}")
            if seg.directive_ids:
                dir_text = tokenizer.decode(seg.directive_ids)[:80]
                print(f"         directive: {dir_text}")

        if traj.answer_text:
            print(f"  Answer: {traj.answer_text[:150]}")

    print(f"\n{'='*70}")
    print(f"SUMMARY ({args.n} problems)")
    print(f"{'='*70}")
    print(f"Accuracy:     {correct}/{args.n} ({100*correct/args.n:.0f}%)")
    print(f"<continue>:   {total_continues} total ({total_continues/args.n:.1f}/problem)")
    print(f"<backoff>:    {total_backoffs} total ({total_backoffs/args.n:.1f}/problem)")
    print(f"Terminated:   {total_terminates}/{args.n}")


if __name__ == "__main__":
    main()
