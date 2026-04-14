"""Quick test of hard mask generation — does it actually terminate with \\boxed{}?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pivot.action_strategies import get_strategy
from src.pivot.tokens import ACTION_TOKENS
from src.data.math import load_math500, extract_boxed_answer, grade_math_answer

MODEL = 'checkpoints/sft_pivot/merged'
NUM_PROBLEMS = 5
MAX_TOKENS = 2048
TEMPERATURE = 0.6

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda:1')
model.eval()

action_ids = [tok.convert_tokens_to_ids(t) for t in ACTION_TOKENS]
print(f"Action token IDs: {dict(zip(ACTION_TOKENS, action_ids))}")

strategy = get_strategy('hard_mask')
processor = strategy.logits_processor(action_ids)

problems = load_math500()[:NUM_PROBLEMS]

SYSTEM = "Solve the following math problem. Please reason step by step, and put your final answer within \\boxed{}."

for i, p in enumerate(problems):
    messages = [{"role": "user", "content": f"{SYSTEM}\n\n{p['question']}"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            logits_processor=[processor],
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][ids.shape[1]:].tolist()
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)

    n_gen = len(gen_ids)
    hit_max = n_gen >= MAX_TOKENS
    cont_count = sum(1 for t in gen_ids if t == 151665)
    ref_count = sum(1 for t in gen_ids if t == 151666)
    term_count = sum(1 for t in gen_ids if t == 151667)
    pred = extract_boxed_answer(gen_text)
    correct = grade_math_answer(pred, p['answer'])

    print(f"\n--- Problem {i+1} (L{p['level']}) ---")
    print(f"  tokens generated: {n_gen} (hit_max: {hit_max})")
    print(f"  actions: <cont>={cont_count}, <ref>={ref_count}, <term>={term_count}")
    print(f"  predicted: {pred}, gold: {p['answer']}, correct: {correct}")
    if term_count > 0:
        # Find position of first <terminate>
        term_pos = gen_ids.index(151667)
        print(f"  <terminate> at position {term_pos}/{n_gen}")
        after = tok.decode(gen_ids[term_pos+1:], skip_special_tokens=False)
        print(f"  after <terminate>: {after[:200]!r}")
