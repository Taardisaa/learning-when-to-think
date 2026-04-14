"""Test hard mask generation in BATCHED mode — does it still work?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pivot.action_strategies import get_strategy
from src.pivot.tokens import ACTION_TOKENS
from src.data.math import load_math500, extract_boxed_answer, grade_math_answer

MODEL = 'checkpoints/sft_pivot/merged'
NUM_PROBLEMS = 4
MAX_TOKENS = 2048

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda:1')
model.eval()

action_ids = [tok.convert_tokens_to_ids(t) for t in ACTION_TOKENS]
strategy = get_strategy('hard_mask')
processor = strategy.logits_processor(action_ids)

problems = load_math500()[:NUM_PROBLEMS]
SYSTEM = "Solve the following math problem. Please reason step by step, and put your final answer within \\boxed{}."

# Build batched prompts
prompts = []
for p in problems:
    messages = [{"role": "user", "content": f"{SYSTEM}\n\n{p['question']}"}]
    prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
print(f"Batched input shape: {enc.input_ids.shape}")

with torch.no_grad():
    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        logits_processor=[processor],
        pad_token_id=tok.eos_token_id,
    )

prompt_len = enc.input_ids.shape[1]
for i, p in enumerate(problems):
    gen_ids = out[i][prompt_len:].tolist()
    # Strip trailing padding
    while gen_ids and gen_ids[-1] == tok.eos_token_id:
        gen_ids.pop()
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)

    n_gen = len(gen_ids)
    cont = sum(1 for t in gen_ids if t == 151665)
    ref = sum(1 for t in gen_ids if t == 151666)
    term = sum(1 for t in gen_ids if t == 151667)
    pred = extract_boxed_answer(gen_text)
    correct = grade_math_answer(pred, p['answer'])

    print(f"\n--- Problem {i+1} (L{p['level']}) ---")
    print(f"  tokens: {n_gen} (hit_max: {n_gen >= MAX_TOKENS})")
    print(f"  actions: cont={cont}, ref={ref}, term={term}")
    print(f"  predicted: {pred}, gold: {p['answer']}, correct: {correct}")
    if term > 0:
        tp = gen_ids.index(151667)
        print(f"  terminate at {tp}/{n_gen}, after: {tok.decode(gen_ids[tp+1:tp+30])!r}")
