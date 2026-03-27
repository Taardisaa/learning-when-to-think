"""Prompt construction utilities.

Ensures <think> tag is always present in generation prompts,
even for models whose chat template doesn't inject it automatically
(e.g. Qwen3-1.7B vs Qwen3-4B-Thinking-2507).
"""

SYSTEM_PROMPT_MATH = (
    "Solve the following math problem. "
    r"Please reason step by step, and put your final answer within \boxed{}."
)


def build_prompt(tokenizer, question: str, system_prompt: str = SYSTEM_PROMPT_MATH) -> str:
    """Build a chat-templated prompt with <think> tag guaranteed.

    Handles models that do/don't support enable_thinking in their
    chat template. Always ensures the prompt ends with <think>\\n
    so the model enters reasoning mode.
    """
    messages = [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Ensure <think> is present
    if not prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n<think>\n"

    return prompt
