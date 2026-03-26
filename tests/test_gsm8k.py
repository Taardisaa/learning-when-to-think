from src.data.gsm8k import extract_answer_number, extract_predicted_number, grade_answer


# --- extract_answer_number (gold) ---

def test_extract_gold_basic():
    assert extract_answer_number("blah blah\n#### 42") == "42"


def test_extract_gold_with_commas():
    assert extract_answer_number("#### 1,234") == "1234"


def test_extract_gold_negative():
    assert extract_answer_number("#### -5") == "-5"


def test_extract_gold_decimal():
    assert extract_answer_number("#### 3.14") == "3.14"


def test_extract_gold_none():
    assert extract_answer_number("no answer here") is None


# --- extract_predicted_number ---

def test_predicted_strict_format():
    text = "<think>let me think... 99 + 1 = 100</think>\n#### 100"
    assert extract_predicted_number(text) == "100"


def test_predicted_ignores_thinking():
    # The 999 in thinking should be ignored, answer is 42
    text = "<think>I got 999 wrong, the answer is 42</think>\n#### 42"
    assert extract_predicted_number(text) == "42"


def test_predicted_flexible_fallback():
    # No #### format, but number present after </think>
    text = "<think>reasoning</think>\nThe answer is 7"
    assert extract_predicted_number(text) == "7"


def test_predicted_no_answer():
    text = "<think>I don't know</think>\nI cannot solve this."
    assert extract_predicted_number(text) is None


def test_predicted_with_dollar():
    text = "#### $1,500"
    assert extract_predicted_number(text) == "1500"


def test_predicted_last_300_chars():
    # Long text with a misleading number early and correct answer at end
    padding = "x" * 400
    text = f"<think>The answer is 999{padding}</think>\n#### 5"
    assert extract_predicted_number(text) == "5"


# --- grade_answer ---

def test_grade_exact_match():
    assert grade_answer("42", "42")


def test_grade_float_tolerance():
    assert grade_answer("3.14", "3.14")
    assert grade_answer("3.1400001", "3.14")


def test_grade_none_inputs():
    assert not grade_answer(None, "42")
    assert not grade_answer("42", None)
    assert not grade_answer(None, None)


def test_grade_wrong_answer():
    assert not grade_answer("41", "42")


def test_grade_string_vs_float():
    # "100" == "100.0" numerically
    assert grade_answer("100", "100.0")
