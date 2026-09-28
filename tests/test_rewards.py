from rlm.rewards import (
    accuracy_reward,
    extract_answer,
    format_reward,
    imv_accuracy_reward,
    normalize_number,
    thinking_length,
)

GOOD = "<think>2 + 2 is 4.</think><answer>4</answer>"
GOOD_SPACED = "  <think>\nStep 1.\nStep 2.\n</think>\n<answer> 4 </answer>\n"
NO_THINK = "The answer is <answer>4</answer>"
DOUBLE_ANSWER = "<think>a</think><answer>4</answer><answer>5</answer>"
BOXED = "Let me compute. \\boxed{1,234}"


def test_format_reward_accepts_strict_structure():
    assert format_reward([None, None], [GOOD, GOOD_SPACED]) == [1.0, 1.0]


def test_format_reward_rejects_missing_or_extra_blocks():
    assert format_reward([None] * 3, [NO_THINK, DOUBLE_ANSWER, "just text"]) == [0.0, 0.0, 0.0]


def test_extract_answer_prefers_answer_block_then_boxed():
    assert extract_answer(GOOD) == "4"
    assert extract_answer(BOXED) == "1,234"
    assert extract_answer(DOUBLE_ANSWER) == "5"
    assert extract_answer("nothing here") is None


def test_normalize_number_supports_english_and_spanish_formats():
    assert normalize_number("$1,234.50") == "1234.5"
    assert normalize_number("1.173,76 €") == "1173.76"
    assert normalize_number("431,26 €") == "431.26"
    assert normalize_number("The total is 18 apples") == "18"
    assert normalize_number("18.0") == "18"
    assert normalize_number("-3") == "-3"
    assert normalize_number("no digits") is None


def test_generic_accuracy_reward_stays_exact_numeric():
    completions = [
        GOOD,
        "<think>x</think><answer>$4.00</answer>",
        "<think>x</think><answer>5</answer>",
    ]
    assert accuracy_reward([None] * 3, completions, answer=["4", "4", "4"]) == [1.0, 1.0, 0.0]


def test_imv_accuracy_reward_uses_half_up_cents():
    completions = [
        "<think>x</think><answer>474.005</answer>",
        "<think>x</think><answer>474,004 €</answer>",
        "<think>x</think><answer>474.014</answer>",
    ]
    assert imv_accuracy_reward(
        [None] * 3, completions, answer=["474.01", "474.00", "474.02"]
    ) == [1.0, 1.0, 0.0]


def test_accuracy_reward_is_zero_without_answer_block():
    assert imv_accuracy_reward([None], ["<think>4</think>"], answer=["4.00"]) == [0.0]


def test_thinking_length_counts_tokens_in_think_block():
    assert thinking_length(GOOD) == 5
    assert thinking_length("no think block") == 0
