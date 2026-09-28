"""Checks for the GRPO training helpers that do not need a model."""

from rlm.train_grpo import zero_answer_rate


def test_zero_answer_rate_flags_only_zero_answers():
    completions = [
        "<think>x</think><answer>0.00</answer>",
        "<think>x</think><answer>0,00 €</answer>",
        "<think>x</think><answer>431.26</answer>",
        "<think>x</think>sin respuesta",
    ]
    assert zero_answer_rate([None] * 4, completions) == [1.0, 1.0, 0.0, 0.0]
