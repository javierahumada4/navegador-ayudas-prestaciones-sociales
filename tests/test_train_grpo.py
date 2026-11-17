"""Checks for the GRPO training helpers that do not need a model."""

from rlm.train_grpo import eligibility_reward, zero_answer_rate


def test_zero_answer_rate_flags_only_zero_answers():
    completions = [
        "<think>x</think><answer>0.00</answer>",
        "<think>x</think><answer>0,00 €</answer>",
        "<think>x</think><answer>431.26</answer>",
        "<think>x</think>sin respuesta",
    ]
    assert zero_answer_rate([None] * 4, completions) == [1.0, 1.0, 0.0, 0.0]


def test_eligibility_reward_scores_the_zero_decision_not_the_amount():
    completions = [
        "<think>x</think><answer>0.00</answer>",  # ineligible, said 0
        "<think>x</think><answer>115.00</answer>",  # ineligible, gave CAPI anyway
        "<think>x</think><answer>500.00</answer>",  # eligible, wrong amount
        "<think>x</think><answer>0.00</answer>",  # eligible, said 0
        "<think>x</think>sin respuesta",
    ]
    answers = ["0.00", "0.00", "733.60", "733.60", "0.00"]
    assert eligibility_reward([None] * 5, completions, answers) == [1.0, 0.0, 1.0, 0.0, 0.0]
