"""Checks for splitting the model output into thinking and answer (no model needed)."""

from rlm.inference import split_thinking


def test_closed_think_and_answer():
    raw = "<think>\nsumo rentas\n</think>\n<answer>441.29</answer>"
    assert split_thinking(raw) == ("sumo rentas", "441.29")


def test_truncated_think_never_leaks_into_answer():
    thinking, answer = split_thinking("<think>\nOkay, let's tackle this problem step by")
    assert thinking == "Okay, let's tackle this problem step by"
    assert answer == ""


def test_closed_think_without_answer_tags_keeps_the_tail():
    assert split_thinking("<think>razono</think>\n441.29") == ("razono", "441.29")


def test_no_tags_at_all_returns_raw():
    assert split_thinking("441.29") == ("", "441.29")


def test_inference_prompts_receive_same_imv_rule_sheet():
    from rlm.data import get_imv_rule_context
    from rlm.inference import ReasoningModel

    model = ReasoningModel(base_model="unused")
    result = model._prompt("Caso IMV")
    assert result[1]["content"] == get_imv_rule_context() + "\n\nCaso IMV"
    custom = model._prompt("Caso IMV", "REGLAS ESPECIFICAS")
    assert custom[1]["content"] == "REGLAS ESPECIFICAS\n\nCaso IMV"


def test_non_imv_inference_does_not_inject_imv_rules():
    from rlm.inference import ReasoningModel

    model = ReasoningModel(base_model="unused", verifier_name="numeric")
    assert model._prompt("Otro dominio")[1]["content"] == "Otro dominio"
