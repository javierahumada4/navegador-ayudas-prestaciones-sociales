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
