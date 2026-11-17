"""The five-step loop with a scripted model: no GPU, no network."""

from tool_use.evaluate import score_case
from tool_use.executor import ToolLoop
from tool_use.registry import ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool(requires_confirmation=True)
def send_email(to: str) -> str:
    """Send an email.

    Args:
        to: Recipient address.
    """
    raise AssertionError("must not run without confirmation")


class ScriptedLoop(ToolLoop):
    def __init__(self, outputs):
        super().__init__(registry=ToolRegistry().register(add, send_email), model_id="fake")
        self.outputs = list(outputs)
        self.seen: list[list[dict]] = []

    def generate(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.outputs.pop(0)


def call(name, **arguments):
    import json

    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def test_answer_without_tools_ends_in_one_turn():
    loop = ScriptedLoop(["<think>Easy.</think>Hola."])
    response = loop.run("Saluda")
    assert response.answer == "Hola." and response.tool_calls == [] and response.turns == 1
    assert "add" in loop.seen[0][0]["content"]  # step 1: definitions in the system prompt


def test_parallel_then_multistep_calls_feed_results_back():
    loop = ScriptedLoop([
        call("add", a=1, b=2) + call("add", a=3, b=4),
        call("add", a=3, b=7),
        "Total: 10",
    ])
    response = loop.run("Suma", max_turns=5)
    assert [r.result for r in response.tool_calls] == [3, 7, 10]
    assert response.answer == "Total: 10" and response.turns == 3
    tool_messages = [m for m in loop.seen[1] if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["3", "7"]


def test_errors_and_confirmation_become_observations():
    loop = ScriptedLoop([
        "<tool_call>{not json}</tool_call>"
        + call("add", a="x", b=1)
        + call("send_email", to="a@b.c"),
        "No puedo.",
    ])
    response = loop.run("Haz cosas")
    assert [r.ok for r in response.tool_calls] == [False, False, False]
    observations = [m["content"] for m in loop.seen[1] if m["role"] == "tool"]
    assert "not valid JSON" in observations[0]
    assert "invalid arguments" in observations[1]
    assert "confirmation" in observations[2]


def test_out_of_turns_is_flagged():
    loop = ScriptedLoop([call("add", a=1, b=1)] * 2)
    response = loop.run("Bucle", max_turns=2)
    assert response.answer.startswith("[max_turns=2") and len(response.tool_calls) == 2


def test_boe_arguments_are_compared_by_meaning():
    case = {
        "needs_tool": True,
        "expected_tool": "consultar_boe",
        "expected_arguments": {"articulo": "13"},
    }
    same = [{
        "name": "consultar_boe",
        "arguments": {"articulo": "artículo 13", "norma": "BOE-A-2021-21007"},
    }]
    other = [{"name": "consultar_boe", "arguments": {"articulo": "art. 14"}}]
    assert score_case(case, same)["arguments_ok"]
    assert not score_case(case, other)["arguments_ok"]


def test_tools_phase_stays_pending_without_model(monkeypatch):
    import pytest

    from tool_use import executor

    monkeypatch.delenv("ARCA_TOOLS_MODEL", raising=False)
    monkeypatch.setattr(executor, "_loop", None)
    with pytest.raises(NotImplementedError, match="ARCA_TOOLS_MODEL"):
        executor.build_tool_loop()
