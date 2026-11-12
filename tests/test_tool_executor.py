"""CPU-only tests: no GPU, model weights, web API or real tool side effects."""

from copy import deepcopy

import pytest

from tool_use.executor import ToolLoop, build_system_prompt
from tool_use.registry import ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers; used to verify tool execution.

    Args:
        a: First integer.
        b: Second integer.
    """
    return a + b


@tool
def double(x: int) -> int:
    """Double a number.

    Args:
        x: Number to double.
    """
    return 2 * x


SIDE_EFFECTS = []


@tool(requires_confirmation=True)
def send_message(text: str) -> str:
    """Write a message with a side effect.

    Args:
        text: Message text.
    """
    SIDE_EFFECTS.append(text)
    return "sent"


class ScriptedLoop(ToolLoop):
    def __init__(self, completions: list[str], registry: ToolRegistry):
        super().__init__(registry=registry, brain="base", model_id="fake-model")
        self.completions = iter(completions)
        self.histories = []

    def generate(self, messages):
        self.histories.append(deepcopy(messages))
        return next(self.completions)


def test_regular_answer_never_calls_a_tool():
    registry = ToolRegistry().register(add)
    loop = ScriptedLoop(["El IMV es una prestación social."], registry)
    response = loop.run("¿Qué es el IMV?")
    assert response.answer.startswith("El IMV")
    assert response.tool_calls == []
    assert response.turns == 1
    assert response.model == "fake-model"


def test_parallel_calls_then_followup_call_in_later_turn():
    registry = ToolRegistry().register(add, double)
    loop = ScriptedLoop(
        [
            '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
            '<tool_call>{"name":"double","arguments":{"x":4}}</tool_call>',
            '<tool_call>{"name":"double","arguments":{"x":5}}</tool_call>',
            "Los resultados son 5, 8 y 10.",
        ],
        registry,
    )
    response = loop.run("Combina operaciones", max_turns=4)
    assert response.answer == "Los resultados son 5, 8 y 10."
    assert [x.result for x in response.tool_calls] == [5, 8, 10]
    assert response.turns == 3
    assert [m["role"] for m in loop.histories[1]] == [
        "system", "user", "assistant", "tool", "tool"
    ]
    assert all(m["name"] in {"add", "double"} for m in loop.histories[1][-2:])
    assert '"ok": true' in loop.histories[1][-2]["content"]
    assert len(loop.histories[2]) == len(loop.histories[1]) + 2


def test_invalid_arguments_are_observed_then_corrected():
    loop = ScriptedLoop(
        [
            '<tool_call>{"name":"add","arguments":{"a":"not a number","b":3}}</tool_call>',
            '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>',
            "<answer>5</answer>",
        ],
        ToolRegistry().register(add),
    )
    result = loop.run("Suma 2 y 3")
    assert not result.tool_calls[0].ok
    assert "invalid arguments" in result.tool_calls[0].error
    assert result.tool_calls[1].ok
    assert result.answer == "5"
    assert '"ok": false' in loop.histories[1][-1]["content"]


def test_unknown_tool_does_not_crash():
    loop = ScriptedLoop(
        ['<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "No dispongo de esa herramienta."],
        ToolRegistry().register(add),
    )
    result = loop.run("Pregunta")
    assert not result.tool_calls[0].ok
    assert "unknown tool" in result.tool_calls[0].error
    assert result.answer.startswith("No dispongo")


def test_missing_information_is_answered_without_calculation():
    loop = ScriptedLoop(
        ["¿Cuál es el patrimonio neto de la unidad de convivencia?"],
        ToolRegistry().register(add),
    )
    result = loop.run("Tengo 900 € de ingresos. ¿Cuánto IMV cobro?")
    assert result.tool_calls == []
    assert "patrimonio" in result.answer


def test_side_effect_tool_is_never_executed():
    SIDE_EFFECTS.clear()
    loop = ScriptedLoop(
        [
            '<tool_call>{"name":"send_message","arguments":{"text":"hola"}}</tool_call>',
            "No puedo enviar el mensaje sin confirmación.",
        ],
        ToolRegistry().register(send_message),
    )
    result = loop.run("Envía un mensaje")
    assert SIDE_EFFECTS == []
    assert not result.tool_calls[0].ok
    assert "Confirmation required" in result.tool_calls[0].error


def test_malformed_tool_call_becomes_observation():
    loop = ScriptedLoop(
        [
            '<tool_call>{"name": "add", "arguments": {oops}}</tool_call>',
            "Debo corregir los argumentos.",
        ],
        ToolRegistry().register(add),
    )
    result = loop.run("Prueba")
    assert len(result.tool_calls) == 1
    assert not result.tool_calls[0].ok
    assert "Malformed tool call" in result.tool_calls[0].error
    assert result.answer.startswith("Debo corregir")


def test_incomplete_tool_call_is_not_a_final_answer():
    loop = ScriptedLoop(
        ['<tool_call>{"name": "add"', "Necesito más información."],
        ToolRegistry().register(add),
    )
    result = loop.run("Prueba")
    assert not result.tool_calls[0].ok
    assert result.tool_calls[0].name == "tool_parser"
    assert result.answer == "Necesito más información."


def test_repeated_tool_call_is_blocked():
    call = '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
    loop = ScriptedLoop([call, call, "Terminado"], ToolRegistry().register(add))
    result = loop.run("Prueba", max_turns=3)
    assert result.tool_calls[0].ok
    assert not result.tool_calls[1].ok
    assert "Repeated identical" in result.tool_calls[1].error


def test_exhaustion_does_not_claim_success():
    loop = ScriptedLoop(
        ['<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'],
        ToolRegistry().register(add),
    )
    result = loop.run("Prueba", max_turns=1)
    assert result.turns == 1
    assert len(result.tool_calls) == 1
    assert "límite de turnos" in result.answer


def test_invalid_turn_budget():
    loop = ScriptedLoop([], ToolRegistry().register(add))
    with pytest.raises(ValueError, match="max_turns"):
        loop.run("Prueba", max_turns=0)


def test_unknown_brain_and_missing_adapter(monkeypatch):
    monkeypatch.delenv("ARCA_TOOLS_ADAPTER", raising=False)
    monkeypatch.delenv("ARCA_RLM_ADAPTER", raising=False)
    with pytest.raises(ValueError, match="ARCA_TOOLS_BRAIN"):
        ToolLoop(brain="magic")
    with pytest.raises(NotImplementedError, match="ARCA_TOOLS_ADAPTER"):
        ToolLoop(brain="rlm")


def test_base_and_rlm_share_policy_and_registry():
    registry = ToolRegistry().register(add)
    base = ToolLoop(registry=registry, brain="base", model_id="Qwen/Qwen3-1.7B")
    rlm = ToolLoop(
        registry=registry, brain="rlm", model_id="Qwen/Qwen3-1.7B",
        adapter_path="/some/weights",
    )
    assert base.registry is rlm.registry
    assert build_system_prompt(base.registry) == build_system_prompt(rlm.registry)
    assert base.model_id == rlm.model_id
    assert "some/weights" in rlm.model_label


def test_native_qwen_template_and_greedy_generation():
    torch = pytest.importorskip("torch")
    registry = ToolRegistry().register(add)
    loop = ToolLoop(registry=registry, brain="base", model_id="mock")
    seen = {}

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 2

        def apply_chat_template(self, messages, **options):
            seen["definitions"] = options["tools"]
            seen["enable_thinking"] = options["enable_thinking"]
            seen["messages"] = messages
            return "Qwen prompt"

        def __call__(self, prompt, return_tensors):
            seen["prompt"] = prompt
            assert return_tensors == "pt"
            return FakeInputs()

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens
            return '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'

    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=torch.tensor([[1, 2]]))

        def to(self, device):
            return self

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            seen["do_sample"] = kwargs["do_sample"]
            seen["max_new_tokens"] = kwargs["max_new_tokens"]
            return torch.tensor([[1, 2, 3]])

    loop.tokenizer = FakeTokenizer()
    loop.model = FakeModel()
    raw = loop.generate([{"role": "user", "content": "Suma"}])
    assert "<tool_call>" in raw
    assert seen["definitions"][0]["function"]["name"] == "add"
    assert seen["do_sample"] is False
    assert seen["enable_thinking"] is False


def test_cache_factory_does_not_load_weights(monkeypatch):
    import tool_use.executor as executor

    monkeypatch.setattr(executor, "_loop", None)
    monkeypatch.setenv("ARCA_TOOLS_BRAIN", "base")
    a = executor.build_tool_loop()
    b = executor.build_tool_loop()
    assert a is b
    assert a.model is None
    monkeypatch.setattr(executor, "_loop", None)


def test_http_tools_endpoint_returns_actual_tool_records(monkeypatch):
    """Verify the public API contract without downloading model weights."""
    from fastapi.testclient import TestClient

    from api.app import app, backends

    scripted = ScriptedLoop(
        [
            '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>',
            "El resultado es 5.",
        ],
        ToolRegistry().register(add),
    )
    monkeypatch.setattr(backends, "tools", lambda: scripted)
    response = TestClient(app).post(
        "/tools", json={"query": "Suma 2 y 3", "max_turns": 3}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answer"] == "El resultado es 5."
    assert data["turns"] == 2
    assert data["model"] == "fake-model"
    assert data["tool_calls"][0]["name"] == "add"
    assert data["tool_calls"][0]["result"] == 5
    assert data["tool_calls"][0]["ok"] is True
    assert data["tool_calls"][0]["latency_ms"] >= 0
