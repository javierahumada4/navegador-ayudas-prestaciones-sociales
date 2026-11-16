"""CPU checks for the cold-start protocol demonstration and train/inference parity."""

from copy import deepcopy

from tool_use.tool_prompt import demo_is_correct, tool_demonstration
from tool_use.tools import default_registry


def test_fixed_tool_demonstration_is_executable_and_independent():
    assert demo_is_correct()
    messages = tool_demonstration()
    assert [m["role"] for m in messages] == [
        "user", "assistant", "tool", "assistant",
    ]
    call = messages[1]["tool_calls"][0]["function"]
    assert call["name"] == "calcular_imv"
    record = default_registry().call(call["name"], deepcopy(call["arguments"]))
    assert record.ok, record.error
    assert record.result["total_monthly_eur"] == 431.26
    assert messages[3]["content"] == "431.26"
    messages[1]["tool_calls"][0]["function"]["arguments"]["case"]["applicant_id"] = "bad"
    assert tool_demonstration()[1]["tool_calls"][0]["function"]["arguments"]["case"]["applicant_id"] == "p1"


def test_training_and_tool_inference_use_same_example_without_real_gpu(tmp_path, monkeypatch):
    import json

    from tool_use import train_tool_grpo
    from tool_use.executor import ToolLoop

    # Fake the model generation, only inspect the resulting prompt.
    holder = {}

    class InspectLoop(ToolLoop):
        def generate(self, messages):
            holder["messages"] = deepcopy(messages)
            return "431.26"

    loop = InspectLoop(
        registry=default_registry(), brain="base",
        model_id="fake-model",
    )
    loop.run("Pregunta actual")
    assert holder["messages"][1:-1] == tool_demonstration()

    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    item = {
        "question": "Pregunta actual", "answer": "431.26",
        "rule_context": "Reglas del IMV",
        "params": tool_demonstration()[1]["tool_calls"][0]["function"]["arguments"]["case"],
    }
    train.write_text(json.dumps(item) + "\n")
    test.write_text(json.dumps({**item, "question": "Pregunta distinta"}) + "\n")
    prepared = train_tool_grpo.prepare_training_examples(train, test, 1)
    assert prepared[0]["prompt"][1:-1] == tool_demonstration()
    assert prepared[0]["prompt"][-1]["content"] == "Reglas del IMV\n\nPregunta actual"
