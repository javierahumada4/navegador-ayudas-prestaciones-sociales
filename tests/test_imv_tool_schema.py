"""CPU tests: GRPO rendering must expose the exact earlier IMV tool schema."""

from copy import deepcopy

import pytest

from tool_use.tools import default_registry
from tool_use.train_tool_grpo import calcular_imv
from tool_use.trl_tool_schema import canonical_imv_tool_schema, inject_imv_schema


class FakeTokenizer:
    def __init__(self):
        self.last_tools = None

    def apply_chat_template(self, *args, **kwargs):
        self.last_tools = kwargs.get("tools")
        return "template"


def test_training_schema_equals_original_registered_pydantic_schema():
    expected = default_registry().definitions()[0]
    schema = canonical_imv_tool_schema()
    assert schema == expected
    assert schema["function"]["name"] == "calcular_imv"
    parameters = schema["function"]["parameters"]
    assert parameters["required"] == ["case"]
    assert "$defs" in parameters
    assert "case" in parameters["properties"]
    assert "$ref" in parameters["properties"]["case"]
    assert any("Person" in key for key in parameters["$defs"])
    assert any("EconomicFacts" in key for key in parameters["$defs"])


def test_train_render_exposes_original_schema_and_runs_actual_callable():
    fake = FakeTokenizer()
    install = inject_imv_schema(fake, calcular_imv)
    assert install is fake
    assert fake.apply_chat_template([], tools=[calcular_imv]) == "template"
    assert fake.last_tools == default_registry().definitions()
    assert fake.last_tools[0]["function"]["parameters"]["$defs"]

    # Leave schema dictionaries alone; never silently change foreign tools.
    original = deepcopy(default_registry().definitions()[0])
    fake.apply_chat_template([], tools=[original])
    assert fake.last_tools == [original]


def test_only_the_calculator_tool_is_accepted():
    fake = FakeTokenizer()
    inject_imv_schema(fake, calcular_imv)

    def another_tool(case: dict) -> dict:
        return case

    with pytest.raises(ValueError, match="Unexpected tool"):
        fake.apply_chat_template([], tools=[another_tool])
    with pytest.raises(ValueError, match="only calcular_imv"):
        fake.apply_chat_template([], tools=[calcular_imv, another_tool])
    with pytest.raises(ValueError, match="already been injected"):
        inject_imv_schema(fake, calcular_imv)


def test_no_tool_rendering_is_preserved():
    fake = FakeTokenizer()
    inject_imv_schema(fake, calcular_imv)
    fake.apply_chat_template([])
    assert fake.last_tools is None
