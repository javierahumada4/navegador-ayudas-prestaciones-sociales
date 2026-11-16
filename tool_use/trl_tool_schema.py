"""Expose the *production* Pydantic tool schema to TRL's Qwen3 chat template.

GRPOTrainer requires an executable Python callable in `tools=[...]`.
Transformers normally derives that callable's schema from its type hints,
but nested Pydantic IMVCase models appear as a bare JSON object. Keep the
callable for TRL's actual tool execution and supply the canonical registry
definition to the tokenizer for prompt rendering instead.

This is instance-local; it does not monkeypatch Transformers globally.
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any

from tool_use.tools import default_registry


def canonical_imv_tool_schema() -> dict[str, Any]:
    """Return the exact `calcular_imv` definition used by ToolLoop inference."""
    registry = default_registry()
    if set(registry.tools) != {"calcular_imv"}:
        raise ValueError("Expected exactly the calcular_imv production tool")
    definition = registry.definitions()[0]
    function = definition.get("function", {})
    params = function.get("parameters", {})
    props = params.get("properties", {})
    if function.get("name") != "calcular_imv" or "case" not in props:
        raise ValueError("Invalid calcular_imv schema in production registry")
    if not params.get("$defs") and "$ref" in str(props.get("case")):
        raise ValueError("Missing Pydantic definitions for calcular_imv.case")
    return definition


def inject_imv_schema(tokenizer: Any, training_tool: Any) -> Any:
    """Render the exact production schema while TRL executes `training_tool`.

    TRL passes Python functions to tokenizer.apply_chat_template(tools=...).
    Replace only our one callable with the Pydantic-generated function schema.
    Dict-form tool definitions already present are left alone. Any unexpected
    tool list fails immediately instead of falling back to the generic schema.
    """
    if getattr(training_tool, "__name__", None) != "calcular_imv":
        raise ValueError("Training callable must be named calcular_imv")
    if getattr(tokenizer, "_imv_schema_injected", False):
        raise ValueError("IMV schema has already been injected")
    canonical = canonical_imv_tool_schema()
    original = tokenizer.apply_chat_template

    @wraps(original)
    def render(*args: Any, **kwargs: Any) -> Any:
        tools = kwargs.get("tools")
        if tools:
            if len(tools) != 1:
                raise ValueError("Tool-GRPO must expose only calcular_imv")
            tool = tools[0]
            if tool is training_tool:
                kwargs["tools"] = [deepcopy(canonical)]
            elif isinstance(tool, dict) and tool == canonical:
                kwargs["tools"] = [deepcopy(canonical)]
            else:
                raise ValueError("Unexpected tool in GRPO chat template")
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = render
    tokenizer._imv_schema_injected = True
    return tokenizer
