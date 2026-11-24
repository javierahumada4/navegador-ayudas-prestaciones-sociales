"""The contrast for phase 4: the same tasks with a *code* agent (smolagents ``CodeAgent``).

CodeAct (Wang et al., 2024) claims that writing actions as Python beats JSON tool calls: fewer
steps (one snippet can call several tools and combine results), and errors come back as Python
exceptions the model already knows how to read. To test that claim and nothing else, the code
agent here uses **the same model, decoding, tools and timeouts** as our JSON agent:

- ``PoolModel`` is a smolagents ``Model`` that generates with ``agent.models`` (same weights
  pool, greedy, same ``enable_thinking`` per brain); the action format is the only difference.
- Every tool of ``agent_registry()`` is wrapped as a smolagents ``Tool`` whose ``forward`` goes
  through ``call_with_timeout`` (same per-tool limits) and refuses ``guardar_informe`` unless the
  task asks for a report (same rule as the JSON agent). Failures raise, so the code agent sees an
  exception, which is how CodeAct expects errors to come back.

Needs the ``agent`` extra (``smolagents``): ``uv run --extra train --extra agent ...``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from agent.models import BrainSpec, brain_spec
from agent.react_agent import (
    TOOL_TIMEOUTS,
    Generator,
    agent_registry,
    call_with_timeout,
    report_requested,
)
from tool_use.registry import ToolRegistry

JSON_TO_SMOL = {"string": "string", "integer": "integer", "number": "number",
                "boolean": "boolean", "array": "array", "object": "object"}


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def to_chat(messages) -> list[dict]:
    """smolagents messages (roles system/user/assistant/tool-call/tool-response) -> HF chat."""
    roles = {"tool-call": "assistant", "tool-response": "user"}
    out: list[dict] = []
    for m in messages:
        data = m if isinstance(m, dict) else m.dict()
        role = getattr(data.get("role"), "value", data.get("role"))
        role = roles.get(role, role)
        text = _content_text(data.get("content"))
        if out and out[-1]["role"] == role and role != "system":
            out[-1]["content"] += "\n" + text  # chat templates expect alternating turns
        else:
            out.append({"role": role, "content": text})
    return out


def make_model(spec: BrainSpec, generator: Generator | None = None):
    from smolagents.models import ChatMessage, MessageRole, Model, TokenUsage

    class PoolModel(Model):
        def __init__(self) -> None:
            super().__init__(model_id=spec.label)

        def generate(self, messages, stop_sequences=None, response_format=None,
                     tools_to_call_from=None, **kwargs):
            chat = to_chat(messages)
            if generator is not None:
                text, prompt_tokens, completion_tokens = generator(chat, spec, [])
            else:
                from agent.models import generate

                text, prompt_tokens, completion_tokens = generate(spec, chat, None)
            for stop in stop_sequences or []:  # HF generate has no stop strings here: cut after
                if stop and stop in text:
                    text = text[: text.index(stop)]
            return ChatMessage(
                role=MessageRole.ASSISTANT, content=text,
                token_usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
            )

    return PoolModel()


def make_tools(registry: ToolRegistry, task: str, used: list[str],
               timeouts: dict[str, float] | None = None) -> list:
    from smolagents import Tool

    timeouts = timeouts or TOOL_TIMEOUTS
    tools = []
    for name, spec in registry.tools.items():
        if name == "final_answer":  # smolagents brings its own final_answer
            continue
        params = spec.definition["function"]["parameters"]
        required = set(params.get("required", []))
        inputs = {}
        for arg, schema in params.get("properties", {}).items():
            entry = {
                "type": JSON_TO_SMOL.get(schema.get("type"), "object"),
                "description": schema.get("description") or arg,
            }
            if arg not in required:
                entry["nullable"] = True
            inputs[arg] = entry

        def forward(self, _name=name, **kwargs):
            used.append(_name)
            arguments = {k: v for k, v in kwargs.items() if v is not None}
            if registry.tools[_name].requires_confirmation and not report_requested(task):
                raise PermissionError(f"{_name} tiene efectos y el usuario no ha pedido guardar un informe")
            record = call_with_timeout(registry, _name, arguments, timeouts.get(_name, 30))
            if record is None:
                raise TimeoutError(f"la herramienta {_name} no respondió a tiempo")
            if not record.ok:
                raise RuntimeError(record.error)
            return record.result

        cls = type(f"{name}_tool", (Tool,), {
            "name": name, "description": spec.description, "inputs": inputs,
            "output_type": "any", "skip_forward_signature_validation": True, "forward": forward,
        })
        tools.append(cls())
    return tools


def run_code_agent(task: str, max_steps: int = 10, brain: str = "base",
                   generator: Generator | None = None,
                   registry_factory: Callable[[], ToolRegistry] = agent_registry) -> dict:
    """Run one task with smolagents' CodeAgent; same keys as ``AgentResponse`` plus tools_used."""
    from smolagents import CodeAgent
    from smolagents.memory import ActionStep

    spec = brain_spec(brain)
    used: list[str] = []
    agent = CodeAgent(
        tools=make_tools(registry_factory(), task, used),
        model=make_model(spec, generator),
        max_steps=max_steps,
        additional_authorized_imports=["json", "math"],
        verbosity_level=0,
    )
    started = time.perf_counter()
    try:
        result = agent.run(task, return_full_result=True)
        output, state = result.output, result.state
        tokens = result.token_usage.total_tokens if result.token_usage else 0
    except Exception as exc:  # noqa: BLE001 - a crashed run is a failed task, not a crashed bench
        output, state, tokens = f"{type(exc).__name__}: {exc}", "error", 0
    actions = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    steps = [
        {
            "thought": (s.model_output or "").strip(),
            "action": {"name": "python", "arguments": {"code": s.code_action or ""}},
            "observation": str(s.observations or s.error or ""),
        }
        for s in actions
    ]
    return {
        "final_answer": "" if output is None else str(output),
        "steps": steps,
        "n_steps": len(actions),
        "succeeded": state == "success",
        "model": f"code:{spec.name}:{spec.label}",
        "tokens_used": tokens,
        "tools_used": sorted(set(used)),
        "wall_s": round(time.perf_counter() - started, 1),
    }
