"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn (parallel calls).

``run_tool_loop`` is what ``POST /tools`` calls. Environment: ``ARCA_TOOLS_MODEL``
(required to enable the endpoint, e.g. ``Qwen/Qwen3-1.7B``),
``ARCA_TOOLS_MAX_NEW_TOKENS`` (1024) and ``ARCA_TOOLS_THINKING`` (``1`` lets Qwen3 think
before each call; off by default).
"""

from __future__ import annotations

import os

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant designed to help users.
You will be given a task to solve. To help you, you have access to a set of tools.
Each tool is a function and has a description explaining what it does and the inputs it
expects:
---
{tools_description}
---
You should decide if you want to call a tool. If so, your call must be formatted as follows:
<tool_call>
{{"name": "tool name", "arguments": {{"argument": "value"}}}}
</tool_call>
You may emit several <tool_call> blocks in one message if the calls are independent.
After the tool calls are executed, you will receive their results and can continue.
When you have everything you need, answer the user directly without calling tools."""


def build_system_prompt(registry: ToolRegistry) -> str:
    """Step 1: the system prompt with the tool definitions."""
    return SYSTEM_PROMPT_TEMPLATE.format(tools_description=registry.render_for_prompt())


class ToolLoop:
    """Holds the model and the registry and runs the five-step workflow."""

    def __init__(self, registry: ToolRegistry | None = None, model_id: str | None = None):
        self.registry = registry or default_registry()
        self.model_id = model_id or os.environ.get("ARCA_TOOLS_MODEL", "Qwen/Qwen3-1.7B")
        self.max_new_tokens = int(os.environ.get("ARCA_TOOLS_MAX_NEW_TOKENS", "1024"))
        # Qwen3 thinking before each call is slower and rarely changes the tool chosen.
        self.enable_thinking = os.environ.get("ARCA_TOOLS_THINKING", "0") == "1"
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load tokenizer and model (bf16 on GPU)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, device_map=device
        ).eval()

    def generate(self, messages: list[dict]) -> str:
        """Step 2: one greedy generation from the conversation so far.

        The tool definitions travel in the system prompt built by ``build_system_prompt``;
        ``apply_chat_template`` only wraps the turns (including role ``tool``) in Qwen's format.
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _execute(self, call: ToolCall) -> ToolCallRecord:
        """Step 3. Parse errors and calls needing confirmation become observations."""
        if call.error is not None:
            return ToolCallRecord(
                name=call.name, arguments=call.arguments, ok=False, error=call.error
            )
        tool = self.registry.tools.get(call.name)
        if tool is not None and tool.requires_confirmation:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error="this tool needs user confirmation and /tools cannot ask for it; "
                "it was not executed. Tell the user what you would do instead.",
            )
        return self.registry.call(call.name, call.arguments)

    def run(self, query: str, max_turns: int = 5) -> ToolsResponse:
        """Steps 1-5, repeating 2-4 while the model keeps calling tools."""
        messages = [
            {"role": "system", "content": build_system_prompt(self.registry)},
            {"role": "user", "content": query},
        ]
        records: list[ToolCallRecord] = []
        parsed = None
        for turn in range(max_turns):
            raw = self.generate(messages)
            parsed = parse_tool_calls(raw)
            if not parsed.tool_calls:
                return ToolsResponse(
                    answer=parsed.text, tool_calls=records, turns=turn + 1, model=self.model_id
                )
            messages.append({"role": "assistant", "content": raw})
            for call in parsed.tool_calls:  # parallel calls run in the same turn
                record = self._execute(call)
                records.append(record)
                payload = record.result if record.ok else {"error": record.error}
                content = format_tool_result(record.name, payload)
                messages.append({"role": "tool", "name": record.name, "content": content})
        answer = parsed.text if parsed and parsed.text else ""
        return ToolsResponse(
            answer=f"[max_turns={max_turns} reached without a final answer] {answer}".strip(),
            tool_calls=records,
            turns=max_turns,
            model=self.model_id,
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        model_id = os.environ.get("ARCA_TOOLS_MODEL")
        if not model_id:
            # Same contract as ARCA_RLM_ADAPTER: /health and CI must not download a model.
            raise NotImplementedError(
                "Phase 2 is not enabled: set ARCA_TOOLS_MODEL (e.g. Qwen/Qwen3-1.7B)."
            )
        loop = ToolLoop(model_id=model_id)
        loop.load()
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns)


__all__ = ["ToolCallRecord", "ToolLoop", "build_system_prompt", "build_tool_loop", "run_tool_loop"]
