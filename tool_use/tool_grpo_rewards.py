"""Pure, CPU-testable GRPO rewards for actual IMV tool calling.

TRL supplies the model's completion messages plus answer/reference_case columns.
The reference is used ONLY for reward computation, never placed in prompts.
Tool arguments are verified by the same Pydantic registry used at inference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from rlm.rewards import imv_numbers_match
from tool_use.imv_tools import IMVCase
from tool_use.parser import parse_tool_calls
from tool_use.tools import default_registry

AMOUNT = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}\Z")
MAX_ARGUMENTS_JSON_CHARS = 40000


def _messages(completion: Any) -> list[dict[str, Any]]:
    if isinstance(completion, dict):
        return [completion]
    if isinstance(completion, list):
        return [m for m in completion if isinstance(m, dict)]
    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]
    return []


def _call_from_structured(call: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    fn = call.get("function", call)
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        return None
    arguments = fn.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return fn["name"], arguments


def _all_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]] | None]:
    """Preserve malformed attempts: never repair JSON or skip invalid calls."""
    calls: list[tuple[str, dict[str, Any]] | None] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        structured = msg.get("tool_calls")
        if structured:
            if not isinstance(structured, list):
                calls.append(None)
                continue
            for call in structured:
                calls.append(_call_from_structured(call) if isinstance(call, dict) else None)
        else:
            content = msg.get("content")
            if isinstance(content, str):
                if "<tool_call" in content and not parse_tool_calls(content).tool_calls:
                    calls.append(None)  # Incomplete/unterminated call.
                for parsed in parse_tool_calls(content).tool_calls:
                    calls.append(
                        (parsed.name, parsed.arguments)
                        if parsed.error is None else None
                    )
    return calls


def _last_answer(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if "<tool_call" in content:
            continue
        # Tool GRPO follows the same plain numeric final-answer contract as ToolLoop.
        # Allow the existing phase-1 <answer> tag only as a single final block.
        from rlm.rewards import extract_unique_final_answer

        if "<answer" in content or "</answer" in content:
            return extract_unique_final_answer(content)
        if "<think" in content or "</think" in content:
            return None
        return content or None
    return None


def _reference(reference_case: str | dict[str, Any]) -> dict[str, Any] | None:
    try:
        case = json.loads(reference_case) if isinstance(reference_case, str) else reference_case
        if not isinstance(case, dict):
            return None
        return IMVCase.model_validate(case).model_dump(mode="json", by_alias=True)
    except (ValueError, TypeError):
        return None


def inspect_rollout(
    completion: Any, answer: str, reference_case: str | dict[str, Any]
) -> dict[str, bool]:
    """Check selection, oracle argument match, real tool result and final answer."""
    messages = _messages(completion)
    calls = _all_calls(messages)
    reference = _reference(reference_case)
    selected = False
    valid = False
    correct_args = False
    executed = False
    # For these one-calculation benchmark problems the desired policy makes
    # exactly one valid tool call; do not reward spraying calls at the engine.
    for candidate in calls if len(calls) == 1 else []:
        if candidate is None or candidate[0] != "calcular_imv":
            continue
        selected = True
        arguments = candidate[1]
        if len(json.dumps(arguments, ensure_ascii=False, default=str)) > MAX_ARGUMENTS_JSON_CHARS:
            continue
        # Re-run the SAME deterministic tool and Pydantic validation used by the
        # production endpoint. TRL may have already run it; this is idempotent.
        record = default_registry().call("calcular_imv", arguments)
        if not record.ok:
            continue
        valid = True
        normalized = IMVCase.model_validate(arguments["case"]).model_dump(
            mode="json", by_alias=True
        )
        if reference is None or normalized != reference:
            continue
        correct_args = True
        result = record.result
        if isinstance(result, dict) and imv_numbers_match(
            str(result.get("total_monthly_eur")), str(answer)
        ):
            executed = True
            break
    final = _last_answer(messages)
    return {
        "selected": selected,
        "arguments_valid": valid,
        "arguments_correct": correct_args,
        "executed_correctly": executed,
        "final_amount_correct": imv_numbers_match(final, str(answer)),
        "final_format": bool(final and AMOUNT.fullmatch(final)),
    }


def _batch_values(
    completions: Sequence, answer: Sequence, reference_case: Sequence
) -> list[dict[str, bool]]:
    if len(completions) != len(answer) or len(completions) != len(reference_case):
        raise ValueError("Reward columns have mismatched batch sizes")
    return [
        inspect_rollout(c, str(a), r)
        for c, a, r in zip(completions, answer, reference_case, strict=True)
    ]


def tool_selection_reward(completions, answer, reference_case, **kwargs) -> list[float]:
    """Small shaping reward for generating a well-formed calcular_imv call."""
    return [
        float(r["selected"]) for r in _batch_values(completions, answer, reference_case)
    ]


def tool_arguments_reward(completions, answer, reference_case, **kwargs) -> list[float]:
    """Reward exactly matching the reference synthetic case, not fabricated fields."""
    return [
        float(r["arguments_correct"])
        for r in _batch_values(completions, answer, reference_case)
    ]


def tool_execution_reward(completions, answer, reference_case, **kwargs) -> list[float]:
    """Reward executing the validated tool with reference-correct arguments."""
    return [
        float(r["executed_correctly"])
        for r in _batch_values(completions, answer, reference_case)
    ]


def final_answer_reward(completions, answer, reference_case, **kwargs) -> list[float]:
    """Reward a correct final numerical answer ONLY after a correct tool execution."""
    return [
        float(r["executed_correctly"] and r["final_amount_correct"])
        for r in _batch_values(completions, answer, reference_case)
    ]


def numeric_format_reward(completions, answer, reference_case, **kwargs) -> list[float]:
    """Weak independent incentive to end with an unambiguous two-decimal value."""
    return [
        float(r["final_format"])
        for r in _batch_values(completions, answer, reference_case)
    ]
