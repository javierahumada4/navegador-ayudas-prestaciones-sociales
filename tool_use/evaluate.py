"""Phase 2 evaluation: does the model pick the right tool with the right arguments?

The benchmark is a JSONL file (see ``tool_use/bench/example_cases.jsonl``) where each line
holds a user query and the expected first tool call::

    {"query": "What's the temperature in Paris right now?",
     "expected_tool": "get_weather", "expected_arguments": {"location": "Paris, France"},
     "needs_tool": true}

Include cases where *no* tool is needed: a model that calls tools for everything is not
using them well. Two metrics are computed:

* tool selection accuracy: the first call names the expected tool (or there is no call when
  ``needs_tool`` is false);
* argument accuracy: among correct selections, the arguments match after normalisation.

Run::

    uv run python -m tool_use.evaluate --bench tool_use/bench/cases.jsonl \\
        --out reports/phase2_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_bench(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalise(value: Any) -> Any:
    """Make argument comparison tolerant to case, spacing and numeric formatting."""
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    return value


def score_case(case: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, bool]:
    """Compare the calls the model made against the expectation for one benchmark case."""
    first = tool_calls[0] if tool_calls else None
    if not case.get("needs_tool", True):
        selection_ok = first is None
        return {"selection_ok": selection_ok, "arguments_ok": selection_ok}
    selection_ok = first is not None and first["name"] == case["expected_tool"]
    arguments_ok = selection_ok and normalise(first["arguments"]) == normalise(
        case.get("expected_arguments", {})
    )
    return {"selection_ok": selection_ok, "arguments_ok": arguments_ok}


def summarise(scores: list[dict[str, bool]]) -> dict[str, float]:
    n = max(len(scores), 1)
    return {
        "tool_selection_accuracy": sum(s["selection_ok"] for s in scores) / n,
        "argument_accuracy": sum(s["arguments_ok"] for s in scores) / n,
        "n_cases": len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="tool_use/bench/example_cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_eval.json")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--brain", choices=["base", "rlm"], default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    from tool_use.executor import ToolLoop

    loop = ToolLoop(
        brain=args.brain, model_id=args.model_id, adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )

    cases = load_bench(args.bench)
    rows = []
    for case in cases:
        raw_generations: list[str] = []
        response = loop.run(
            case["query"], max_turns=args.max_turns, trace=raw_generations
        )
        calls = [c.model_dump() for c in response.tool_calls]
        expected_tool = case.get("expected_tool")
        execution_ok = (
            any(call["name"] == expected_tool and call["ok"] for call in calls)
            if case.get("needs_tool", True) else None
        )
        scores = score_case(case, calls)
        first_repaired = bool(calls and calls[0].get("repaired", False))
        native_execution_ok = (
            any(
                call["name"] == expected_tool and call["ok"]
                and not call.get("repaired", False)
                for call in calls
            )
            if case.get("needs_tool", True) else None
        )
        expected_phrases = case.get("expected_answer_contains")
        answer_check_ok = (
            all(phrase.casefold() in response.answer.casefold()
                for phrase in expected_phrases)
            if expected_phrases else None
        )
        rows.append({
            **case,
            "tool_calls": calls,
            "raw_generations": raw_generations,
            "answer": response.answer,
            "tool_execution_ok": execution_ok,
            "native_tool_execution_ok": native_execution_ok,
            "answer_check_ok": answer_check_ok,
            "native_selection_ok": scores["selection_ok"] and not first_repaired,
            "native_arguments_ok": scores["arguments_ok"] and not first_repaired,
            "repaired_calls": sum(call.get("repaired", False) for call in calls),
            **scores,
        })
        print(f"{'✓' if rows[-1]['selection_ok'] else '✗'} {case['query'][:70]}")

    summary = summarise(rows)
    requiring_tool = [r for r in rows if r.get("needs_tool", True)]
    with_answer_checks = [r for r in rows if r["answer_check_ok"] is not None]
    summary["native_tool_selection_accuracy"] = (
        sum(r["native_selection_ok"] for r in rows) / len(rows) if rows else None
    )
    summary["native_argument_accuracy"] = (
        sum(r["native_arguments_ok"] for r in rows) / len(rows) if rows else None
    )
    summary["native_tool_execution_accuracy"] = (
        sum(r["native_tool_execution_ok"] for r in requiring_tool) / len(requiring_tool)
        if requiring_tool else None
    )
    summary["repaired_tool_calls"] = sum(r["repaired_calls"] for r in rows)
    summary["tool_execution_accuracy"] = (
        sum(r["tool_execution_ok"] for r in requiring_tool) / len(requiring_tool)
        if requiring_tool else None
    )
    summary["answer_check_accuracy"] = (
        sum(r["answer_check_ok"] for r in with_answer_checks) /
        len(with_answer_checks) if with_answer_checks else None
    )
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "configuration": {
                    "brain": loop.brain,
                    "model": loop.model_label,
                    "max_turns": args.max_turns,
                    "max_new_tokens": loop.max_new_tokens,
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
