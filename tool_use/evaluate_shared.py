"""Evaluate Qwen3 + calcular_imv on EXACTLY the Phase-1 IMV test questions.

Input: the SAME test.jsonl (question/answer/params/rule_context) used by
rlm.evaluate. Only rule_context and question are sent to the model; oracle
answers and params are never given to it. CPU unit tests can stub ToolLoop.
Numeric accuracy uses the Phase-1 IMV verifier, but the tool answer has a
different output contract from the Phase-1 <think>/<answer> trace. Report
numeric accuracy separately rather than equating incompatible format scores.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rlm.rewards import imv_numbers_match

TWO_DECIMALS = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}\Z")


def load_shared_test(path: str | Path, n_examples: int | None = None) -> list[dict[str, Any]]:
    """Load the Phase-1 JSONL without modifying or filtering its questions."""
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case.get("question"), str) or not isinstance(
                case.get("answer"), (str, int, float)
            ):
                raise ValueError("Every test case needs question and answer")
            cases.append(case)
            if n_examples is not None and len(cases) >= n_examples:
                break
    return cases


def prompt_for_case(case: dict[str, Any]) -> str:
    """Mirror the user-message rule sheet used by Phase-1 inference."""
    context = case.get("rule_context")
    if not context:
        from rlm.generate_problems import build_rule_context
        from rlm.imv_engine import load_ruleset

        context = build_rule_context(load_ruleset())
    return f"{context}\n\n{case['question']}"


def score_response(
    case: dict[str, Any], response: Any, raw_generations: list[str]
) -> dict[str, Any]:
    answer = response.answer.strip()
    expected = str(case["answer"])
    numeric_correct = imv_numbers_match(answer, expected)
    strict_format = TWO_DECIMALS.fullmatch(answer) is not None
    calls = [call.model_dump() for call in response.tool_calls]
    tool_used = any(call["name"] == "calcular_imv" and call["ok"] for call in calls)
    return {
        "question": case["question"],
        "expected": expected,
        "family": (case.get("branches") or {}).get("family"),
        "answer": answer,
        "numeric_correct": numeric_correct,
        "strict_numeric_format": strict_format,
        "tool_execution_ok": tool_used,
        "correct_with_tool": bool(numeric_correct and tool_used),
        "tool_calls": calls,
        "raw_generations": raw_generations,
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_problems": 0, "numeric_accuracy": None, "strict_numeric_format_rate": None,
            "tool_execution_rate": None, "correct_with_tool_rate": None,
        }
    return {
        "n_problems": len(rows),
        "numeric_accuracy": sum(r["numeric_correct"] for r in rows) / len(rows),
        "strict_numeric_format_rate": (
            sum(r["strict_numeric_format"] for r in rows) / len(rows)
        ),
        "tool_execution_rate": sum(r["tool_execution_ok"] for r in rows) / len(rows),
        "correct_with_tool_rate": sum(r["correct_with_tool"] for r in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="EXACT Phase-1 test.jsonl path")
    parser.add_argument("--out", default="reports/tool_vs_phase1.json")
    parser.add_argument("--brain", choices=["base", "rlm"], default="base")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.n_examples is not None and args.n_examples <= 0:
        parser.error("--n-examples must be positive")

    from tool_use.executor import ToolLoop

    loop = ToolLoop(
        brain=args.brain, model_id=args.model_id, adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )
    cases = load_shared_test(args.data, args.n_examples)
    rows = []
    for index, case in enumerate(cases, 1):
        trace: list[str] = []
        response = loop.run(prompt_for_case(case), max_turns=args.max_turns, trace=trace)
        row = score_response(case, response, trace)
        rows.append(row)
        print(
            f"{index}/{len(cases)} numeric={'yes' if row['numeric_correct'] else 'no'} "
            f"tool={'yes' if row['tool_execution_ok'] else 'no'}",
            flush=True,
        )
    result = {
        "configuration": {
            "phase1_data": str(args.data), "brain": loop.brain, "model": loop.model_label,
            "max_turns": args.max_turns, "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "scoring_note": (
                "Numeric accuracy uses the Phase-1 IMV verifier. Phase-1 pass@1 "
                "also requires <think>/<answer> formatting, which is not the "
                "tool-loop output contract; compare numeric correctness separately."
            ),
        },
        "summary": summarise(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
