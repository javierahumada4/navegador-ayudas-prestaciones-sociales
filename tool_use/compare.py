"""Evaluate same Qwen model with and without tools on the 30-case phase-2 benchmark.

This script runs *actual* inference; results must be measured on the DGX/GPU,
not invented. External BDNS calls are real. Report files are sandboxed.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

from tool_use.evaluate import load_bench, score_case, summarise
from tool_use.executor import ToolLoop


def evaluate_configuration(cases, loop, max_turns):
    scores, rows = [], []
    for case in cases:
        try:
            result = loop.run(case["query"], max_turns=max_turns)
            calls = [r.model_dump(mode="json") for r in result.tool_calls]
            score = score_case(case, calls)
            rows.append({
                "query": case["query"], "expected_tool": case.get("expected_tool"),
                "expected_arguments": case.get("expected_arguments"),
                "answer": result.answer, "tool_calls": calls,
                "turns": result.turns, **score,
            })
            scores.append(score)
        except Exception as exc:
            score = {"selection_ok": False, "arguments_ok": False}
            scores.append(score)
            rows.append({"query": case["query"], "error": str(exc), **score})
    return {"summary": summarise(scores), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="tool_use/bench/cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_comparison.json")
    parser.add_argument("--model", default=os.getenv("ARCA_TOOLS_MODEL_ID", "Qwen/Qwen3-1.7B"))
    parser.add_argument("--brain", choices=["base", "rlm"], default="base")
    parser.add_argument("--adapter", default=os.getenv("ARCA_RLM_ADAPTER"))
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()
    cases = load_bench(args.bench)
    if len(cases) != 30:
        raise ValueError(f"Expected 30 benchmark cases, found {len(cases)}")
    loop = ToolLoop(
        model_id=args.model, brain=args.brain, adapter_path=args.adapter,
        use_tools=True,
    )
    without = ToolLoop(
        model_id=args.model, brain=args.brain, adapter_path=args.adapter,
        use_tools=False,
    )
    with tempfile.TemporaryDirectory(prefix="arca-phase2-") as folder:
        previous = os.environ.get("ARCA_IMV_REPORTS_DIR")
        os.environ["ARCA_IMV_REPORTS_DIR"] = folder
        try:
            results = {
                "model": loop.model_label,
                "with_tools": evaluate_configuration(cases, loop, args.max_turns),
                "without_tools": evaluate_configuration(cases, without, 1),
                "limitations": (
                    "Tool selection is intentionally zero on tool-required cases "
                    "without tools; argument correctness is a strict exact-match "
                    "metric, not semantic correctness. BDNS is a live external API."
                ),
            }
        finally:
            if previous is None:
                os.environ.pop("ARCA_IMV_REPORTS_DIR", None)
            else:
                os.environ["ARCA_IMV_REPORTS_DIR"] = previous
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        config: value["summary"]
        for config, value in results.items() if isinstance(value, dict) and "summary" in value
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
