"""Full phase-2 comparison: actual inference with and without tools.

Reports are inspected (bytes, SHA256 and model text) before the isolated
temporary output directory is removed. No scores are hard-coded.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from tool_use.evaluate import load_bench, score_case, summarise
from tool_use.executor import ToolLoop
from tool_use.metrics import score_workflow, summarise_workflows


def evaluate_configuration(cases, loop, max_turns, *, report_root: Path | None = None):
    scores, rows = [], []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            result = loop.run(case["query"], max_turns=max_turns)
            calls = [record.model_dump(mode="json") for record in result.tool_calls]
            score = score_case(case, calls)
            workflow = score_workflow(case, result.answer, calls, report_root=report_root)
            row = {
                "id": case.get("id"), "category": case.get("category"),
                "query": case["query"], "expected_tool": case.get("expected_tool"),
                "expected_tools": case.get("expected_tools"),
                "expected_arguments": case.get("expected_arguments"),
                "answer": result.answer, "tool_calls": calls,
                "turns": result.turns, "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000, 2
                ), **score, "workflow": workflow,
            }
        except Exception as exc:
            score = {"selection_ok": False, "arguments_ok": False}
            row = {
                "id": case.get("id"), "category": case.get("category"),
                "query": case["query"], "error": f"{type(exc).__name__}: {exc}",
                "tool_calls": [], "answer": "",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                **score, "workflow": score_workflow(
                    case, "", [], report_root=report_root
                ),
            }
        scores.append(score)
        rows.append(row)
        print(
            f"[{index:02d}/{len(cases):02d}] {loop.model_label} "
            f"tools={loop.use_tools} {case.get('category', '?')}: "
            f"first={int(score['selection_ok'])} "
            f"workflow={int(row['workflow']['end_to_end_success'])}",
            flush=True,
        )

    summary = {
        **summarise(scores),
        **summarise_workflows(rows),
    }
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="tool_use/bench/cases.jsonl")
    parser.add_argument("--out", default="reports/phase2_comparison.json")
    parser.add_argument("--model", default=os.getenv("ARCA_TOOLS_MODEL_ID", "Qwen/Qwen3-1.7B"))
    parser.add_argument("--brain", choices=["base", "rlm"], default="base")
    parser.add_argument("--adapter", default=os.getenv("ARCA_RLM_ADAPTER"))
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Run the first N cases for a quick smoke test.")
    args = parser.parse_args()

    cases = load_bench(args.bench)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    if not cases:
        parser.error("Benchmark contains no cases")

    with_tools = ToolLoop(
        model_id=args.model, brain=args.brain, adapter_path=args.adapter,
        use_tools=True,
    )
    without_tools = ToolLoop(
        model_id=args.model, brain=args.brain, adapter_path=args.adapter,
        use_tools=False,
    )

    with tempfile.TemporaryDirectory(prefix="arca-phase2-") as folder:
        previous = os.environ.get("ARCA_IMV_REPORTS_DIR")
        os.environ["ARCA_IMV_REPORTS_DIR"] = folder
        try:
            results = {
                "model": with_tools.model_label,
                "benchmark_cases": len(cases),
                "with_tools": evaluate_configuration(
                    cases, with_tools, args.max_turns, report_root=Path(folder)
                ),
                "without_tools": evaluate_configuration(
                    cases, without_tools, 1, report_root=Path(folder)
                ),
                "metric_notes": {
                    "tool_selection_accuracy": "Teacher's first-call accuracy (unchanged).",
                    "argument_accuracy": "Teacher's strict normalized first-call arguments.",
                    "multi_step_success": "All required tools executed successfully; independent of first-call score.",
                    "report_created_and_verified": "A real TXT existed within the temporary directory; content, byte count and SHA256 matched the tool record.",
                    "amount_fidelity": "Three IMV/CAPI/total euro amounts found in answer or saved report, checked against an independently run deterministic oracle.",
                    "explanation_heuristic_accuracy": "Rule-based keyword and contradiction checks, NOT a validated legal fact-check.",
                    "end_to_end_success": "Calls completed, verified file where required, amounts and explanation checks met.",
                    "live_api": "BDNS is queried over real HTTP; network failure lowers success rates.",
                    "without_tools": "The first-tool metric is necessarily zero for requests that require external actions.",
                },
            }
        finally:
            if previous is None:
                os.environ.pop("ARCA_IMV_REPORTS_DIR", None)
            else:
                os.environ["ARCA_IMV_REPORTS_DIR"] = previous

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for label in ("with_tools", "without_tools"):
        print(label, json.dumps(results[label]["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
