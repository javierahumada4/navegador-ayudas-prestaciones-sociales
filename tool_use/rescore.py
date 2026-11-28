"""Re-score saved ``tool_use.compare`` outputs with the current bench and metrics.

The model generations, tool calls and observations stored in each row are reused as they
are; only the scoring is recomputed. Report files were verified on disk (bytes and SHA-256)
when the evaluation ran and are not kept in Git, so their stored verdict is reused and the
report text is taken from the ``guardar_informe`` call it was verified against.

Run::

    uv run python -m tool_use.rescore results/fase2/phase2_base.json \
        results/fase2/phase2_grpo.json --out results/fase2/phase2_rescored.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_use.evaluate import load_bench, score_case, summarise
from tool_use.metrics import score_workflow, summarise_workflows


def _stored_report(row: dict[str, Any]) -> tuple[bool, str]:
    if not row["workflow"].get("report_created_and_verified"):
        return False, ""
    saved = next(
        c for c in row["tool_calls"] if c["name"] == "guardar_informe" and c.get("ok")
    )
    return True, saved["arguments"]["contenido"]


def rescore_configuration(config: dict[str, Any], cases: dict[str, dict]) -> dict[str, Any]:
    scores, rows = [], []
    for row in config["rows"]:
        case = cases[row["id"]]
        score = score_case(case, row["tool_calls"])
        workflow = score_workflow(
            case, row["answer"], row["tool_calls"], report_status=_stored_report(row)
        )
        scores.append(score)
        rows.append({"id": row["id"], "category": row["category"], **score, "workflow": workflow})
    return {"summary": {**summarise(scores), **summarise_workflows(rows)}, "rows": rows}


def changed_cases(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for old, new in zip(before["rows"], after["rows"], strict=True):
        fields = {
            name: [old["workflow"].get(name), new["workflow"].get(name)]
            for name in new["workflow"]
            if name != "expected_engine" and old["workflow"].get(name) != new["workflow"].get(name)
        }
        for name in ("selection_ok", "arguments_ok"):
            if old[name] != new[name]:
                fields[name] = [old[name], new[name]]
        if fields:
            changes.append({"id": old["id"], "category": old["category"], "changes": fields})
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", help="JSON files written by tool_use.compare")
    parser.add_argument("--bench", default="tool_use/bench/cases.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = {case["id"]: case for case in load_bench(args.bench)}
    output = {"bench": args.bench, "runs": {}}
    for path in args.results:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        run = {"model": saved["model"]}
        for mode in ("with_tools", "without_tools"):
            rescored = rescore_configuration(saved[mode], cases)
            run[mode] = {
                "original": saved[mode]["summary"],
                "rescored": rescored["summary"],
                "changed_cases": changed_cases(saved[mode], rescored),
            }
            print(
                f"{Path(path).name} {mode}: end_to_end "
                f"{saved[mode]['summary']['end_to_end_success']:.3f} -> "
                f"{rescored['summary']['end_to_end_success']:.3f}",
                flush=True,
            )
        output["runs"][Path(path).name] = run
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
