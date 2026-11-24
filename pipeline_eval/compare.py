"""Delta table between two scorecards (baseline vs fixes), metric by metric.

    python -m pipeline_eval.compare base/scorecard.json fixes/scorecard.json --out delta.md

Higher is better for every metric except the ones in ``LOWER_IS_BETTER`` (errors, latency,
truncation, false abstentions, steps and tokens). With n = 40-58 per phase, a difference of one or
two items is noise: the table prints the counts so nobody reads 0.025 as a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("phase1", "pass@1"), ("phase1", "format"), ("phase1", "truncated"), ("phase1", "over_300s"),
    ("phase1", "p50_s"),
    ("phase2", "selection"), ("phase2", "arguments"), ("phase2", "tool_sequence"),
    ("phase2", "end_to_end"), ("phase2", "amount_fidelity"), ("phase2", "over_300s"), ("phase2", "p50_s"),
    ("phase3", "recall@5"), ("phase3", "mrr"), ("phase3", "cited_relevant"), ("phase3", "citations_exist"),
    ("phase3", "false_abstentions"), ("phase3", "unanswerable_abstained"), ("phase3", "p50_s"),
]
AGENT_ROWS = ["success", "impossible_ok", "chain_complete", "mean_steps", "mean_tokens", "over_300s", "p50_s"]
LOWER_IS_BETTER = {"truncated", "over_300s", "p50_s", "p90_s", "http_errors", "false_abstentions",
                   "mean_steps", "mean_tokens"}


def fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def verdict(metric: str, before, after) -> str:
    if before is None or after is None or before == after:
        return ""
    better = (after < before) if metric in LOWER_IS_BETTER else (after > before)
    return "mejor" if better else "peor"


def table(before: dict, after: dict) -> str:
    sb, sa = before["summary"], after["summary"]
    lines = [f"# {before.get('label')} → {after.get('label')}", "",
             "| fase | métrica | antes | después | Δ | |", "|---|---|---:|---:|---:|---|"]
    for phase, metric in ROWS:
        b, a = sb.get(phase, {}).get(metric), sa.get(phase, {}).get(metric)
        delta = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        lines.append(f"| {phase} | {metric} | {fmt(b)} | {fmt(a)} | {fmt(delta)} | {verdict(metric, b, a)} |")
    for brain in sorted(set(sb.get("phase4", {})) | set(sa.get("phase4", {}))):
        for metric in AGENT_ROWS:
            b = sb.get("phase4", {}).get(brain, {}).get(metric)
            a = sa.get("phase4", {}).get(brain, {}).get(metric)
            delta = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            lines.append(f"| phase4 {brain} | {metric} | {fmt(b)} | {fmt(a)} | {fmt(delta)} | "
                         f"{verdict(metric, b, a)} |")
    mb, ma = sb.get("memory", {}), sa.get("memory", {})
    lines.append(f"| memoria | conversaciones OK | {mb.get('correct', '—')}/{mb.get('n', '—')} | "
                 f"{ma.get('correct', '—')}/{ma.get('n', '—')} | | |")
    n = {p: sa.get(p, {}).get("n") for p in ("phase1", "phase2", "phase3")}
    lines += ["", f"n por fase: {n}; agente 23 tareas por cerebro. Una o dos preguntas de "
              "diferencia son ruido con estos tamaños."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = table(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
