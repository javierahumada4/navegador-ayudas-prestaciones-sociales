"""Delta table between two scorecards (baseline vs fixes), metric by metric.

    python -m pipeline_eval.compare base/scorecard.json fixes/scorecard.json --out delta.md

Higher is better for every metric except the ones in ``LOWER_IS_BETTER`` (errors, latency,
truncation, false abstentions, steps and tokens). With n = 40-58 per phase, a difference of one or
two items is noise: the table prints the counts so nobody reads 0.025 as a result.
"""

from __future__ import annotations

import argparse
import json
import math
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


COUNT_METRICS = {"over_300s", "http_errors", "p50_s", "p90_s", "max_s", "mean_steps", "mean_tokens", "n"}


def beyond_noise(metric: str, before, after, n_before, n_after) -> str:
    """'sí' when a difference of rates exceeds 2 binomial standard errors, '' when it does not."""
    if metric in COUNT_METRICS or not all(isinstance(v, (int, float)) for v in (before, after)):
        return ""
    if not (0 <= before <= 1 and 0 <= after <= 1) or not n_before or not n_after:
        return ""
    se = math.sqrt(before * (1 - before) / n_before + after * (1 - after) / n_after)
    if se == 0:
        return "sí" if before != after else ""
    return "sí" if abs(after - before) > 2 * se else "no"


def verdict(metric: str, before, after) -> str:
    if before is None or after is None or before == after:
        return ""
    better = (after < before) if metric in LOWER_IS_BETTER else (after > before)
    return "mejor" if better else "peor"


def table(before: dict, after: dict) -> str:
    sb, sa = before["summary"], after["summary"]
    lines = [f"# {before.get('label')} → {after.get('label')}", "",
             "| fase | métrica | antes | después | Δ | | ¿> 2σ? |", "|---|---|---:|---:|---:|---|---|"]
    for phase, metric in ROWS:
        b, a = sb.get(phase, {}).get(metric), sa.get(phase, {}).get(metric)
        delta = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        noise = beyond_noise(metric, b, a, sb.get(phase, {}).get("n"), sa.get(phase, {}).get("n"))
        lines.append(f"| {phase} | {metric} | {fmt(b)} | {fmt(a)} | {fmt(delta)} | "
                     f"{verdict(metric, b, a)} | {noise} |")
    for brain in sorted(set(sb.get("phase4", {})) | set(sa.get("phase4", {}))):
        for metric in AGENT_ROWS:
            b = sb.get("phase4", {}).get(brain, {}).get(metric)
            a = sa.get("phase4", {}).get(brain, {}).get(metric)
            delta = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            n = 20 if metric in {"success", "chain_complete"} else (3 if metric == "impossible_ok" else None)
            lines.append(f"| phase4 {brain} | {metric} | {fmt(b)} | {fmt(a)} | {fmt(delta)} | "
                         f"{verdict(metric, b, a)} | {beyond_noise(metric, b, a, n, n)} |")
    mb, ma = sb.get("memory", {}), sa.get("memory", {})
    lines.append(f"| memoria | conversaciones OK | {mb.get('correct', '—')}/{mb.get('n', '—')} | "
                 f"{ma.get('correct', '—')}/{ma.get('n', '—')} | | | |")
    n = {p: sa.get(p, {}).get("n") for p in ("phase1", "phase2", "phase3")}
    lines += ["", f"n por fase: {n}; agente 23 tareas por cerebro. Una o dos preguntas de "
              "diferencia son ruido con estos tamaños."]
    return "\n".join(lines) + "\n"


def rescore(card: dict, bench: Path = Path("agent/bench/tasks.jsonl")) -> dict:
    """Re-apply the current agent-bench checks to the stored answers of every phase-4 row.

    The checks were corrected after the baseline run (right answers scored as wrong); applying
    the same checks to both variants keeps the comparison fair. Rows from before ``succeeded``
    was stored infer it from the agent's own failure messages ("No he podido ...").
    """
    from agent.evaluate import check_answer
    from pipeline_eval.scorecard import summarize_phase4

    tasks = {t["id"]: t for t in (json.loads(x) for x in bench.read_text(encoding="utf-8").splitlines() if x.strip())}
    for name, rows in card.get("rows", {}).items():
        if not name.startswith("phase4_"):
            continue
        for row in rows:
            task = tasks.get(row["id"])
            if task is None or row.get("status") != 200:
                continue
            answer = row.get("final_answer", "")
            ok = row.get("succeeded", not answer.startswith("No he podido"))
            row["correct"] = bool(ok) and check_answer(task, answer)
        card["summary"].setdefault("phase4", {})[name.removeprefix("phase4_")] = summarize_phase4(rows)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-rescore", action="store_true", help="keep the checks of the run")
    args = parser.parse_args()
    before, after = json.loads(args.before.read_text()), json.loads(args.after.read_text())
    if not args.no_rescore:
        before, after = rescore(before), rescore(after)
    text = table(before, after)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
