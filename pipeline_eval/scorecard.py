"""Scorecard of the whole pipeline, measured the way the course grades it: through the API.

"No leemos vuestros notebooks para saber si funciona: lanzamos peticiones." This script sends the
requests a grader would send to a running API and scores every phase with that phase's own
metrics, so one file says how well the deployed system does today:

- **health**: every phase reports ``ready``.
- **phase 1** ``/reasoning``: a fixed sample of the rlm_final test split with ``expected_answer``:
  pass@1 (verifier verdict), valid format, truncation, tokens.
- **phase 2** ``/tools``: the 40-case phase-2 bench, scored with ``tool_use.evaluate.score_case``
  and ``tool_use.metrics.score_workflow`` (selection, arguments, sequence, end to end...).
- **phase 3** ``/rag``: the 50 gold questions (Recall@5 and MRR on the returned chunks, citations
  that exist, a relevant chunk cited) and the 8 unanswerable ones (correct "No lo sé").
- **phase 4** ``/agent``: the 23-task agent bench per brain (``agent.evaluate.score``) and two
  two-turn conversations that only succeed if the second turn remembers the first.
- **robustness**, for every call: HTTP errors and requests slower than 300 s (ngrok cuts them and
  the grader gets an HTML error page instead of JSON).

    uv run python -m pipeline_eval.scorecard --api http://localhost:8899 --out reports/scorecard.json

Results are written after every request (``--resume`` continues a cut run).
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
import time
from pathlib import Path

import httpx

NGROK_LIMIT_S = 300
PHASE1_DATA = Path("results/rlm_final/data/test.jsonl")
PHASE2_BENCH = Path("tool_use/bench/cases.jsonl")
AGENT_BENCH = Path("agent/bench/tasks.jsonl")
ABSTENTION = re.compile(r"(?i)^\s*no lo s[eé]")
CASE_A = {
    "application_date": "2026-06-01", "applicant_id": "adult",
    "persons": [{"id": "adult", "date_of_birth": "1980-01-01", "legal_residence_in_spain": True,
                 "effective_residence_in_spain": True,
                 "continuous_legal_effective_residence_since": "2020-01-01"}],
    "domicile": {"co_resident_ids": ["adult"], "same_domicile_since": "2020-01-01"},
    "relationships": [], "applicant_history": {"independent_from_parents_since": "2010-01-01"},
    "economic": {"countable_income_annual_eur": 0, "net_worth_eur": 0, "non_corporate_assets_eur": 0},
}
CONVERSATIONS = [  # (turn 1, turn 2, check on turn 2): turn 2 is unanswerable without turn 1
    (f"Calcula con el motor el IMV de este caso: {json.dumps(CASE_A, ensure_ascii=False)}",
     "¿Y cuánto suma esa cantidad en un año (12 pagas)?",
     {"type": "number", "value": 8803.2, "tolerance": 0.05}),
    ("Busca en la base de conocimiento la edad mínima para ser titular del IMV.",
     "¿Y cuántos años le faltan para cumplirla a alguien de 19 años?",
     {"type": "number", "value": 4}),
]


class Client:
    def __init__(self, base: str, timeout: float, transport: httpx.BaseTransport | None = None) -> None:
        self.http = httpx.Client(base_url=base.rstrip("/"), timeout=timeout, transport=transport)

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        started = time.perf_counter()
        try:
            reply = self.http.request(method, path, json=body)
            status, payload = reply.status_code, None
            try:
                payload = reply.json()
            except ValueError:
                payload = {"raw": reply.text[:500]}
        except httpx.HTTPError as exc:
            status, payload = 0, {"error": f"{type(exc).__name__}: {exc}"}
        wall = time.perf_counter() - started
        return {"status": status, "body": payload, "wall_s": round(wall, 1),
                "over_ngrok_limit": wall > NGROK_LIMIT_S}


def latency(rows: list[dict]) -> dict:
    walls = sorted(r["wall_s"] for r in rows) or [0.0]
    return {
        "n": len(rows),
        "http_errors": sum(r["status"] != 200 for r in rows),
        "over_300s": sum(r["over_ngrok_limit"] for r in rows),
        "p50_s": statistics.median(walls),
        "p90_s": walls[min(len(walls) - 1, int(0.9 * len(walls)))],
        "max_s": walls[-1],
    }


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- phases


def phase1_items(n: int) -> list[dict]:
    rows = jsonl(PHASE1_DATA)
    random.Random(0).shuffle(rows)  # fixed sample, every family represented in expectation
    def family(row: dict) -> str:
        branches = row.get("branches")
        if isinstance(branches, str):
            try:
                branches = ast.literal_eval(branches)
            except (ValueError, SyntaxError):
                return ""
        return branches.get("family", "") if isinstance(branches, dict) else ""

    return [{"id": f"P1-{i:03d}", "question": r["question"], "expected": str(r["answer"]),
             "family": family(r)} for i, r in enumerate(rows[:n])]


def run_phase1(client: Client, item: dict) -> dict:
    res = client.call("POST", "/reasoning", {"question": item["question"],
                                             "expected_answer": item["expected"]})
    body = res["body"] if res["status"] == 200 else {}
    verdict = body.get("verifier") or {}
    return {**res, "id": item["id"], "family": item["family"],
            "correct": bool(verdict.get("is_correct")), "format": bool(body.get("has_valid_format")),
            "truncated": bool(body.get("truncated")), "tokens": body.get("tokens_generated", 0),
            "answer": body.get("answer", ""), "expected": item["expected"]}


def summarize_phase1(rows: list[dict]) -> dict:
    return {"pass@1": mean(r["correct"] for r in rows), "format": mean(r["format"] for r in rows),
            "truncated": mean(r["truncated"] for r in rows),
            "mean_tokens": mean(r["tokens"] for r in rows), **latency(rows)}


def run_phase2(client: Client, case: dict) -> dict:
    from tool_use.evaluate import score_case
    from tool_use.metrics import score_workflow

    res = client.call("POST", "/tools", {"query": case["query"], "max_turns": 5})
    body = res["body"] if res["status"] == 200 else {}
    calls = body.get("tool_calls", [])
    return {**res, "id": case.get("id"), "category": case.get("category"),
            **score_case(case, calls),
            "workflow": score_workflow(case, body.get("answer", ""), calls),
            "answer": body.get("answer", "")}


def summarize_phase2(rows: list[dict]) -> dict:
    from tool_use.metrics import summarise_workflows

    wf = summarise_workflows(rows)
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["workflow"]["end_to_end_success"])
    return {"selection": mean(r["selection_ok"] for r in rows),
            "arguments": mean(r["arguments_ok"] for r in rows),
            "end_to_end": wf["end_to_end_success"], "tool_sequence": wf["tool_sequence_success"],
            "amount_fidelity": wf["amount_fidelity"],
            "end_to_end_by_category": {k: mean(v) for k, v in sorted(by_cat.items())},
            **latency(rows)}


def rag_items() -> tuple[list, list[str]]:
    import os

    from rag.chunking import build_chunks
    from rag.generate import DEFAULT_CHUNKING
    from rag.gold import load_gold, load_unanswerable, resolve_gold
    from rag.ingest import load_corpus

    spec = os.environ.get("ARCA_RAG_CHUNKING", DEFAULT_CHUNKING)
    if not spec.startswith("recursive"):
        raise SystemExit("the scorecard resolves gold ids with a recursive chunking only")
    chunks = build_chunks(spec, load_corpus())
    return resolve_gold(load_gold(), chunks), load_unanswerable()


def run_phase3(client: Client, item, retriever: str) -> dict:
    from rag.gold import item_reciprocal_rank, item_recall

    question = item if isinstance(item, str) else item.question
    res = client.call("POST", "/rag", {"question": question, "top_k": 5, "retriever": retriever})
    body = res["body"] if res["status"] == 200 else {}
    ids = [c["id"] for c in body.get("chunks", [])]
    cited = body.get("citations", [])
    answer = body.get("answer", "")
    row = {**res, "question": question, "answer": answer, "citations": cited,
           "abstained": bool(ABSTENTION.search(answer)),
           "citations_exist": set(cited) <= set(ids)}
    if isinstance(item, str):
        return {**row, "id": "U-" + question[:30], "answerable": False}
    relevant = item.relevant_chunk_ids
    return {**row, "id": item.id, "answerable": True, "recall@5": item_recall(ids, item, 5),
            "mrr": item_reciprocal_rank(ids, item), "cited_relevant": bool(set(cited) & relevant),
            "returned_ids": ids}


def summarize_phase3(rows: list[dict]) -> dict:
    gold = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    return {"recall@5": mean(r["recall@5"] for r in gold), "mrr": mean(r["mrr"] for r in gold),
            "cited_relevant": mean(r["cited_relevant"] for r in gold),
            "false_abstentions": mean(r["abstained"] for r in gold),
            "citations_exist": mean(r["citations_exist"] for r in rows),
            "unanswerable_abstained": mean(r["abstained"] for r in neg), **latency(rows)}


def run_phase4(client: Client, task: dict, brain: str, max_steps: int) -> dict:
    from agent.evaluate import score

    res = client.call("POST", "/agent", {"task": task["task"], "brain": brain, "max_steps": max_steps})
    body = res["body"] if res["status"] == 200 else {"final_answer": "", "steps": [],
                                                     "succeeded": False, "n_steps": 0, "tokens_used": 0}
    return {**res, "id": task["id"], "category": task["category"], "brain": brain,
            **score(task, body), "final_answer": body.get("final_answer", ""),
            "n_steps": body.get("n_steps", 0), "tokens_used": body.get("tokens_used", 0)}


def summarize_phase4(rows: list[dict]) -> dict:
    chained = [r for r in rows if not r["id"].startswith("I")]
    impossible = [r for r in rows if r["id"].startswith("I")]
    return {"success": mean(r["correct"] for r in chained),
            "impossible_ok": mean(r["correct"] for r in impossible),
            "chain_complete": mean(r["required_tools_used"] for r in chained),
            "mean_steps": mean(r["n_steps"] for r in rows),
            "mean_tokens": mean(r["tokens_used"] for r in rows), **latency(rows)}


def run_conversation(client: Client, conv: tuple, brain: str) -> dict:
    from agent.evaluate import check_answer

    first, second, check = conv
    one = client.call("POST", "/agent", {"task": first, "brain": brain, "max_steps": 6,
                                         "session_id": "new"})
    sid = (one["body"] or {}).get("session_id") if one["status"] == 200 else None
    if not sid:
        return {**one, "id": "conv-" + first[:20], "correct": False, "detail": "no session_id returned"}
    two = client.call("POST", "/agent", {"task": second, "brain": brain, "max_steps": 6,
                                         "session_id": sid})
    answer = (two["body"] or {}).get("final_answer", "") if two["status"] == 200 else ""
    return {**two, "wall_s": one["wall_s"] + two["wall_s"], "id": "conv-" + second[:20],
            "correct": check_answer({}, answer, check), "answer": answer}


# --------------------------------------------------------------------------- report


def markdown(card: dict) -> str:
    s = card["summary"]
    lines = [f"# Scorecard — {card['label']}", "", f"API: `{card['api']}` · {card['finished']}", ""]
    h = s.get("health", {})
    lines += [f"**Health:** {h.get('ready', '?')}/{h.get('phases', '?')} fases ready "
              f"({', '.join(f'{k}={v}' for k, v in h.get('status', {}).items())})", ""]

    def fmt(v):
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    for phase, keys in (
        ("phase1", ["pass@1", "format", "truncated", "mean_tokens"]),
        ("phase2", ["selection", "arguments", "tool_sequence", "end_to_end", "amount_fidelity"]),
        ("phase3", ["recall@5", "mrr", "cited_relevant", "citations_exist", "false_abstentions",
                    "unanswerable_abstained"]),
    ):
        if phase not in s:
            continue
        m = s[phase]
        lines += [f"## {phase}", "", "| " + " | ".join(keys + ["n", "errores", ">300 s", "p50 s", "p90 s"]) + " |",
                  "|" + "---|" * (len(keys) + 5),
                  "| " + " | ".join(fmt(m.get(k)) for k in keys + ["n", "http_errors", "over_300s", "p50_s", "p90_s"]) + " |", ""]
    if "phase4" in s:
        keys = ["success", "impossible_ok", "chain_complete", "mean_steps", "mean_tokens"]
        lines += ["## phase4", "", "| cerebro | " + " | ".join(keys + ["errores", ">300 s", "p50 s", "p90 s"]) + " |",
                  "|---|" + "---|" * (len(keys) + 4)]
        for brain, m in s["phase4"].items():
            lines.append(f"| {brain} | " + " | ".join(fmt(m.get(k)) for k in keys + ["http_errors", "over_300s", "p50_s", "p90_s"]) + " |")
        lines.append("")
    if "memory" in s:
        lines += [f"**Memoria multi-turno:** {s['memory']['correct']}/{s['memory']['n']} conversaciones "
                  "recuerdan el turno anterior.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--label", default="pipeline")
    parser.add_argument("--out", default="reports/scorecard.json")
    parser.add_argument("--phases", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--phase1-n", type=int, default=40)
    parser.add_argument("--rag-retrievers", nargs="+", default=["hybrid"])
    parser.add_argument("--brains", nargs="+", default=["base", "rlm", "thinking"])
    parser.add_argument("--agent-max-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    card = json.loads(out.read_text()) if args.resume and out.exists() else {"rows": {}}
    card.update({"label": args.label, "api": args.api})
    client = Client(args.api, args.timeout)

    def save() -> None:
        card["finished"] = time.strftime("%Y-%m-%d %H:%M")
        out.write_text(json.dumps(card, indent=2, ensure_ascii=False, default=list))
        out.with_suffix(".md").write_text(markdown(card), encoding="utf-8")

    oom_streak = [0]

    def section(name: str, items, runner, summarizer, label=lambda x: x["id"]):
        # Resume keeps only successful rows: an HTTP error (OOM, a cut run) is retried.
        rows = card["rows"][name] = [r for r in card["rows"].get(name, []) if r.get("status") == 200]
        done = {r["id"] for r in rows}
        for item in items:
            if label(item) in done:
                continue
            row = runner(item)
            if row.get("status") != 200 and "out of memory" in json.dumps(row.get("body", "")).lower():
                oom_streak[0] += 1
                if oom_streak[0] >= 3:
                    save()
                    raise SystemExit("3 CUDA OOM errors in a row: the GPU is shared, stop the other "
                                     "process and re-run with --resume (failed rows are retried)")
            else:
                oom_streak[0] = 0
            rows.append(row)
            print(f"{name:>16} {row['id']}: status={row['status']} {row['wall_s']:.0f}s "
                  f"{'OK' if row.get('correct', row.get('recall@5', 1)) else '--'}", flush=True)
            card.setdefault("summary", {})
            save()
        return summarizer(rows)

    summary = card.setdefault("summary", {})
    health = client.call("GET", "/health")
    phases = (health["body"] or {}).get("phases", {}) if health["status"] == 200 else {}
    summary["health"] = {"phases": len(phases), "ready": sum(p.get("status") == "ready" for p in phases.values()),
                         "status": {k: p.get("status") for k, p in phases.items()}}
    save()

    if "1" in args.phases:
        summary["phase1"] = section("phase1", phase1_items(args.phase1_n),
                                    lambda it: run_phase1(client, it), summarize_phase1)
    if "2" in args.phases:
        cases = jsonl(PHASE2_BENCH)
        summary["phase2"] = section("phase2", cases, lambda c: run_phase2(client, c), summarize_phase2,
                                    label=lambda c: c.get("id"))
    if "3" in args.phases:
        gold, unanswerable = rag_items()
        for retriever in args.rag_retrievers:
            key = f"phase3_{retriever}"
            items = list(gold) + list(unanswerable)
            summary[key] = section(
                key, items, lambda it, r=retriever: run_phase3(client, it, r), summarize_phase3,
                label=lambda it: ("U-" + it[:30]) if isinstance(it, str) else it.id,
            )
        if "phase3_hybrid" in summary:
            summary["phase3"] = summary["phase3_hybrid"]
    if "4" in args.phases:
        tasks = jsonl(AGENT_BENCH)
        summary["phase4"] = {}
        for brain in args.brains:
            summary["phase4"][brain] = section(
                f"phase4_{brain}", tasks,
                lambda t, b=brain: run_phase4(client, t, b, args.agent_max_steps), summarize_phase4,
            )
        convs = card["rows"]["memory"] = [c for c in card["rows"].get("memory", []) if c.get("status") == 200]
        if not convs:
            for conv in CONVERSATIONS:
                convs.append(run_conversation(client, conv, "base"))
                save()
        summary["memory"] = {"n": len(convs), "correct": sum(c["correct"] for c in convs), **latency(convs)}
    save()
    print(markdown(card))


if __name__ == "__main__":
    main()
