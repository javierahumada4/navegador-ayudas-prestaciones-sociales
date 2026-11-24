"""Phase 4 evaluation: success rate, steps and tokens on multi-tool tasks, per brain and agent.

Each line of the benchmark (``agent/bench/tasks.jsonl``, built by ``agent/bench/build_tasks.py``)
has a task, the tools that must be involved, and an automatic check of the final answer. Run every
brain with our JSON agent and with the smolagents code agent::

    uv run --extra train --extra rag --extra agent python -m agent.evaluate \\
        --bench agent/bench/tasks.jsonl --brains base rlm thinking --agents json code

Metrics per (agent, brain):

- ``success_rate``: chained tasks (T*) whose run finished and whose answer passes the check.
- ``impossible_ok``: impossible tasks (I*) where the agent said it could not, instead of inventing.
- ``required_tools_rate``: runs that used every tool the task needs (the chain really happened).
- ``mean_steps`` / ``mean_tokens`` / ``mean_wall_s``: cost of a run.

Ablations: ``--break-tool consultar_ayudas`` makes that tool raise ``ConnectionError`` on every
call (how does the agent react to a broken tool?), ``--only T03 T09`` restricts the tasks and
``--tag`` names the run (``json:base@broken``); the memory ablation is the same bench with a
smaller ``ARCA_AGENT_TOKEN_BUDGET``.

Results are written after every task (``--resume`` continues a cut run) as JSON plus a Markdown
table next to it.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

NUMBER = re.compile(r"-?\d[\d.,]*\d|-?\d")


def parse_number(token: str) -> float | None:
    """'733,60' / '26.409,60' / '733.6' / '1,234.5' / '8803.20' -> float (Spanish or English)."""
    token = token.strip(".,")
    if "," in token and "." in token:
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        token = token.replace(thousands, "").replace(decimal, ".")
    elif "," in token:
        head, _, tail = token.rpartition(",")
        token = f"{head.replace(',', '')}.{tail}" if len(tail) != 3 else token.replace(",", "")
    elif token.count(".") > 1 or ("." in token and len(token.rpartition(".")[2]) == 3):
        token = token.replace(".", "")  # 26.409 -> thousands separator
    try:
        return float(token)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    return [n for n in (parse_number(t) for t in NUMBER.findall(text)) if n is not None]


def check_answer(task: dict, final_answer: str, check: dict | None = None) -> bool:
    """Automatic check of the final answer, according to the task's ``check`` field."""
    check = task.get("check", {}) if check is None else check
    kind, value = check.get("type"), check.get("value")
    if kind == "all":
        return all(check_answer(task, final_answer, c) for c in check["checks"])
    if kind == "any":
        return any(check_answer(task, final_answer, c) for c in check["checks"])
    if kind == "contains":
        return str(value).lower() in final_answer.lower()
    if kind == "regex":
        return re.search(value, final_answer) is not None
    if kind == "number":
        tolerance = check.get("tolerance", 0)
        return any(abs(n - float(value)) <= tolerance + 1e-9 for n in numbers_in(final_answer))
    return False


def tools_used(steps: list[dict]) -> set[str]:
    return {s["action"]["name"] for s in steps if s.get("action")} - {"final_answer"}


def score(task: dict, response: dict) -> dict:
    used = set(response.get("tools_used") or tools_used(response["steps"]))
    return {
        "correct": bool(response["succeeded"]) and check_answer(task, response["final_answer"]),
        "tools_used": sorted(used),
        "required_tools_used": set(task.get("required_tools", [])) <= used,
    }


def summarize(rows: list[dict]) -> dict:
    chained = [r for r in rows if not r["id"].startswith("I")]
    impossible = [r for r in rows if r["id"].startswith("I")]
    n = max(len(rows), 1)
    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in chained:
        by_category[r["category"]].append(r["correct"])
    return {
        "n": len(rows),
        "success_rate": sum(r["correct"] for r in chained) / max(len(chained), 1),
        "impossible_ok": sum(r["correct"] for r in impossible) / max(len(impossible), 1),
        "required_tools_rate": sum(r["required_tools_used"] for r in chained) / max(len(chained), 1),
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
        "mean_wall_s": sum(r["wall_s"] for r in rows) / n,
        "by_category": {k: sum(v) / len(v) for k, v in sorted(by_category.items())},
    }


def markdown(results: dict) -> str:
    lines = [
        "| agente | cerebro | éxito (T) | imposibles OK | cadena completa | pasos | tokens | s/tarea |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, res in results.items():
        s = res["summary"]
        agent, brain = key.split(":")
        lines.append(
            f"| {agent} | {brain} | {s['success_rate']:.2f} | {s['impossible_ok']:.2f} | "
            f"{s['required_tools_rate']:.2f} | {s['mean_steps']:.1f} | {s['mean_tokens']:.0f} | "
            f"{s['mean_wall_s']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def break_tool(registry, name: str):
    """Replace ``name`` with a copy that always fails like a dead HTTP service."""
    from dataclasses import replace

    def broken(**_kwargs):
        raise ConnectionError(f"{name}: servicio no disponible (fallo simulado para la evaluación)")

    registry.tools[name] = replace(registry.tools[name], fn=broken)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["base", "rlm", "thinking"])
    parser.add_argument("--agents", nargs="+", default=["json"], choices=["json", "code"])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke runs)")
    parser.add_argument("--out", default="reports/phase4_eval.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="+", default=[], help="task ids to run")
    parser.add_argument("--break-tool", default="", help="tool that always fails")
    parser.add_argument("--tag", default="", help="suffix of the run name, e.g. @broken")
    args = parser.parse_args()

    tasks = [
        json.loads(line)
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.only:
        tasks = [t for t in tasks if t["id"] in set(args.only)]
    if args.limit:
        tasks = tasks[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if args.resume and out.exists() else {}

    json_agent = None
    for agent_kind in args.agents:
        for brain in args.brains:
            key = f"{agent_kind}:{brain}{args.tag}"
            done = {r["id"]: r for r in results.get(key, {}).get("rows", [])}
            rows = []
            for task in tasks:
                if task["id"] in done:
                    rows.append(done[task["id"]])
                    continue
                started = time.perf_counter()
                if agent_kind == "json":
                    if json_agent is None:
                        from agent.react_agent import ReActAgent

                        json_agent = ReActAgent.from_env()
                        if args.break_tool:
                            break_tool(json_agent.registry, args.break_tool)
                    response = json_agent.run(task["task"], max_steps=args.max_steps, brain=brain)
                    response = response.model_dump()
                else:
                    from agent.code_agent import run_code_agent

                    from agent.react_agent import agent_registry

                    def factory():
                        registry = agent_registry()
                        return break_tool(registry, args.break_tool) if args.break_tool else registry

                    response = run_code_agent(task["task"], max_steps=args.max_steps, brain=brain,
                                              registry_factory=factory)
                wall = round(time.perf_counter() - started, 1)
                row = {**task, **response, **score(task, response), "wall_s": wall}
                row.pop("check", None)
                rows.append(row)
                print(
                    f"{key:>14} {task['id']}: {'OK ' if row['correct'] else 'BAD'} "
                    f"steps={row['n_steps']} tokens={row['tokens_used']} {wall:.0f}s "
                    f"tools={','.join(row['tools_used'])} | {row['final_answer'][:120]!r}",
                    flush=True,
                )
                results[key] = {"summary": summarize(rows), "rows": rows}
                out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))
            results[key] = {"summary": summarize(rows), "rows": rows}
            s = results[key]["summary"]
            print(f"{key:>14}: success {s['success_rate']:.2f} · impossible {s['impossible_ok']:.2f} "
                  f"· steps {s['mean_steps']:.1f} · tokens {s['mean_tokens']:.0f}", flush=True)
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))
    out.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))


if __name__ == "__main__":
    main()
