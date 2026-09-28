"""Evaluate base vs SFT vs GRPO on held-out IMV problems.

Every model sees the same problems with the same decoding settings. Sampling is the
default: greedy decoding makes small reasoning models loop until ``--max-new-tokens``.
With ``--samples k`` each problem is answered k times and pass@1 is the mean accuracy
over all samples, which is much less noisy than a single draw on a small test set.

For IMV data every wrong answer is diagnosed against the reference engine (missed CAPI,
wrongly denied, forgot to subtract income...), and a handful of diverse failures is
selected for the manual analysis that goes into EXPERIMENTS.md.

    uv run python -m rlm.evaluate --data rlm/data/test.jsonl \\
        --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/grpo_lora \\
        --max-new-tokens 4096 --out reports/phase1_eval.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from rlm.data import load_gsm8k
from rlm.imv_engine import evaluate_imv_case
from rlm.rewards import has_valid_format, imv_money, parse_imv_decimal, thinking_length
from rlm.verifier import Verifier, build_verifier

# Answers within this distance of the expected amount are counted as near misses.
NEAR_MISS_EUR = Decimal("1.00")


def load_problems(data: str, n_examples: int | None) -> list[dict]:
    """Return problems as plain dicts: question, answer, and for IMV also branches/params.

    IMV JSONL is read directly instead of through ``datasets`` so the nested ``params``
    reach the reference engine exactly as the generator wrote them.
    """
    if data == "gsm8k":
        dataset = load_gsm8k("test", n_examples=n_examples or 200)
        return [
            {"id": i, "question": ex["prompt"][-1]["content"], "answer": str(ex["answer"])}
            for i, ex in enumerate(dataset)
        ]
    problems = []
    with open(data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            problems.append(
                {
                    "id": i,
                    "question": ex["question"],
                    "answer": str(ex["answer"]),
                    "branches": ex.get("branches"),
                    "params": ex.get("params"),
                }
            )
    return problems[:n_examples] if n_examples else problems


def diagnose_failure(row: dict, params: dict | None) -> tuple[str, dict]:
    """Classify a wrong answer. Returns (category, details for the report)."""
    if row["predicted"] is None:
        return ("truncated" if row["truncated"] else "no_answer"), {}
    predicted = parse_imv_decimal(row["predicted"])
    if predicted is None:
        return "unparseable", {}
    predicted = imv_money(predicted)
    if params is None:
        return "wrong_amount", {}

    oracle = evaluate_imv_case(params)
    expected = imv_money(Decimal(str(oracle["total_monthly_eur"])))
    imv = imv_money(Decimal(str(oracle["imv_monthly_eur"])))
    capi = imv_money(Decimal(str(oracle["capi_monthly_eur"])))
    guaranteed = imv_money(Decimal(str(oracle["guaranteed_income_monthly_eur"])))
    details = {
        "oracle_imv": str(imv),
        "oracle_capi": str(capi),
        "oracle_guaranteed_income": str(guaranteed),
        "failed_requirements": oracle["failed_requirements"],
    }
    if expected == 0:
        return "missed_ineligibility", details
    if predicted == 0:
        return "wrongly_denied", details
    if capi > 0 and predicted == imv:
        return "missed_capi", details
    if imv > 0 and predicted == capi:
        return "missed_imv", details
    if predicted == guaranteed and guaranteed != expected:
        return "income_not_subtracted", details
    if abs(predicted - expected) <= NEAR_MISS_EUR:
        return "near_miss", details
    return "wrong_amount", details


def evaluate_model(
    base_model: str,
    adapter: str | None,
    problems: list[dict],
    verifier: Verifier,
    *,
    max_new_tokens: int,
    samples: int,
    do_sample: bool,
    temperature: float,
    batch_size: int,
    seed: int,
) -> list[dict]:
    import torch

    from rlm.inference import ReasoningModel

    model = ReasoningModel(base_model=base_model, adapter_path=adapter, verifier_name=verifier.name)
    model.load()
    rows: list[dict] = []
    for sample_idx in range(samples):
        # Same seed per sample index for every model: reproducible and comparable.
        torch.manual_seed(seed + sample_idx)
        for start in range(0, len(problems), batch_size):
            batch = problems[start : start + batch_size]
            outputs = model.generate_batch(
                [p["question"] for p in batch],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            for problem, (raw, n_tokens) in zip(batch, outputs, strict=True):
                rows.append(score_row(problem, raw, n_tokens, verifier, max_new_tokens, sample_idx))
            done = sample_idx * len(problems) + start + len(batch)
            print(f"  {done}/{samples * len(problems)} generations", flush=True)
    return rows


def score_row(
    problem: dict,
    raw: str,
    n_tokens: int,
    verifier: Verifier,
    max_new_tokens: int,
    sample_idx: int = 0,
) -> dict:
    verdict = verifier.verify(raw, problem["answer"])
    branches = problem.get("branches") or {}
    row = {
        "id": problem["id"],
        "sample": sample_idx,
        "family": branches.get("family"),
        "question": problem["question"],
        "expected": problem["answer"],
        "raw": raw,
        "predicted": verdict.predicted,
        "is_correct": verdict.is_correct,
        "has_valid_format": has_valid_format(raw),
        "n_tokens": n_tokens,
        "thinking_words": thinking_length(raw),
        "truncated": n_tokens >= max_new_tokens,
        "failure": None,
        "failure_details": {},
    }
    if not row["is_correct"]:
        row["failure"], row["failure_details"] = diagnose_failure(row, problem.get("params"))
    return row


def _rate(rows: list[dict], predicate) -> float:
    return sum(1 for r in rows if predicate(r)) / max(len(rows), 1)


def _is_zero(text: str | None) -> bool:
    value = parse_imv_decimal(text)
    return value is not None and value == 0


def summarize(rows: list[dict]) -> dict:
    """Aggregate metrics for one model. pass@1 is the mean accuracy over all samples."""
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["family"] or "all"].append(r)
    return {
        "n_generations": len(rows),
        "n_problems": len({r["id"] for r in rows}),
        "pass@1": _rate(rows, lambda r: r["is_correct"]),
        "format_rate": _rate(rows, lambda r: r["has_valid_format"]),
        "truncated_rate": _rate(rows, lambda r: r["truncated"]),
        # Compare with expected_zero_rate: far above it means the model learnt to answer 0.
        "zero_answer_rate": _rate(rows, lambda r: _is_zero(r["predicted"])),
        "expected_zero_rate": _rate(rows, lambda r: _is_zero(r["expected"])),
        "mean_tokens": sum(r["n_tokens"] for r in rows) / max(len(rows), 1),
        "mean_thinking_words": sum(r["thinking_words"] for r in rows) / max(len(rows), 1),
        "pass@1_by_family": {
            family: {"n": len(rs), "pass@1": _rate(rs, lambda r: r["is_correct"])}
            for family, rs in sorted(by_family.items())
        },
        "failures": dict(Counter(r["failure"] for r in rows if r["failure"]).most_common()),
    }


def select_failures(rows: list[dict], n: int = 5) -> list[dict]:
    """Pick up to n failures covering as many (category, family) pairs as possible."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["failure"]:
            groups[(r["failure"], r["family"])].append(r)
    # Most frequent failure modes first, then round-robin so no single mode dominates.
    ordered = sorted(groups.values(), key=len, reverse=True)
    picked: list[dict] = []
    depth = 0
    while len(picked) < n and any(depth < len(g) for g in ordered):
        for group in ordered:
            if depth < len(group) and len(picked) < n:
                picked.append(group[depth])
        depth += 1
    return picked


def render_markdown(results: dict) -> str:
    """Tables ready to paste into EXPERIMENTS.md."""
    models = results["models"]
    lines = [
        "| Modelo | pass@1 | Formato | Truncadas | Responde 0 | Tokens medios |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        s = m["summary"]
        lines.append(
            f"| {name} | {s['pass@1']:.3f} | {s['format_rate']:.3f} | {s['truncated_rate']:.3f} "
            f"| {s['zero_answer_rate']:.3f} (esperado {s['expected_zero_rate']:.3f}) "
            f"| {s['mean_tokens']:.0f} |"
        )
    families = sorted({f for m in models.values() for f in m["summary"]["pass@1_by_family"]})
    lines += ["", "| Familia | Generaciones | " + " | ".join(models) + " |"]
    lines.append("|---|---|" + "---|" * len(models))
    for family in families:
        cells = []
        n = ""
        for m in models.values():
            stats = m["summary"]["pass@1_by_family"].get(family)
            cells.append(f"{stats['pass@1']:.3f}" if stats else "—")
            n = n or (str(stats["n"]) if stats else "")
        lines.append(f"| {family} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "| Fallo | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    categories = sorted({c for m in models.values() for c in m["summary"]["failures"]})
    for category in categories:
        counts = [str(m["summary"]["failures"].get(category, 0)) for m in models.values()]
        lines.append(f"| {category} | " + " | ".join(counts) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data", default="rlm/data/test.jsonl", help="'gsm8k' or a JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["base=none"],
        help="name=path pairs; use 'none' for the bare base model",
    )
    parser.add_argument("--verifier", default="imv")
    parser.add_argument("--n-examples", type=int, default=None, help="first N problems only")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=1, help="generations per problem")
    parser.add_argument("--greedy", action="store_true", help="greedy decoding (may loop)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-failures", type=int, default=5, help="failures kept for analysis")
    parser.add_argument(
        "--resume", action="store_true", help="skip models already present in --out"
    )
    parser.add_argument("--out", default="reports/phase1_eval.json")
    args = parser.parse_args()

    problems = load_problems(args.data, args.n_examples)
    verifier = build_verifier("numeric" if args.data == "gsm8k" else args.verifier)
    config = {
        k: getattr(args, k)
        for k in (
            "data",
            "model",
            "n_examples",
            "max_new_tokens",
            "samples",
            "greedy",
            "temperature",
            "seed",
        )
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "models": {}}
    if args.resume and out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if previous.get("config") != config:
            raise SystemExit(f"--resume: {out} was produced with a different configuration")
        results = previous

    print(f"{len(problems)} problems x {args.samples} samples from {args.data}")
    for pair in args.adapters:
        name, path = pair.split("=", 1)
        if name in results["models"]:
            print(f"{name:>8}: already in {out}, skipped")
            continue
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            problems,
            verifier,
            max_new_tokens=args.max_new_tokens,
            samples=args.samples,
            do_sample=not args.greedy,
            temperature=args.temperature,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        summary = summarize(rows)
        results["models"][name] = {
            "adapter": path,
            "summary": summary,
            "failures_to_analyse": select_failures(rows, args.n_failures),
            "rows": rows,
        }
        # Save after every model so a dead session does not lose the finished ones.
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{name:>8}: pass@1 = {summary['pass@1']:.3f}  format = {summary['format_rate']:.3f}  "
            f"truncated = {summary['truncated_rate']:.3f}  zero = {summary['zero_answer_rate']:.3f}"
        )

    table = out.with_suffix(".md")
    table.write_text(render_markdown(results), encoding="utf-8")
    print(f"details -> {out}\ntables  -> {table}")


if __name__ == "__main__":
    main()
