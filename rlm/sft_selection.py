"""Select complete, short SFT targets without silently truncating teacher reasoning.

The limit applies to *assistant completion tokens* under the student's tokenizer,
not to the system prompt or to the teacher's generation budget.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from rlm.chat_template import render_supervised_exchange
from rlm.data import build_prompt, get_imv_rule_context
from rlm.rewards import has_valid_imv_format
from rlm.sft_preflight import _token_ids


def select_budgeted_traces(
    dataset, tokenizer, *, max_completion_tokens: int, max_length: int,
    max_per_problem: int, min_problem_coverage: float, min_families: int,
    report_path: str | Path, reserve_tokens: int = 8,
):
    """Select shortest valid traces for each question, then enforce actual coverage.

    Each trace must be *verified*, valid, and fit both the completion budget and
    the full chat context. We never crop a trace or manufacture an answer.
    A coverage failure writes its report before blocking SFT.
    """
    if max_completion_tokens <= 0 or max_per_problem <= 0:
        raise ValueError("SFT completion budget and traces per problem must be positive")
    if max_length <= max_completion_tokens or reserve_tokens < 0:
        raise ValueError("SFT max_length must accommodate prompt and completion")
    if not 0 <= min_problem_coverage <= 1 or min_families <= 0:
        raise ValueError("SFT coverage must be in [0, 1] and min_families positive")
    if "verified" not in dataset.column_names:
        raise ValueError("SFT teacher data must contain verified status")

    questions = {row["question"] for row in dataset}
    families = {
        row["question"]: (row.get("branches") or {}).get("family") or "unknown"
        for row in dataset
    }
    candidate_indices = defaultdict(list)
    excluded = Counter()
    for index, row in enumerate(dataset):
        if row["verified"] is not True:
            continue
        trace = row.get("trace")
        if not isinstance(trace, str) or not has_valid_imv_format(trace):
            excluded["invalid_trace"] += 1
            continue
        prompt = build_prompt(
            row["question"],
            rule_context=row.get("rule_context") or get_imv_rule_context(),
        )
        try:
            prefix, full = render_supervised_exchange(tokenizer, prompt, trace)
            prefix_ids, full_ids = _token_ids(tokenizer, prefix), _token_ids(tokenizer, full)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"SFT tokenizer/chat template failed at trace {index}: {exc}") from exc
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"SFT tokenization prefix mismatch at trace {index}")
        completion_tokens = len(full_ids) - len(prefix_ids)
        if completion_tokens <= 0:
            excluded["empty_completion"] += 1
        elif completion_tokens > max_completion_tokens:
            excluded["over_completion_budget"] += 1
        elif len(full_ids) + reserve_tokens > max_length:
            excluded["over_context_budget"] += 1
        else:
            candidate_indices[row["question"]].append((completion_tokens, index, trace))

    chosen = []
    covered = set()
    duplicates = 0
    for question, entries in candidate_indices.items():
        seen = set()
        for token_count, index, trace in sorted(entries):
            if trace in seen:
                duplicates += 1
                continue
            seen.add(trace)
            chosen.append(index)
            covered.add(question)
            if len(seen) == max_per_problem:
                break
    total_families = set(families.values())
    covered_families = {families[q] for q in covered}
    coverage = len(covered) / max(1, len(questions))
    report = {
        "max_completion_tokens": max_completion_tokens,
        "max_sequence_length": max_length,
        "total_problems": len(questions),
        "covered_problems": len(covered),
        "problem_coverage": round(coverage, 6),
        "minimum_problem_coverage": min_problem_coverage,
        "total_families": len(total_families),
        "covered_families": len(covered_families),
        "minimum_families": min_families,
        "traces_selected": len(chosen),
        "verified_traces_available": sum(row["verified"] is True for row in dataset),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "duplicate_traces_skipped": duplicates,
        "by_family": {
            family: {
                "problems": sum(families[q] == family for q in questions),
                "covered": sum(families[q] == family for q in covered),
            }
            for family in sorted(total_families)
        },
    }
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not chosen or coverage < min_problem_coverage or len(covered_families) < min_families:
        raise ValueError(
            "Insufficient SFT-eligible coverage: "
            f"{len(covered)}/{len(questions)} problems ({coverage:.1%}; "
            f"minimum {min_problem_coverage:.1%}), "
            f"{len(covered_families)} families (minimum {min_families}). "
            f"See {out}. Increase teacher samples or improve concise reasoning "
            "and start a new run; never lower the token cap or truncate targets."
        )
    return dataset.select(sorted(chosen)), report
