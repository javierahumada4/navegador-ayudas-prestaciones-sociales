"""Validate fresh IMV/CAPI data and teacher traces before training (stdlib only)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from rlm.imv_engine import expected_answer
from rlm.rewards import has_valid_imv_format
from rlm.verifier import IMVAmountVerifier


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing JSONL file: {path}")
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{i}: expected JSON object")
        records.append(row)
    if not records:
        raise ValueError(f"Empty JSONL: {path}")
    return records


def validate_case_dates(params, place="case"):
    """Reject chronologically impossible synthetic case facts before oracle checks."""
    try:
        application = date.fromisoformat(params["application_date"])
        residents = params["persons"]
        if not residents:
            raise ValueError("empty household")
        births = []
        for person in residents:
            born = date.fromisoformat(person["date_of_birth"])
            residence = date.fromisoformat(
                person["continuous_legal_effective_residence_since"]
            )
            if not born <= residence <= application:
                raise ValueError(
                    f"{person['id']}: residence start {residence} outside "
                    f"birth {born} .. application {application}"
                )
            births.append(born)
        together = date.fromisoformat(params["domicile"]["same_domicile_since"])
        if not max(births) <= together <= application:
            raise ValueError(
                f"household co-residence {together} outside "
                f"youngest birth {max(births)} .. application {application}"
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{place}: invalid chronology: {exc}") from exc


def validate_splits(train, test, ood, extra_ood=None):
    from rlm.generate_problems import OOD_FAMILIES, case_fingerprint

    seen_params, seen_questions, summary = {}, {}, {}
    paths = {"train": train, "test": test, "ood": ood}
    for split, path in (extra_ood or {}).items():
        if split not in OOD_FAMILIES or split == "ood":
            raise ValueError(f"unknown or duplicate OOD split: {split}")
        paths[split] = path
    for split, path in paths.items():
        rows = read_jsonl(path)
        families = Counter()
        for n, row in enumerate(rows, 1):
            place = f"{path}:{n}"
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"{place}: missing question")
            if not isinstance(row.get("params"), dict):
                raise ValueError(f"{place}: missing params")
            if row.get("split") != split:
                raise ValueError(f"{place}: split mismatch")
            validate_case_dates(row["params"], place)
            branch = row.get("branches")
            if not isinstance(branch, dict) or not isinstance(branch.get("family"), str):
                raise ValueError(f"{place}: missing family")
            family = branch["family"]
            if split in OOD_FAMILIES:
                if family != OOD_FAMILIES[split]:
                    raise ValueError(f"{place}: wrong family for split {split}: {family}")
            elif family in OOD_FAMILIES.values():
                raise ValueError(f"{place}: OOD family {family} leaked into {split}")
            label = expected_answer(row["params"])
            if row.get("answer") != label:
                raise ValueError(f"{place}: stale oracle label {row.get('answer')!r} != {label!r}")
            fingerprint = case_fingerprint(row["params"])
            if fingerprint in seen_params:
                raise ValueError(f"{place}: duplicate parameters with {seen_params[fingerprint]}")
            seen_params[fingerprint] = place
            question = row["question"].strip()
            if question in seen_questions:
                raise ValueError(f"{place}: duplicate question with {seen_questions[question]}")
            seen_questions[question] = place
            families[family] += 1
        summary[split] = {"count": len(rows), "families": dict(families)}
    return summary


def validate_traces(path, train, expected_samples=None):
    training = read_jsonl(train)
    answers = {x["question"]: x["answer"] for x in training}
    family = {x["question"]: x["branches"]["family"] for x in training}
    verifier = IMVAmountVerifier()
    counts, accepted, seen = Counter(), Counter(), set()
    accepted_questions = set()
    rejection_reasons = Counter()
    max_token_count = 0
    rows = read_jsonl(path)
    for n, row in enumerate(rows, 1):
        place = f"{path}:{n}"
        question = row.get("question")
        if question not in answers or row.get("answer") != answers[question]:
            raise ValueError(f"{place}: not a training question or target mismatch")
        index = row.get("sample_idx")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"{place}: invalid sample_idx")
        if expected_samples is not None and index >= expected_samples:
            raise ValueError(f"{place}: sample_idx exceeds expected samples")
        key = (question, index)
        if key in seen:
            raise ValueError(f"{place}: repeated sample index")
        seen.add(key)
        bucket = family[question]
        counts[bucket] += 1
        if row.get("verified") is True:
            trace = row.get("trace")
            if not isinstance(trace, str) or not has_valid_imv_format(trace):
                raise ValueError(f"{place}: malformed verified trace")
            if not verifier.verify(trace, answers[question]).is_correct:
                raise ValueError(f"{place}: verified trace disagrees with oracle")
            accepted[bucket] += 1
            accepted_questions.add(question)
        else:
            reason = (
                "incorrect_amount" if row.get("trace") is not None
                else row.get("format_status") or "unknown_rejection"
            )
            rejection_reasons[reason] += 1
        if row.get("hit_max_tokens"):
            max_token_count += 1
    if expected_samples is not None:
        missing = [
            (question, index)
            for question in answers
            for index in range(expected_samples)
            if (question, index) not in seen
        ]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} teacher samples; first missing: {missing[0]}"
            )
    kept = sum(accepted.values())
    return {
        "generated": len(rows), "verified": kept,
        "verified_unique_problems": len(accepted_questions),
        "verified_families": sum(bool(n) for n in accepted.values()),
        "rejected_by_reason": dict(rejection_reasons),
        "hit_max_tokens": max_token_count,
        "acceptance_rate": round(kept / len(rows), 6),
        "by_family": {
            name: {"generated": count, "verified": accepted[name],
                   "acceptance_rate": round(accepted[name] / count, 6)}
            for name, count in sorted(counts.items())
        },
    }


def require_trace_quality(report, *, min_verified=1, min_unique_problems=1,
                          min_families=1):
    """Prevent SFT on a handful of verified but unrepresentative teacher traces."""
    requirements = (
        ("verified traces", report["verified"], min_verified),
        ("unique verified problems", report["verified_unique_problems"], min_unique_problems),
        ("verified families", report["verified_families"], min_families),
    )
    failed = [
        f"{label}: {actual} < required {minimum}"
        for label, actual, minimum in requirements if actual < minimum
    ]
    if failed:
        raise ValueError(
            "Insufficient SFT data; training aborted ("
            + "; ".join(failed)
            + "). Inspect rejected_by_reason and hit_max_tokens; "
            "increase the teacher token budget or improve the teacher."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test")
    parser.add_argument("--ood")
    parser.add_argument(
        "--ood-extra", nargs=2, action="append", metavar=("SPLIT", "PATH"),
        help="additional held-out suites: ood_threshold, ood_composition, ood_wording",
    )
    parser.add_argument("--traces")
    parser.add_argument("--expected-trace-samples", type=int)
    parser.add_argument("--min-verified-traces", type=int, default=1)
    parser.add_argument("--min-unique-trace-problems", type=int, default=1)
    parser.add_argument("--min-verified-families", type=int, default=1)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if bool(args.test) != bool(args.ood):
        parser.error("--test and --ood must be provided together")
    if not args.test and not args.traces:
        parser.error("provide --test and --ood and/or --traces")
    summary = {}
    if args.test:
        extras = dict(args.ood_extra or [])
        if len(extras) != len(args.ood_extra or []):
            parser.error("duplicate --ood-extra split")
        summary["splits"] = validate_splits(
            args.train, args.test, args.ood, extra_ood=extras
        )
    if args.traces:
        summary["distillation"] = validate_traces(
            args.traces, args.train, expected_samples=args.expected_trace_samples
        )
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    if args.traces:
        require_trace_quality(
            summary["distillation"],
            min_verified=args.min_verified_traces,
            min_unique_problems=args.min_unique_trace_problems,
            min_families=args.min_verified_families,
        )


if __name__ == "__main__":
    main()
