"""Validate fresh IMV/CAPI data and teacher traces before training (stdlib only)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rlm.imv_engine import expected_answer
from rlm.rewards import has_valid_format
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


def validate_splits(train, test, ood):
    seen_params, seen_questions, summary = {}, {}, {}
    for split, path in {"train": train, "test": test, "ood": ood}.items():
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
            branch = row.get("branches")
            if not isinstance(branch, dict) or not isinstance(branch.get("family"), str):
                raise ValueError(f"{place}: missing family")
            family = branch["family"]
            if (split == "ood") != (family == "capi_only"):
                raise ValueError(f"{place}: capi_only must be held out in OOD")
            label = expected_answer(row["params"])
            if row.get("answer") != label:
                raise ValueError(f"{place}: stale oracle label {row.get('answer')!r} != {label!r}")
            fingerprint = json.dumps(row["params"], sort_keys=True, ensure_ascii=False)
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


def validate_traces(path, train):
    training = read_jsonl(train)
    answers = {x["question"]: x["answer"] for x in training}
    family = {x["question"]: x["branches"]["family"] for x in training}
    verifier = IMVAmountVerifier()
    counts, accepted, seen = Counter(), Counter(), set()
    rows = read_jsonl(path)
    for n, row in enumerate(rows, 1):
        place = f"{path}:{n}"
        question = row.get("question")
        if question not in answers or row.get("answer") != answers[question]:
            raise ValueError(f"{place}: not a training question or target mismatch")
        key = (question, row.get("sample_idx"))
        if key in seen:
            raise ValueError(f"{place}: repeated sample index")
        seen.add(key)
        bucket = family[question]
        counts[bucket] += 1
        if row.get("verified") is True:
            trace = row.get("trace")
            if not isinstance(trace, str) or not has_valid_format(trace):
                raise ValueError(f"{place}: malformed verified trace")
            if not verifier.verify(trace, answers[question]).is_correct:
                raise ValueError(f"{place}: verified trace disagrees with oracle")
            accepted[bucket] += 1
    kept = sum(accepted.values())
    if not kept:
        raise ValueError("No verified teacher traces: SFT must not start")
    return {
        "generated": len(rows), "verified": kept,
        "acceptance_rate": round(kept / len(rows), 6),
        "by_family": {
            name: {"generated": count, "verified": accepted[name],
                   "acceptance_rate": round(accepted[name] / count, 6)}
            for name, count in sorted(counts.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test")
    parser.add_argument("--ood")
    parser.add_argument("--traces")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if bool(args.test) != bool(args.ood):
        parser.error("--test and --ood must be provided together")
    if not args.test and not args.traces:
        parser.error("provide --test and --ood and/or --traces")
    summary = {}
    if args.test:
        summary["splits"] = validate_splits(args.train, args.test, args.ood)
    if args.traces:
        summary["distillation"] = validate_traces(args.traces, args.train)
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
