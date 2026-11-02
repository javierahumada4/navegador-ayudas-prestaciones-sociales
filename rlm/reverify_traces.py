"""Reverify legacy teacher traces rejected only for their language (no GPU).

The output is a new JSONL file. The existing run's original traces remain
untouched until the operator explicitly backs them up and replaces them.
Only original 'non_spanish_reasoning' rejects can be promoted and only after
checking BOTH the canonical training target and the actual trace's format
and monetary answer against the verifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.rewards import has_valid_imv_format
from rlm.validate_data import read_jsonl
from rlm.verifier import IMVAmountVerifier


def reverify_language_rejections(train_path: str | Path, input_path: str | Path,
                                 output_path: str | Path) -> dict:
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Use a different --output so the original traces remain intact")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    training = read_jsonl(train_path)
    expected = {row["question"]: row["answer"] for row in training}
    verifier = IMVAmountVerifier()
    rows = read_jsonl(source)
    promoted = 0
    unchanged = 0
    for index, row in enumerate(rows, start=1):
        question = row.get("question")
        if question not in expected or row.get("answer") != expected[question]:
            raise ValueError(f"Trace {index}: unknown training question or target mismatch")
        if row.get("format_status") == "non_spanish_reasoning" and row.get("verified") is False:
            trace = row.get("trace")
            if (isinstance(trace, str) and has_valid_imv_format(trace)
                    and verifier.verify(trace, expected[question]).is_correct):
                row["verified"] = True
                row["format_status"] = "reverified_no_language_filter"
                promoted += 1
            else:
                unchanged += 1
        else:
            unchanged += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "total": len(rows), "promoted": promoted, "unchanged": unchanged,
        "output": str(destination),
    }


def reverify_unclosed_no_thinking_rejections(
    train_path: str | Path, input_path: str | Path, output_path: str | Path,
) -> dict:
    """Repair *only* no-thinking-prefilled rows with an actual final answer.

    A valid shape is necessary but insufficient: the oracle must confirm the
    total. The original JSONL remains immutable and bad amounts stay rejected.
    """
    from rlm.distill import _structure_trace
    from rlm.rewards import extract_answer

    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Use a different --output so the original traces remain intact")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    training = read_jsonl(train_path)
    expected = {row["question"]: row["answer"] for row in training}
    verifier = IMVAmountVerifier()
    rows = read_jsonl(source)
    promoted = 0
    repaired_but_incorrect = 0
    unchanged = 0
    for index, row in enumerate(rows, start=1):
        question = row.get("question")
        if question not in expected or row.get("answer") != expected[question]:
            raise ValueError(f"Trace {index}: unknown training question or target mismatch")
        if (row.get("format_status") == "unclosed_think"
                and row.get("verified") is False
                and row.get("thinking") is False
                and row.get("trace") is None):
            raw = row.get("raw")
            if not isinstance(raw, str):
                unchanged += 1
                continue
            trace, status = _structure_trace(raw, allow_prefilled_think=True)
            if trace is None or not has_valid_imv_format(trace):
                unchanged += 1
                continue
            correct = verifier.verify(trace, expected[question]).is_correct
            row["trace"] = trace
            row["predicted"] = extract_answer(trace)
            row["verified"] = correct
            row["format_status"] = (
                "reverified_no_thinking" if correct else "incorrect_amount"
            )
            if correct:
                promoted += 1
            else:
                repaired_but_incorrect += 1
        else:
            unchanged += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "total": len(rows), "promoted": promoted,
        "repaired_but_incorrect": repaired_but_incorrect,
        "unchanged": unchanged, "output": str(destination),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--repair-no-thinking", action="store_true",
        help="repair missing </think> only for teacher-no-thinking final answers",
    )
    args = parser.parse_args(argv)
    action = (
        reverify_unclosed_no_thinking_rejections
        if args.repair_no_thinking else reverify_language_rejections
    )
    print(json.dumps(
        action(args.train, args.input, args.output), indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
