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

from rlm.rewards import has_valid_format
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
            if (isinstance(trace, str) and has_valid_format(trace)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(reverify_language_rejections(
        args.train, args.input, args.output
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
