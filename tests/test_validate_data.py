"""No GPU: check dataset separation, labels, and verified teacher traces."""
import json
from dataclasses import asdict

import pytest

from rlm.generate_problems import IMVProblemGenerator
from rlm.validate_data import validate_splits, validate_traces


def _write(path, problems, split):
    with path.open("w", encoding="utf-8") as file:
        for problem in problems:
            row = asdict(problem)
            row["split"] = split
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _splits(tmp_path):
    generator = IMVProblemGenerator()
    train, test, ood = (tmp_path / name for name in
                        ("train.jsonl", "test.jsonl", "test_ood.jsonl"))
    for path, split, n in ((train, "train", 12), (test, "test", 6), (ood, "ood", 6)):
        _write(path, generator.generate(n, split, seed=19), split)
    return train, test, ood


def test_fresh_splits_are_valid(tmp_path):
    train, test, ood = _splits(tmp_path)
    report = validate_splits(train, test, ood)
    assert (report["train"]["count"], report["test"]["count"],
            report["ood"]["count"]) == (12, 6, 6)


def test_tampered_answer_fails(tmp_path):
    train, test, ood = _splits(tmp_path)
    rows = [json.loads(line) for line in test.read_text(encoding="utf-8").splitlines()]
    rows[0]["answer"] = "999999.99"
    test.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale oracle label"):
        validate_splits(train, test, ood)


def test_cross_split_leakage_fails(tmp_path):
    train, test, ood = _splits(tmp_path)
    row = json.loads(train.read_text(encoding="utf-8").splitlines()[0])
    row["split"] = "test"
    with test.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate parameters"):
        validate_splits(train, test, ood)


def test_verified_teacher_trace_uses_training_label(tmp_path):
    train, _, _ = _splits(tmp_path)
    row = json.loads(train.read_text(encoding="utf-8").splitlines()[0])
    trace = f"<think>Aplico las reglas.</think><answer>{row['answer']}</answer>"
    path = tmp_path / "sft_traces.jsonl"
    path.write_text(json.dumps({
        "question": row["question"], "answer": row["answer"],
        "sample_idx": 0, "verified": True, "trace": trace
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    report = validate_traces(path, train)
    assert report["verified"] == 1
    assert report["acceptance_rate"] == 1.0


def test_residence_before_birth_fails_validation(tmp_path):
    train, test, ood = _splits(tmp_path)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    youngest = min(rows[0]["params"]["persons"], key=lambda p: p["date_of_birth"])
    youngest["continuous_legal_effective_residence_since"] = "1900-01-01"
    train.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid chronology"):
        validate_splits(train, test, ood)


def test_household_cannot_predate_youngest_member(tmp_path):
    train, test, ood = _splits(tmp_path)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    rows[0]["params"]["domicile"]["same_domicile_since"] = "1900-01-01"
    train.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid chronology"):
        validate_splits(train, test, ood)
