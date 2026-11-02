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


def test_gate_rejects_three_verified_traces_from_two_problems(tmp_path):
    from rlm.validate_data import require_trace_quality

    train, _, _ = _splits(tmp_path)
    training = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    traces = tmp_path / "teacher_traces.jsonl"
    samples = []
    for n in range(6):
        example = training[n // 3]
        verified = n < 3
        samples.append({
            "question": example["question"], "answer": example["answer"],
            "sample_idx": n % 3, "verified": verified,
            "trace": f"<think>Razono.</think><answer>{example['answer']}</answer>" if verified else None,
            "format_status": "clean" if verified else "unclosed_think",
            "hit_max_tokens": not verified,
        })
    traces.write_text("\n".join(json.dumps(r) for r in samples) + "\n", encoding="utf-8")
    report = validate_traces(traces, train)
    assert report["verified"] == 3
    assert report["verified_unique_problems"] == 1
    assert report["hit_max_tokens"] == 3
    assert report["rejected_by_reason"]["unclosed_think"] == 3
    with pytest.raises(ValueError, match="Insufficient SFT data"):
        require_trace_quality(report, min_verified=6, min_unique_problems=4, min_families=3)


def test_gate_saves_failure_report_if_no_teacher_trace_survives(tmp_path):
    from rlm.validate_data import main as validate_main

    train, _, _ = _splits(tmp_path)
    example = json.loads(train.read_text(encoding="utf-8").splitlines()[0])
    traces = tmp_path / "no_success.jsonl"
    report = tmp_path / "report.json"
    traces.write_text(
        json.dumps({"question": example["question"], "answer": example["answer"],
                    "sample_idx": 0, "verified": False, "trace": None,
                    "format_status": "unclosed_think", "hit_max_tokens": True}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Insufficient SFT data"):
        validate_main(["--train", str(train), "--traces", str(traces),
                       "--out", str(report), "--min-verified-traces", "1"])
    data = json.loads(report.read_text(encoding="utf-8"))["distillation"]
    assert data["verified"] == 0 and data["rejected_by_reason"] == {"unclosed_think": 1}


def test_traces_cannot_silently_drop_problems_or_sample_indices(tmp_path):
    train, _, _ = _splits(tmp_path)
    row = json.loads(train.read_text(encoding="utf-8").splitlines()[0])
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps({
            "question": row["question"], "answer": row["answer"], "sample_idx": 0,
            "verified": False, "format_status": "unclosed_think",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing .* teacher samples"):
        validate_traces(trace_path, train, expected_samples=2)
