"""Distillation must be recoverable without rerunning completed teacher samples."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")

from rlm.distill import _verified_resume_offset  # noqa: E402


def _row(example, sample, answer="100.00", teacher="teacher"):
    return {
        "example_id": example,
        "sample_idx": sample,
        "question": f"Question {example}",
        "answer": answer,
        "teacher": teacher,
        "thinking": True,
        "verified": True,
        "trace": "<think>Calculation</think><answer>100.00</answer>",
    }


def test_resume_drops_only_incomplete_trailing_example(tmp_path):
    data = [
        {"prompt": [{"role": "user", "content": f"Question {n}"}],
         "answer": "100.00"}
        for n in range(2)
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "".join(json.dumps(_row(0, k)) + "\n" for k in range(2))
        + json.dumps(_row(1, 0)) + "\n" + '{"example_id":',
        encoding="utf-8",
    )
    assert _verified_resume_offset(path, data, 2, 0, "teacher") == 1
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {row["example_id"] for row in records} == {0}


def test_resume_rejects_different_teacher(tmp_path):
    data = [{"prompt": [{"role": "user", "content": "Question 0"}],
             "answer": "100.00"}]
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(_row(0, 0)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _verified_resume_offset(path, data, 1, 0, "other")
