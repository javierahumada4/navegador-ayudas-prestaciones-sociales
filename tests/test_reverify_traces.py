"""Old non-Spanish rejects can be recovered without distilling again."""
import json

import pytest

from rlm.reverify_traces import reverify_language_rejections


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_reverify_english_reject_only_if_correct(tmp_path):
    train, source, output = (tmp_path / n for n in ("train.jsonl", "traces.jsonl", "recovered.jsonl"))
    _write(train, [{"question": "Caso A", "answer": "35.50"}])
    english = " ".join(["we check the income and the monthly amount"] * 4)
    rows = [
        {"question": "Caso A", "answer": "35.50", "sample_idx": 0,
         "verified": False, "format_status": "non_spanish_reasoning",
         "trace": f"<think>{english}</think><answer>35.50</answer>"},
        {"question": "Caso A", "answer": "35.50", "sample_idx": 1,
         "verified": False, "format_status": "incorrect_amount",
         "trace": "<think>R.</think><answer>9.00</answer>"},
        {"question": "Caso A", "answer": "35.50", "sample_idx": 2,
         "verified": False, "format_status": "non_spanish_reasoning",
         "trace": "<think>English.</think><answer>9.00</answer>"},
        {"question": "Caso A", "answer": "35.50", "sample_idx": 3,
         "verified": False, "format_status": "non_spanish_reasoning",
         "trace": "<think>English</think>"},
    ]
    _write(source, rows)
    stats = reverify_language_rejections(train, source, output)
    recovered = [json.loads(s) for s in output.read_text(encoding="utf-8").splitlines()]
    assert stats["promoted"] == 1
    assert stats["unchanged"] == 3
    assert recovered[0]["verified"] is True
    assert recovered[0]["format_status"] == "reverified_no_language_filter"
    assert all(not r["verified"] for r in recovered[1:])
    assert [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()] == rows
    with pytest.raises(FileExistsError):
        reverify_language_rejections(train, source, output)


def test_reverify_never_promotes_unknown_training_target(tmp_path):
    train, source, out = (tmp_path / n for n in ("train", "traces", "out"))
    _write(train, [{"question": "Caso A", "answer": "35.50"}])
    _write(source, [{
        "question": "Caso A", "answer": "35.51", "verified": False,
        "format_status": "non_spanish_reasoning",
        "trace": "<think>English</think><answer>35.51</answer>"
    }])
    with pytest.raises(ValueError, match="target mismatch"):
        reverify_language_rejections(train, source, out)
    assert not out.exists()


def test_reverify_refuses_in_place_overwrite(tmp_path):
    train, source = tmp_path / "train", tmp_path / "source"
    _write(train, [{"question": "A", "answer": "0.00"}])
    _write(source, [{
        "question": "A", "answer": "0.00",
        "verified": False, "format_status": "non_spanish_reasoning",
        "trace": "<think>English</think><answer>0.00</answer>",
    }])
    with pytest.raises(ValueError, match="different"):
        reverify_language_rejections(train, source, source)
