"""CPU checks for the five-arm comparison. No Qwen model is downloaded."""

import csv
import json

import pytest

from tool_use import compare_imv as ev
from tool_use.imv_data import assert_disjoint, read_cases, user_prompt


def item(question="Calcula el IMV", answer="431.26"):
    return {
        "question": question, "answer": answer, "rule_context": "Reglas 2026",
        "params": {"secret": "never pass this to the LLM"},
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def test_five_arms_exact_scope():
    assert [v.name for v in ev.VARIANTS] == [
        "base", "sft", "sft_grpo", "sft_grpo_tool",
        "sft_grpo_tool_grpo",
    ]
    assert [v.with_tool for v in ev.VARIANTS] == [
        False, False, False, True, True,
    ]
    assert ev.VARIANT_BY_NAME["sft_grpo_tool"].adapter_key == "sft_grpo"
    assert ev.VARIANT_BY_NAME["sft_grpo_tool_grpo"].adapter_key == "tool_grpo"


def test_scores_same_money_contract_for_all_five():
    case = item()
    for variant in ev.VARIANTS:
        calls = (
            [{"name": "calcular_imv", "ok": True}] if variant.with_tool else []
        )
        row = ev.score_case(
            case, "431.26", variant=variant, calls=calls,
            raw=["generated"], duration_ms=10, tokens=7,
        )
        assert row["numeric_correct"]
        assert row["absolute_error_eur"] == 0
        assert row["tool_executed"] is (True if variant.with_tool else None)
        assert ev.summarize([row], variant)["numeric_accuracy"] == 1.0


def test_reasoning_parser_does_not_read_intermediate_numbers():
    assert ev.final_answer_from_reasoning(
        "<think>373.76 + 57.50</think><answer>431.26</answer>"
    ) == "431.26"
    assert ev.final_answer_from_reasoning("431.26") == "431.26"
    assert ev.final_answer_from_reasoning(
        "<think>373.76</think><answer>431.26</answer><answer>0.00</answer>"
    ) is None
    assert ev.final_answer_from_reasoning("<think>373.76") is None
    assert ev.error_eur("373.76 + 57.50 = 431.26", "431.26") is None


def test_shared_inputs_and_overlap_guard(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, [item("Entreno")])
    write_jsonl(test, [item("Test")])
    rows = read_cases(test)
    assert rows[0]["question"] == "Test"
    assert user_prompt(rows[0]) == "Reglas 2026\n\nTest"
    assert "secret" not in user_prompt(rows[0])
    assert_disjoint(read_cases(train), rows)
    write_jsonl(test, [item("Entreno")])
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        assert_disjoint(read_cases(train), read_cases(test))


def test_missing_checkpoint_is_not_silent_fallback(tmp_path):
    config = {"adapters": {"sft": str(tmp_path / "sft")}}
    assert ev.resolve_adapter(ev.VARIANT_BY_NAME["base"], config) is None
    with pytest.raises(FileNotFoundError):
        ev.resolve_adapter(ev.VARIANT_BY_NAME["sft"], config)
    with pytest.raises(ValueError, match="missing"):
        ev.resolve_adapter(ev.VARIANT_BY_NAME["sft_grpo"], config)


def test_no_model_loaded_by_dry_run_and_all_five_csv_rows(
    tmp_path, monkeypatch,
):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, [item("Entreno")])
    write_jsonl(test, [item("Test")])
    cfg = tmp_path / "conf.json"
    cfg.write_text('{"adapters":{}}', encoding="utf-8")
    out = tmp_path / "out"

    def not_ever(*args, **kwargs):
        raise AssertionError("dry-run loaded a model")

    monkeypatch.setattr(ev, "run_variant", not_ever)
    ev.main([
        "--all", "--dry-run", "--data", str(test), "--train-data", str(train),
        "--config", str(cfg), "--out-dir", str(out),
    ])
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["variants"]) == 5
    assert manifest["variants"]["base"]["status"] == "ready"
    assert all(
        manifest["variants"][v]["status"] == "missing_checkpoint"
        for v in ("sft", "sft_grpo", "sft_grpo_tool", "sft_grpo_tool_grpo")
    )
    with (out / "comparison.csv").open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 5
    assert not list(out.glob("base.json"))


def test_csv_json_evaluation_with_stubbed_runner(tmp_path, monkeypatch):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, [item("Entreno")])
    write_jsonl(test, [item("Test")])
    cfg = tmp_path / "config.json"
    cfg.write_text('{"adapters":{}}', encoding="utf-8")
    out = tmp_path / "out"
    seen = []

    def fake_runner(variant, cases, **kwargs):
        seen.extend(cases)
        return [ev.score_case(
            cases[0], "431.26", variant=variant,
            raw=["431.26"], calls=[], duration_ms=3, tokens=5,
        )]

    monkeypatch.setattr(ev, "run_variant", fake_runner)
    ev.main([
        "--variant", "base", "--data", str(test), "--train-data", str(train),
        "--config", str(cfg), "--out-dir", str(out),
    ])
    assert seen == [item("Test")]
    result = json.loads((out / "base.json").read_text())
    assert result["summary"]["numeric_accuracy"] == 1.0
    assert result["rows"][0]["predicted"] == "431.26"
    assert "secret" not in str(result["rows"])
    with (out / "comparison.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["numeric_accuracy"] == "1.0"
    assert rows[1]["status"] == "not_selected"
