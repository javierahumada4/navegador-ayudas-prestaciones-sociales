"""CPU-only checks for the common 12-arm IMV benchmark (never load Qwen3)."""

from __future__ import annotations

import csv
import json

import pytest

from tool_use import compare_variants as cv


def case(question="Calcula el IMV", answer="431.26"):
    return {
        "question": question, "answer": answer,
        "rule_context": "Normativa IMV 2026",
        "branches": {"family": "ordinary"},
        "params": {"secret": "never pass to model"},
    }


def write_jsonl(path, items):
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def test_matrix_has_exactly_all_twelve_factorial_variants():
    assert len(cv.VARIANTS) == 12
    assert len(cv.VARIANT_BY_NAME) == 12
    assert {v.checkpoint for v in cv.VARIANTS} == set(cv.CHECKPOINTS)
    assert {v.name for v in cv.VARIANTS if not v.with_tool} == {
        "base", "sft", "grpo_zero", "sft_grpo"
    }
    assert sum(v.with_tool and not v.tool_grpo for v in cv.VARIANTS) == 4
    assert sum(v.tool_grpo for v in cv.VARIANTS) == 4
    assert cv.VARIANT_BY_NAME["grpo_zero"].adapter_key == "grpo_zero"
    assert cv.VARIANT_BY_NAME["sft_grpo_tool"].adapter_key == "sft_grpo"
    assert cv.VARIANT_BY_NAME["base_tool_grpo"].adapter_key == "tool_grpo_base"
    assert cv.VARIANT_BY_NAME["sft_grpo_tool_grpo"].adapter_key == "tool_grpo_sft_grpo"


def test_extract_only_one_final_amount_without_reasoning_leak():
    assert cv.select_final_answer("<think>373.76 + 57.50</think><answer>431.26</answer>") == "431.26"
    assert cv.select_final_answer("431.26") == "431.26"
    assert cv.select_final_answer(
        "<think>373.76</think><answer>431.26</answer><answer>0.00</answer>"
    ) is None
    assert cv.select_final_answer("<think>431.26") is None
    assert cv.select_final_answer("Ingreso 373.76; CAPI 57.50, total 431.26") is not None
    assert not cv.imv_numbers_match(
        cv.select_final_answer("Ingreso 373.76; CAPI 57.50, total 431.26"), "431.26"
    )


def test_shared_numeric_score_for_tool_and_non_tool():
    source = case()
    for name in ("base", "base_tool", "sft_grpo", "sft_grpo_tool_grpo"):
        variant = cv.VARIANT_BY_NAME[name]
        row = cv.make_row(
            source, "431.26", variant=variant, sample=0, elapsed_ms=5.0,
            n_tokens=3, raw_generations=["raw"],
            tool_calls=[{"name": "calcular_imv", "ok": True}] if variant.with_tool else [],
            valid_format=True,
        )
        assert row["numeric_correct"] is True
        assert row["absolute_error_eur"] == 0.0
        assert row["tool_execution_ok"] is (True if variant.with_tool else None)
        summary = cv.summarize([row], variant)
        assert summary["numeric_accuracy"] == 1.0
        assert summary["mae_eur"] == 0.0
    wrong = cv.make_row(
        source, "430.26", variant=cv.VARIANT_BY_NAME["base"],
        sample=0, elapsed_ms=1, n_tokens=1,
        raw_generations=[], tool_calls=[], valid_format=False,
    )
    assert wrong["absolute_error_eur"] == 1.0
    assert not wrong["numeric_correct"]


def test_ambiguous_answers_have_no_mae_and_score_zero():
    row = cv.make_row(
        case(), "373.76 + 57.50 = 431.26", variant=cv.VARIANT_BY_NAME["base"],
        sample=0, elapsed_ms=1, n_tokens=None,
        raw_generations=[], tool_calls=[], valid_format=False,
    )
    assert not row["numeric_correct"]
    assert row["absolute_error_eur"] is None
    summary = cv.summarize([row], cv.VARIANT_BY_NAME["base"])
    assert summary["mae_eur"] is None
    assert summary["parseable_rate"] == 0


def test_checkpoint_inventory_requires_real_final_lora(tmp_path):
    config = {"adapters": {"sft": str(tmp_path / "sft")}}
    assert cv.resolve_adapter(cv.VARIANT_BY_NAME["base"], config) is None
    with pytest.raises(FileNotFoundError):
        cv.resolve_adapter(cv.VARIANT_BY_NAME["sft"], config)
    folder = tmp_path / "sft"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert cv.resolve_adapter(cv.VARIANT_BY_NAME["sft"], config) == str(folder)
    with pytest.raises(ValueError, match="Missing checkpoint"):
        cv.resolve_adapter(cv.VARIANT_BY_NAME["grpo_zero"], config)


def test_json_manifest_rejects_unknown_adapter_fields(tmp_path):
    path = tmp_path / "variants.json"
    path.write_text('{"adapters":{"impossible":"/fake"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        cv.load_config(path)


def test_held_out_guard_detects_exact_question_overlap(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, [case("same")])
    write_jsonl(test, [case("same")])
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        cv.check_test_is_held_out(cv.load_shared_test(test), train)
    write_jsonl(test, [case("different")])
    cv.check_test_is_held_out(cv.load_shared_test(test), train)


def test_dry_run_lists_all_twelve_and_no_models_loaded(tmp_path, monkeypatch):
    test_data, train = tmp_path / "test.jsonl", tmp_path / "train.jsonl"
    write_jsonl(test_data, [case()])
    write_jsonl(train, [case("other")])
    conf = tmp_path / "variants.json"
    conf.write_text('{"base_model":"Qwen/Qwen3-1.7B","adapters":{}}', encoding="utf-8")
    output = tmp_path / "out"

    def never_run(*args, **kwargs):
        raise AssertionError("--dry-run must not load any model")

    monkeypatch.setattr(cv, "run_variant", never_run)
    cv.main([
        "--all", "--dry-run", "--data", str(test_data),
        "--train-data", str(train), "--config", str(conf),
        "--out-dir", str(output),
    ])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["data_sha256"] == cv.file_hash(test_data)
    assert set(manifest["variants"]) == set(cv.VARIANT_BY_NAME)
    assert manifest["variants"]["base"]["status"] == "ready"
    assert manifest["variants"]["base_tool"]["status"] == "ready"
    assert manifest["variants"]["sft"]["status"] == "missing_checkpoint"
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert rows[0]["status"] == "ready"
    assert rows[3]["status"] == "missing_checkpoint"
    assert not list(output.glob("base.json"))


def test_one_variant_run_writes_json_and_csv_without_gpu(tmp_path, monkeypatch):
    test_data = tmp_path / "test.jsonl"
    write_jsonl(test_data, [case()])
    conf = tmp_path / "variants.json"
    conf.write_text('{"adapters":{}}', encoding="utf-8")
    output = tmp_path / "out"
    seen = []

    def fake_run(variant, cases, **kwargs):
        seen.extend(cases)
        return [
            cv.make_row(
                cases[0], "431.26", variant=variant, sample=0,
                elapsed_ms=1, n_tokens=2,
                raw_generations=["<answer>431.26</answer>"],
                tool_calls=[], valid_format=False,
            )
        ]

    monkeypatch.setattr(cv, "run_variant", fake_run)
    cv.main([
        "--variant", "base", "--data", str(test_data),
        "--train-data", "", "--config", str(conf), "--out-dir", str(output),
    ])
    assert seen == [case()]
    result = json.loads((output / "base.json").read_text())
    assert result["summary"]["numeric_accuracy"] == 1.0
    assert result["summary"]["mae_eur"] == 0.0
    assert result["rows"][0]["question"] == case()["question"]
    assert "secret" not in str(result["rows"])
    with (output / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["variant"] == "base"
    assert rows[0]["numeric_accuracy"] == "1.0"
    assert rows[1]["status"] == "not_selected"


def test_invalid_selection_is_error(tmp_path):
    with pytest.raises(SystemExit):
        cv.main(["--variant", "does_not_exist"])
