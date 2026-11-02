"""IMV prompts share a Spanish task contract; GSM8K keeps its original prompt."""

import json

import pytest

pytest.importorskip("datasets")

from rlm.data import (  # noqa: E402
    IMV_SYSTEM_PROMPT,
    R1_ZERO_SYSTEM_PROMPT,
    build_prompt,
    load_domain_dataset,
)


def test_imv_uses_spanish_system_instructions_and_machine_readable_answer():
    messages = build_prompt("Calcula el IMV y el CAPI.")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Calcula el IMV y el CAPI."}
    system = messages[0]["content"]
    assert system == IMV_SYSTEM_PROMPT
    for term in ("razonamiento en español", "importe mensual", "dos decimales"):
        assert term in system
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        assert tag in system
    assert "IMPORTE" not in system
    assert "entre las etiquetas de apertura <answer> y cierre </answer>" in system
    assert "<answer>431.26</answer>" not in system
    assert "The domain is Spain" not in system


def test_generic_r1_zero_prompt_is_preserved_for_gsm8k():
    assert R1_ZERO_SYSTEM_PROMPT.startswith("A conversation between User and Assistant.")
    assert build_prompt("What is 2 + 2?", R1_ZERO_SYSTEM_PROMPT)[0]["content"] == (
        R1_ZERO_SYSTEM_PROMPT
    )


def test_imv_dataset_loader_uses_the_same_spanish_system_prompt(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"question": "Calcula el IMV", "answer": "431.26"}) + "\n",
        encoding="utf-8",
    )
    dataset = load_domain_dataset(path)
    assert dataset[0]["prompt"][0] == {"role": "system", "content": IMV_SYSTEM_PROMPT}
    assert dataset[0]["prompt"][1]["content"].endswith("\n\nCalcula el IMV")
    assert "REGLAS EXACTAS IMV/CAPI 2026" in dataset[0]["prompt"][1]["content"]
    assert dataset[0]["answer"] == "431.26"


def test_rule_context_is_identical_for_student_and_teacher_dataset(tmp_path):
    from rlm.distill import _teacher_prompt

    row = {
        "question": "Caso X", "answer": "0.00",
        "rule_context": "REGLAS EXACTAS PERSONALIZADAS",
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    student = load_domain_dataset(path)
    teacher = load_domain_dataset(path, include_rules=False)
    assert student[0]["prompt"][1]["content"] == "REGLAS EXACTAS PERSONALIZADAS\n\nCaso X"
    assert teacher[0]["prompt"][1]["content"] == "Caso X"
    teacher_prompt = _teacher_prompt(teacher[0], use_rule_context=True, thinking=True)
    assert teacher_prompt[1]["content"] == student[0]["prompt"][1]["content"]
    assert student[0]["prompt"][1]["content"].count("REGLAS EXACTAS PERSONALIZADAS") == 1


def test_legacy_teacher_traces_receive_current_rules_in_sft(tmp_path):
    from rlm.train_sft import load_sft_dataset

    trace_file = tmp_path / "sft.jsonl"
    trace_file.write_text(
        json.dumps({
            "question": "Caso Y", "answer": "0.00", "verified": True,
            "trace": "<think>Razono brevemente.</think><answer>0.00</answer>",
        }, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dataset = load_sft_dataset(trace_file)
    assert dataset[0]["prompt"][1]["content"].endswith("\n\nCaso Y")
    assert "REGLAS EXACTAS IMV/CAPI 2026" in dataset[0]["prompt"][1]["content"]


def test_sft_preserves_teacher_rule_snapshot(tmp_path):
    from rlm.train_sft import load_sft_dataset

    trace_file = tmp_path / "sft.jsonl"
    trace_file.write_text(
        json.dumps({
            "question": "Caso Z", "answer": "57.50", "verified": True,
            "rule_context": "RULES FROM THE ORIGINAL RUN",
            "trace": "<think>R.</think><answer>57.50</answer>",
        }) + "\n", encoding="utf-8"
    )
    dataset = load_sft_dataset(trace_file)
    assert dataset[0]["prompt"][1]["content"] == "RULES FROM THE ORIGINAL RUN\n\nCaso Z"


def test_teacher_prompt_does_not_contain_copyable_placeholder():
    from rlm.distill import TEACHER_THINKING_SYSTEM_PROMPT

    assert "IMPORTE" not in TEACHER_THINKING_SYSTEM_PROMPT
