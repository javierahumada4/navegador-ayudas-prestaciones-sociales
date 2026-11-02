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
    assert "<answer>IMPORTE</answer>" in system
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
    assert dataset[0]["prompt"][1]["content"] == "Calcula el IMV"
    assert dataset[0]["answer"] == "431.26"
