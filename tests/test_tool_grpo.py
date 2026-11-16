"""CPU-only tests: correct GRPO tool rewards, leakage checks and CLI validation."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rlm.imv_engine import evaluate_imv_case
from tool_use import tool_grpo_rewards as reward
from tool_use import train_tool_grpo as trainer
from tool_use.imv_tools import IMVCase
from tool_use.tools import default_registry


@pytest.fixture()
def case():
    def person(pid, birth):
        return {
            "id": pid,
            "date_of_birth": birth,
            "legal_residence_in_spain": True,
            "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": "2010-01-01",
        }

    return {
        "application_date": "2026-09-26",
        "applicant_id": "p1",
        "persons": [
            person("p1", "1985-05-10"),
            person("p2", "1984-01-01"),
            person("p3", "2010-06-01"),
        ],
        "domicile": {
            "co_resident_ids": ["p1", "p2", "p3"],
            "same_domicile_since": "2010-06-01",
        },
        "relationships": [
            {"person_a": "p1", "person_b": "p2", "type": "spouse"},
            {"person_a": "p1", "person_b": "p3", "type": "parent_child", "custody": "shared"},
            {"person_a": "p2", "person_b": "p3", "type": "parent_child", "custody": "shared"},
        ],
        "economic": {
            "countable_income_annual_eur": 9600,
            "net_worth_eur": 5000,
            "non_corporate_assets_eur": 5000,
        },
    }


def reference(case):
    return json.dumps(
        IMVCase.model_validate(case).model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
    )


def completion(case, final="431.26", *, structured=True):
    content = {"name": "calcular_imv", "arguments": {"case": case}}
    if structured:
        model_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": content}],
        }
    else:
        model_call = {
            "role": "assistant",
            "content": "<tool_call>" + json.dumps(content) + "</tool_call>",
        }
    return [
        model_call,
        {"role": "tool", "name": "calcular_imv", "content": "observacion"},
        {"role": "assistant", "content": final},
    ]


def test_training_wrapper_is_exactly_single_production_tool(case):
    assert set(default_registry().tools) == {"calcular_imv"}
    assert trainer.calcular_imv.__name__ == "calcular_imv"
    assert trainer.calcular_imv.__annotations__["case"] is dict
    actual = trainer.calcular_imv(case)
    assert actual["total_monthly_eur"] == 431.26


def test_reward_checks_real_execution_and_reference_case(case):
    ref = reference(case)
    for structured in (True, False):
        episode = completion(case, structured=structured)
        flags = reward.inspect_rollout(episode, "431.26", ref)
        assert all(flags.values()), flags
        kwargs = {"completions": [episode], "answer": ["431.26"], "reference_case": [ref]}
        for function in (
            reward.tool_selection_reward,
            reward.tool_arguments_reward,
            reward.tool_execution_reward,
            reward.final_answer_reward,
            reward.numeric_format_reward,
        ):
            assert function(**kwargs) == [1.0]


def test_bad_arguments_do_not_get_execution_or_case_reward(case):
    wrong = deepcopy(case)
    wrong["economic"]["countable_income_annual_eur"] = 18000
    flags = reward.inspect_rollout(completion(wrong, "431.26"), "431.26", reference(case))
    assert flags["selected"]
    assert flags["arguments_valid"]
    assert not flags["arguments_correct"]
    assert not flags["executed_correctly"]
    assert flags["final_amount_correct"]

    incomplete = deepcopy(case)
    incomplete["economic"].pop("net_worth_eur")
    flags = reward.inspect_rollout(
        completion(incomplete), "431.26", reference(case)
    )
    assert flags["selected"]
    assert not flags["arguments_valid"]
    assert not flags["arguments_correct"]
    assert not flags["executed_correctly"]


def test_invalid_json_and_wrong_tool_get_no_selection_reward(case):
    malformed = [
        {"role": "assistant", "content": '<tool_call>{"name":"calcular_imv","arguments":{oops}}</tool_call>'},
        {"role": "assistant", "content": "431.26"},
    ]
    assert not reward.inspect_rollout(malformed, "431.26", reference(case))["selected"]
    wrong_tool = [
        {"role": "assistant", "tool_calls": [{"function": {
            "name": "get_weather", "arguments": {"location": "Madrid"},
        }}]},
        {"role": "assistant", "content": "431.26"},
    ]
    assert not reward.inspect_rollout(wrong_tool, "431.26", reference(case))["selected"]


def test_duplicate_calls_get_no_reward_for_correct_execution(case):
    episode = completion(case)
    episode.insert(1, episode[0].copy())
    flags = reward.inspect_rollout(episode, "431.26", reference(case))
    assert not flags["selected"]
    assert not flags["executed_correctly"]
    assert not flags["arguments_correct"]


def test_final_is_not_taken_from_thinking_or_tool_observation(case):
    episode = completion(case)
    episode.pop()
    flags = reward.inspect_rollout(episode, "431.26", reference(case))
    assert flags["executed_correctly"]
    assert not flags["final_amount_correct"]
    assert not flags["final_format"]
    episode.append({"role": "assistant", "content": "IMV 373.76 + CAPI 57.50 = 431.26"})
    assert not reward.inspect_rollout(
        episode, "431.26", reference(case)
    )["final_amount_correct"]


def test_training_dataset_keeps_answers_and_params_out_of_prompts(tmp_path, case):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train_case = {
        "question": "Pregunta del entrenamiento.",
        "answer": "431.26",
        "rule_context": "Normativa del IMV",
        "params": {**case, "_family": "ordinary"},
    }
    test_case = {
        "question": "Pregunta del test diferente.",
        "answer": "0.00",
        "params": case,
    }
    train.write_text(json.dumps(train_case) + "\n", encoding="utf-8")
    test.write_text(json.dumps(test_case) + "\n", encoding="utf-8")
    examples = trainer.prepare_training_examples(train, test, None)
    assert len(examples) == 1
    entry = examples[0]
    prompt = json.dumps(entry["prompt"], ensure_ascii=False)
    assert entry["prompt"][0]["role"] == "system"
    assert "Pregunta del entrenamiento." in prompt
    assert "Normativa del IMV" in prompt
    assert "431.26" not in prompt
    assert "net_worth_eur" not in prompt
    assert "reference_case" in entry
    assert "431.26" == entry["answer"]
    assert "economic" in json.loads(entry["reference_case"])
    assert "_family" not in entry["reference_case"]
    assert evaluate_imv_case(case)["total_monthly_eur"] == 431.26


def test_training_dataset_rejects_leakage(tmp_path, case):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    row = {"question": "Duplicada", "answer": "431.26", "params": case}
    train.write_text(json.dumps(row) + "\n", encoding="utf-8")
    test.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        trainer.prepare_training_examples(train, test, None)


def test_training_data_rejects_missing_params(tmp_path, case):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(json.dumps({"question": "Entreno", "answer": "431.26"}) + "\n")
    test.write_text(json.dumps({"question": "Test", "answer": "0.00", "params": case}) + "\n")
    with pytest.raises(ValueError, match="lacks oracle params"):
        trainer.prepare_training_examples(train, test, None)


def test_training_cli_dry_run_does_not_start_gpu_or_write_checkpoint(
    tmp_path, case, monkeypatch, capsys
):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    conf = tmp_path / "variants.json"
    dest = tmp_path / "weights"
    train.write_text(json.dumps({
        "question": "Pregunta de train", "answer": "431.26", "params": case,
    }) + "\n")
    test.write_text(json.dumps({
        "question": "Pregunta test", "answer": "431.26", "params": case,
    }) + "\n")
    conf.write_text('{"base_model":"Qwen/Qwen3-1.7B","adapters":{}}')
    monkeypatch.setattr(
        trainer, "train",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected GPU training")),
    )
    trainer.main([
        "--source", "base", "--config", str(conf), "--train-data", str(train),
        "--test-data", str(test), "--output", str(dest), "--dry-run",
    ])
    info = json.loads(capsys.readouterr().out)
    assert info["source"] == "base"
    assert info["n_train_examples"] == 1
    assert info["initial_adapter"] is None
    assert len(info["train_sha256"]) == 64
    assert not dest.exists()


def test_training_cli_blocks_overwriting_input_adapter(tmp_path, case):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train.write_text(json.dumps({"question": "train", "answer": "431.26", "params": case}) + "\n")
    test.write_text(json.dumps({"question": "test", "answer": "0.00", "params": case}) + "\n")
    adapter = tmp_path / "sft"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps({"adapters": {"sft": str(adapter)}}))
    with pytest.raises(SystemExit):
        trainer.main([
            "--source", "sft", "--config", str(conf),
            "--train-data", str(train), "--test-data", str(test),
            "--output", str(adapter), "--dry-run",
        ])
