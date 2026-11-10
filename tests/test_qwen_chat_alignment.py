"""Native Qwen3 reasoning chat prefix must match SFT and generation."""
import pytest

from rlm.chat_template import (
    render_supervised_exchange, render_thinking_prompt, restore_thinking_prefix,
)
from rlm.rewards import has_valid_imv_format
from rlm.sft_selection import select_budgeted_traces


class NativeQwenTemplate:
    """Small model-free stand-in for the published Qwen3 thinking template."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=False, enable_thinking=True):
        assert enable_thinking
        prompt = "".join(
            "<|im_start|>" + msg["role"] + "\n" + msg["content"] + "<|im_end|>\n"
            for msg in messages if msg["role"] != "assistant"
        )
        prefix = prompt + "<|im_start|>assistant\n"
        if add_generation_prompt:
            return prefix
        completion = messages[-1]["content"]
        assert completion.startswith("<think>")
        thought, answer = completion.split("</think>", 1)
        return (
            prefix + "<think>\n" + thought.removeprefix("<think>").strip()
            + "\n</think>\n\n" + answer.strip()
            + "<|im_end|>\n"
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}


def test_sft_and_generation_have_identical_native_prefix(tmp_path):
    tokenizer = NativeQwenTemplate()
    question = [{"role": "user", "content": "Caso de prueba"}]
    trace = "<think>Razonamiento corto.</think><answer>41.20</answer>"
    prefix, full = render_supervised_exchange(tokenizer, question, trace)
    assert prefix == render_thinking_prompt(tokenizer, question)
    assert full.startswith(prefix)
    assert has_valid_imv_format(full[len(prefix):].replace("<|im_end|>", "").strip())
    from datasets import Dataset

    raw = Dataset.from_list([{
        "question": "Caso de prueba", "trace": trace, "verified": True,
        "rule_context": "Reglas", "branches": {"family": "ordinary"},
    }])
    selected, report = select_budgeted_traces(
        raw, tokenizer, max_completion_tokens=1024, max_length=8192,
        max_per_problem=2, min_problem_coverage=1.0, min_families=1,
        report_path=tmp_path / "valid.json",
    )
    assert len(selected) == 1
    assert report["traces_selected"] == 1


def test_selection_rejects_nested_final_in_qwen_template(tmp_path):
    from datasets import Dataset

    bad = "<think><answer>0.00</answer> R.</think><answer>0.00</answer>"
    raw = Dataset.from_list([{
        "question": "Caso", "trace": bad, "verified": True,
        "rule_context": "Reglas", "branches": {"family": "ordinary"},
    }])
    with pytest.raises(ValueError, match="No verified complete SFT traces"):
        select_budgeted_traces(
            raw, NativeQwenTemplate(), max_completion_tokens=1024, max_length=8192,
            max_per_problem=2, min_problem_coverage=1.0, min_families=1,
            report_path=tmp_path / "invalid.json",
        )


def test_prefix_recovery_only_when_native_template_already_started_thinking():
    assert restore_thinking_prefix(
        "<|im_start|>assistant\n", "<think>R.</think><answer>1.00</answer>"
    ) == "<think>R.</think><answer>1.00</answer>"
    assert restore_thinking_prefix(
        "<|im_start|>assistant\n<think>\n", "R.</think><answer>1.00</answer>"
    ) == "<think>\nR.</think><answer>1.00</answer>"
    assert restore_thinking_prefix(
        "<|im_start|>assistant\n<think>\n", "<think>R.</think><answer>1.00</answer>"
    ) == "<think>R.</think><answer>1.00</answer>"
