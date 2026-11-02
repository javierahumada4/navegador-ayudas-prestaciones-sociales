"""Native Qwen3 reasoning chat prefix must match SFT and generation."""
import pytest

from rlm.chat_template import (
    render_supervised_exchange, render_thinking_prompt, restore_thinking_prefix,
)
from rlm.rewards import has_valid_imv_format
from rlm.sft_preflight import validate_sft_sequences


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
    row = {"prompt": question, "completion": [{"role": "assistant", "content": trace}]}
    report = validate_sft_sequences([row], tokenizer, 1024, tmp_path / "valid.json")
    assert report["accepted"] == 1


def test_preflight_rejects_nested_final_in_qwen_template(tmp_path):
    tok = NativeQwenTemplate()
    bad = "<think><answer>0.00</answer> R.</think><answer>0.00</answer>"
    row = {
        "prompt": [{"role": "user", "content": "Caso"}],
        "completion": [{"role": "assistant", "content": bad}],
    }
    with pytest.raises(ValueError, match="SFT preflight rejected"):
        validate_sft_sequences([row], tok, 1024, tmp_path / "invalid.json")


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
