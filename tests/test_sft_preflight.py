"""SFT preflight must refuse any chat where the answer is lost."""
import json

import pytest

from rlm.sft_preflight import validate_sft_sequences


class TinyTokenizer:
    """Model-independent stub that renders chat like a causal LM tokenizer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>"+m["content"]+"</>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}


def row(trace):
    return {
        "prompt": [{"role": "user", "content": "Reglas y caso."}],
        "completion": [{"role": "assistant", "content": trace}],
    }


def test_complete_short_sft_target_passes(tmp_path):
    tok = TinyTokenizer()
    # TinyTokenizer outputs slightly different assistant headers for the prompt
    # and complete conversation unless the header is normalised below.
    tok.apply_chat_template = lambda messages, tokenize=False, add_generation_prompt=False: (
        "".join(f"<{m['role']}>" + m["content"] + "</>" for m in messages[:-1])
        + ("<assistant>" + messages[-1]["content"] + "</>" if messages[-1]["role"] == "assistant"
           else f"<{messages[-1]['role']}>" + messages[-1]["content"] + "</>"
                + ("<assistant>" if add_generation_prompt else ""))
    )
    report = validate_sft_sequences(
        [row("<think>Breve.</think><answer>57.50</answer>")],
        tok, 500, tmp_path / "report.json",
    )
    assert report["accepted"] == 1
    assert json.loads((tmp_path / "report.json").read_text())["rejected"] == 0


def test_sft_preflight_blocks_overlength_answer_and_saves_report(tmp_path):
    tok = TinyTokenizer()
    def chat(messages, tokenize=False, add_generation_prompt=False):
        return ("<user>Reglas y caso.</>" + "<assistant>"
                + (messages[-1]["content"] + "</>" if messages[-1]["role"] == "assistant" else ""))
    tok.apply_chat_template = chat
    trace = "<think>" + "x" * 200 + "</think><answer>57.50</answer>"
    path = tmp_path / "preflight.json"
    with pytest.raises(ValueError, match="SFT preflight rejected"):
        validate_sft_sequences([row(trace)], tok, 100, path)
    saved = json.loads(path.read_text())
    assert saved["rejected_by_reason"] == {"answer_may_be_truncated": 1}
    assert saved["accepted"] == 0


def test_sft_preflight_rejects_bad_or_missing_answer(tmp_path):
    tok = TinyTokenizer()
    for trace in ("<think>Incompleto", "<think>R</think><answer>IMPORTE</answer>"):
        with pytest.raises(ValueError, match="SFT preflight rejected"):
            validate_sft_sequences([row(trace)], tok, 1000, tmp_path / "failure.json")
        assert json.loads((tmp_path / "failure.json").read_text())["rejected"] == 1


def test_qwen_native_thinking_prefix_does_not_trigger_false_failure(tmp_path):
    class NativeThinkingTokenizer(TinyTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "<|im_start|>user\nReglas y caso.<|im_end|><|im_start|>assistant\n<think>\n"
            trace = messages[-1]["content"]
            return ("<|im_start|>user\nReglas y caso.<|im_end|>"
                    "<|im_start|>assistant\n<think>\n"
                    + trace.removeprefix("<think>") + "<|im_end|>")
    result = validate_sft_sequences(
        [row("<think>Breve.</think><answer>57.50</answer>")],
        NativeThinkingTokenizer(), 500, tmp_path / "native.json",
    )
    assert result["accepted"] == 1
