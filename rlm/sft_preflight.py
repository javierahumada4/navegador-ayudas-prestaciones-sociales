"""Fail-closed SFT token/chat-template preflight.

TRL truncates prompt+completion to SFTConfig.max_length. For IMV, the
supervision must include the *final* </answer> tag; silently teaching a
truncated reasoning prefix is worse than refusing an overlength example.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from rlm.rewards import has_valid_format

FINAL_AMOUNT = re.compile(r"<answer>\d+\.\d{2}</answer>\s*$")


def _token_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def validate_sft_sequences(dataset, tokenizer, max_length: int, report_path: str | Path,
                           *, reserve_tokens: int = 8) -> dict:
    """Audit every actually-rendered Qwen3 chat and fail if any target may be truncated.

    Uses the same chat template and tokenizer as SFTTrainer. The small reserve
    accommodates EOS/template suffix differences; no invalid row is discarded
    silently and no model weights are loaded during this check.
    """
    if max_length <= reserve_tokens:
        raise ValueError("SFT max_length must exceed the end-of-sequence safety reserve")
    reasons: Counter[str] = Counter()
    rejected: list[dict] = []
    longest = 0
    longest_prompt = 0
    for idx, row in enumerate(dataset):
        prompt, completion = row["prompt"], row["completion"]
        trace = completion[0]["content"] if len(completion) == 1 else ""
        reason = None
        tokens = 0
        prompt_tokens = 0
        if not has_valid_format(trace) or not FINAL_AMOUNT.search(trace):
            reason = "invalid_or_unclosed_answer_tags"
        else:
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                full_text = tokenizer.apply_chat_template(
                    prompt + completion, tokenize=False, add_generation_prompt=False
                )
                if not full_text.startswith(prompt_text):
                    reason = "chat_template_prompt_mismatch"
                else:
                    # Qwen3 may include the opening <think> in its assistant
                    # generation prefix. Inspect the full *assistant turn*,
                    # not only the continuation after that prefix.
                    marker = "<|im_start|>assistant"
                    if marker in full_text:
                        assistant_text = full_text.rsplit(marker, 1)[-1]
                    elif "<assistant>" in full_text:
                        assistant_text = full_text.rsplit("<assistant>", 1)[-1]
                    else:
                        assistant_text = full_text[len(prompt_text):]
                    expected_answer = re.search(r"<answer>\d+\.\d{2}</answer>", trace)
                    if (assistant_text.count("<think>") != 1
                            or assistant_text.count("</think>") != 1
                            or assistant_text.count("<answer>") != 1
                            or assistant_text.count("</answer>") != 1
                            or expected_answer is None
                            or expected_answer.group() not in assistant_text):
                        reason = "assistant_completion_tags_mismatch"
                    else:
                        prompt_tokens = len(_token_ids(tokenizer, prompt_text))
                        tokens = len(_token_ids(tokenizer, full_text))
                        if tokens <= prompt_tokens:
                            reason = "no_completion_tokens"
                        elif tokens + reserve_tokens > max_length:
                            reason = "answer_may_be_truncated"
            except (KeyError, TypeError, ValueError) as exc:
                reason = "tokenizer_error"
                rejected.append({"index": idx, "reason": reason, "details": str(exc)})
                reasons[reason] += 1
                continue
        longest = max(longest, tokens)
        longest_prompt = max(longest_prompt, prompt_tokens)
        if reason:
            reasons[reason] += 1
            rejected.append({
                "index": idx, "reason": reason,
                "full_tokens": tokens, "prompt_tokens": prompt_tokens,
            })
    report = {
        "checked": len(dataset),
        "accepted": len(dataset) - len(rejected),
        "rejected": len(rejected),
        "max_length": max_length,
        "reserve_tokens": reserve_tokens,
        "max_observed_tokens": longest,
        "max_prompt_tokens": longest_prompt,
        "rejected_by_reason": dict(reasons),
        "examples": rejected[:20],
    }
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not len(dataset) or rejected:
        raise ValueError(
            "SFT preflight rejected "
            f"{len(rejected)}/{len(dataset)} traces; see {out}. "
            "At least one <answer> tag is malformed or may be truncated by max_length. "
            "Do not train on incomplete targets. Use a new run with an appropriate "
            "context window or improve/shorten the teacher traces."
        )
    return report
