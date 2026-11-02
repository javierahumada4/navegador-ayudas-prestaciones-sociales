"""One native Qwen3 thinking-mode chat rendering contract for all stages.

We ask the tokenizer to format the assistant prefix, never hand-prefill a
second <think> in training or inference. TRL's conversational SFT path uses
the tokenizer's default chat template; preflight verifies that its default
and explicit thinking renders agree before training.
"""
from __future__ import annotations


def render_thinking_prompt(tokenizer, messages: list[dict]) -> str:
    options = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=True, **options)
    except TypeError as exc:
        # Simple tokenizer stand-ins and templates with no thinking switch.
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **options)


def render_supervised_exchange(tokenizer, prompt: list[dict], trace: str):
    """Return prompt and full SFT rendering; assert they have identical prefixes."""
    prefix = render_thinking_prompt(tokenizer, prompt)
    complete = prompt + [{"role": "assistant", "content": trace}]
    full = tokenizer.apply_chat_template(
        complete, tokenize=False, add_generation_prompt=False
    )
    if not full.startswith(prefix):
        raise ValueError("Qwen3 SFT chat_template does not preserve the generation prefix")
    # SFTTrainer applies this same default chat template to conversational
    # prompt/completion datasets. Refuse model revisions that change the switch.
    default_prefix = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    if default_prefix != prefix:
        raise ValueError("Qwen3 default chat template disagrees with enable_thinking=True")
    return prefix, full


def restore_thinking_prefix(generation_prefix: str, generated: str) -> str:
    """Include a native assistant-prefix <think> when decoding excludes it."""
    if generation_prefix.rstrip().endswith("<think>") and not generated.lstrip().startswith("<think>"):
        return "<think>\n" + generated
    return generated
