"""Generate verified IMV reasoning traces for cold-start SFT.

    Inspired in distilation methods like Sky-T1, OpenThoughts and DeepSeek's cold start.
    We take a strong model that writes solutions with visible reasoning, a verifier throws away the wrong ones,
    and what survives becomes SFT data. 
    Here the teacher is any model that can think in the
    ``<think>…</think><answer>…</answer>`` format 

    Run::

        uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
            --samples 4 --output rlm/data/sft_traces.jsonl

    Output: one JSON line per generated trace with 
    ``question``, ``answer``, ``trace``, ``verified`` and ``teacher``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlm.data import load_domain_dataset
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import Verifier, build_verifier


def _canonical_trace(trace: str) -> str:
    """
    Helper function to create a canonical trace.
    
    Args: 
        trace (str): Original model trace.

    Returns:
        str: Correclly formated trace.
    """
    if has_valid_format(trace):
        return trace
    answer = extract_answer(trace)
    if answer is None:
        return trace
    # Preserve the teacher's text as reasoning but force the canonical training format.
    return f"<think>{trace.strip()}</think><answer>{answer.strip()}</answer>"

def _load_model(teacher: str) -> AutoModelForCausalLM: 
    """
    Load the teacher model taking into account available resources.

    Args: 
        teacher (str): Teacher model name.

    Returns:
        model (AutoModelForCausalLM): Instance of the model ready to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    model = AutoModelForCausalLM.from_pretrained(
        teacher, dtype=dtype, device_map=device
    ).eval()

def _prepend_rules(
    example, 
    ) -> None:
    """
    Prepend the rule_context.
    Checks for the first user message with rule context and prepends it to the promp.

    Only 1 message of each prompt is prepended with the rules.

    Args:
        example: Example taken from the dataset.
    
    Returns:
        prompt (List[dict]): List of prompts with the rules prepended if asked.
    """

    prompt = [dict(message) for message in example["prompt"]]
    for message in prompt:
        if message.get("role") == "user":
            message["content"] = (
                f"{example['rule_context']}\n\n{message['content']}"
            )
            break
    return prompt

def generate_traces(
    dataset: Iterable[Any],
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    *,
    teacher_uses_rule_context: bool = True,
) -> list[dict]:
    """
    Generates traces using the teacher model.

    Args:
        dataset (Iterable): Question dataset.
        teacher (str): Teacher model name.
        samples (int): Number of samples generated per example.
        max_new_tokens (int): Maximun number of tokens added to the answer through 
                            traces.
        verifier (Verifier): Verifier to delete bad examples.
        teacher_uses_rule_context (bool): TODO.

    Returns:
        rows (list[dicts]): New dataset made of formated json traces.
    """
    model = _load_model(teacher)

    rows: list[dict] = []
    for example in dataset:
        if teacher_uses_rule_context and example.get("rule_context"):
            _prepend_rules(example)
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for seq in generated:
            raw = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            trace = _canonical_trace(raw)
            result = verifier.verify(trace, str(example["answer"]))
            rows.append(
                {
                    "question": example["prompt"][-1]["content"],
                    "answer": str(example["answer"]),
                    "trace": trace,
                    "verified": result.is_correct,
                    "teacher": teacher,
                    "params": example.get("params"),
                    "branches": example.get("branches"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--verifier", default="imv")
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    parser.add_argument(
        "--no-teacher-rules",
        action="store_true",
        help="do not prepend row.rule_context to the teacher prompt",
    )
    args = parser.parse_args()

    dataset = load_domain_dataset(args.data)
    traces = generate_traces(
        dataset,
        args.teacher,
        args.samples,
        args.max_new_tokens,
        build_verifier(args.verifier),
        teacher_uses_rule_context=not args.no_teacher_rules,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = sum(1 for t in traces if t["verified"])
    print(
        f"{kept}/{len(traces)} traces verified "
        f"({100 * kept / max(len(traces), 1):.1f}%) -> {out}"
    )


if __name__ == "__main__":
    main()
