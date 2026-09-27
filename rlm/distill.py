"""Generate verified IMV reasoning traces for cold-start SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.data import load_domain_dataset
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import Verifier, build_verifier


def _canonical_trace(trace: str) -> str:
    if has_valid_format(trace):
        return trace
    answer = extract_answer(trace)
    if answer is None:
        return trace
    # Preserve the teacher's text as reasoning but force the canonical training format.
    return f"<think>{trace.strip()}</think><answer>{answer.strip()}</answer>"


def generate_traces(
    dataset,
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    *,
    teacher_uses_rule_context: bool = True,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    model = AutoModelForCausalLM.from_pretrained(
        teacher, dtype=dtype, device_map=device
    ).eval()

    rows: list[dict] = []
    for example in dataset:
        prompt = [dict(m) for m in example["prompt"]]
        if teacher_uses_rule_context and example.get("rule_context"):
            for message in prompt:
                if message.get("role") == "user":
                    message["content"] = (
                        f"{example['rule_context']}\n\n{message['content']}"
                    )
                    break

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
