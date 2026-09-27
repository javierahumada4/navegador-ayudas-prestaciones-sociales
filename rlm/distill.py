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
    ``question``, ``answer``, ``trace``, ``verified`` and ``teacher``,
    plus per-sample metrics (``predicted``, ``n_tokens``, ``hit_max_tokens``...).

    Log: one JSON line per example in ``--log-path``, appended as the run goes so a
    crashed session keeps what it did. Load with ``pd.read_json(path, lines=True)``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from dotenv import load_dotenv

from rlm.data import load_domain_dataset
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import Verifier, build_verifier

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


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

def _load_model(teacher: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load the teacher model taking into account available resources.

    Args: 
        teacher (str): Teacher model name.

    Returns:
        model (PreTrainedModel): Instance of the model ready to use.
        tokenizer (PreTrainedTokenizerBase): Tokenizer of the teacher.
    """
    # Heavy imports here so the module (and its helpers) import without the train extra.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    model = AutoModelForCausalLM.from_pretrained(
        teacher, dtype=dtype, device_map=device
    ).eval()
    return model, tokenizer

def _append_jsonl(path: Path, record: dict) -> None:
    """
    Append one record to a JSON Lines file, flushing it to disk immediately.

    Args:
        path (Path): JSONL file.
        record (dict): Record to write as one line.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def _prepend_rules(
    example: dict, 
    ) -> list[dict]:
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
    log_path: Path | None = None,
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
        teacher_uses_rule_context (bool): Prepend each row's ``rule_context``
                            to the teacher prompt only.
        log_path (Path | None): Per-example metrics JSONL. ``None`` disables it.

    Returns:
        rows (list[dicts]): New dataset made of formated json traces.
    """
    import torch
    from tqdm.auto import tqdm

    print(f"Loading teacher {teacher} (first run downloads it)...", flush=True)
    model, tokenizer = _load_model(teacher)

    rows: list[dict] = []
    kept = 0
    progress = tqdm(dataset, desc="distill", unit="example")
    for i, example in enumerate(progress):
        prompt = example["prompt"]
        if teacher_uses_rule_context and example.get("rule_context"):
            prompt = _prepend_rules(example)
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        start = time.perf_counter()
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

        gen_seconds = time.perf_counter() - start

        prompt_len = inputs["input_ids"].shape[1]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        n_verified = 0
        for j, seq in enumerate(generated):
            completion_ids = seq[prompt_len:]
            raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
            trace = _canonical_trace(raw)
            result = verifier.verify(trace, str(example["answer"]))
            n_verified += result.is_correct
            n_tokens = int((completion_ids != pad_id).sum())
            rows.append(
                {
                    "example_id": i,
                    "sample_idx": j,
                    "question": example["prompt"][-1]["content"],
                    "answer": str(example["answer"]),
                    "predicted": extract_answer(trace),
                    "trace": trace,
                    "verified": result.is_correct,
                    "raw_valid_format": has_valid_format(raw),
                    "n_tokens": n_tokens,
                    "hit_max_tokens": n_tokens >= max_new_tokens,
                    "teacher": teacher,
                    "params": example.get("params"),
                    "branches": example.get("branches"),
                }
            )

        kept += n_verified
        progress.set_postfix(
            verified=f"{kept}/{len(rows)}", accept=f"{100 * kept / len(rows):.0f}%"
        )

        if log_path is not None:
            _append_jsonl(
                log_path,
                {
                    "example_id": i,
                    "template_id": example.get("template_id"),
                    "branches": example.get("branches"),
                    "n_samples": samples,
                    "n_verified": n_verified,
                    "prompt_tokens": prompt_len,
                    "gen_seconds": round(gen_seconds, 2),
                },
            )
    return rows


def main() -> None:
    # HF_TOKEN (and friends) from the repo's .env; variables already set in the shell win.
    load_dotenv(override=False)
    print(f"HF_TOKEN: {'found' if os.environ.get('HF_TOKEN') else 'not set (public models only)'}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--verifier", default="imv")
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    parser.add_argument(
        "--log-path",
        default="rlm/data/logs/distill_metrics.jsonl",
        help="per-example metrics JSONL, overwritten at the start of each run",
    )
    parser.add_argument(
        "--no-teacher-rules",
        action="store_true",
        help="do not prepend row.rule_context to the teacher prompt",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    dataset = load_domain_dataset(args.data)
    traces = generate_traces(
        dataset,
        args.teacher,
        args.samples,
        args.max_new_tokens,
        build_verifier(args.verifier),
        teacher_uses_rule_context=not args.no_teacher_rules,
        log_path=log_path,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    kept = sum(1 for t in traces if t["verified"])
    print(
        f"{kept}/{len(traces)} traces verified "
        f"({100 * kept / max(len(traces), 1):.1f}%) -> {out} | log -> {log_path}"
    )


if __name__ == "__main__":
    main()
