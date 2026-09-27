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
from typing import TYPE_CHECKING, Any
import torch
from tqdm.auto import tqdm
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

def _generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    samples: int,
    max_new_tokens: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Generate ``samples`` completions for each text in one ``generate`` call.

    If the batch does not fit in GPU memory it is split in half and retried,
    so a too large ``--batch-size`` slows the run down instead of killing it.

    Args:
        model (PreTrainedModel): Teacher model.
        tokenizer (PreTrainedTokenizerBase): Teacher tokenizer, left padded.
        texts (list[str]): Chat-formatted prompts.
        samples (int): Completions per prompt.
        max_new_tokens (int): Generation budget per completion.

    Returns:
        completions (list[Tensor]): Completion token ids, ``samples`` consecutive
                            entries per text, in the order of ``texts``.
        prompt_lens (list[int]): Prompt length in tokens (without padding) per text.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        oom = True
    else:
        oom = False

    if oom:
        # Retry outside the except block so the failed attempt's tensors can be freed.
        del inputs
        torch.cuda.empty_cache()
        half = len(texts) // 2
        print(f"OOM with {len(texts)} prompts, retrying as {half} + {len(texts) - half}")
        first, first_lens = _generate_completions(
            model, tokenizer, texts[:half], samples, max_new_tokens
        )
        second, second_lens = _generate_completions(
            model, tokenizer, texts[half:], samples, max_new_tokens
        )
        return first + second, first_lens + second_lens

    # Left padding: every prompt ends at the same column, completions start after it.
    padded_len = inputs["input_ids"].shape[1]
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    return list(generated[:, padded_len:]), prompt_lens

def generate_traces(
    dataset: Any,
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    *,
    batch_size: int = 1,
    teacher_uses_rule_context: bool = True,
    log_path: Path | None = None,
) -> list[dict]:
    """
    Generates traces using the teacher model.

    Args:
        dataset (Dataset): Indexable question dataset.
        teacher (str): Teacher model name.
        samples (int): Number of samples generated per example.
        max_new_tokens (int): Maximun number of tokens added to the answer through 
                            traces.
        verifier (Verifier): Verifier to delete bad examples.
        batch_size (int): Problems per ``generate`` call; each one yields
                            ``samples`` sequences, so the GPU decodes
                            ``batch_size * samples`` sequences at once.
        teacher_uses_rule_context (bool): Prepend each row's ``rule_context``
                            to the teacher prompt only.
        log_path (Path | None): Per-example metrics JSONL. ``None`` disables it.

    Returns:
        rows (list[dicts]): New dataset made of formated json traces.
    """

    model, tokenizer = _load_model(teacher)
    # Batched generation needs left padding so every completion starts at the same column.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict] = []
    kept = 0
    progress = tqdm(total=len(dataset), desc="distill", unit="example")

    def stage(name: str) -> None:
        progress.set_description(f"distill [{name:<10}]")

    for batch_start in range(0, len(dataset), batch_size):
        batch = [
            dataset[k]
            for k in range(batch_start, min(batch_start + batch_size, len(dataset)))
        ]

        stage("prepending")
        prompts = [
            _prepend_rules(example)
            if teacher_uses_rule_context and example.get("rule_context")
            else example["prompt"]
            for example in batch
        ]
        stage("tokenizing")
        texts = [
            tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
        stage("generating")
        start = time.perf_counter()
        completions, prompt_lens = _generate_completions(
            model, tokenizer, texts, samples, max_new_tokens
        )
        gen_seconds = time.perf_counter() - start

        stage("verifying")
        batch_verified: list[int] = []
        for b, example in enumerate(batch):
            i = batch_start + b
            n_verified = 0
            for j in range(samples):
                completion_ids = completions[b * samples + j]
                raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
                trace = _canonical_trace(raw)
                result = verifier.verify(trace, str(example["answer"]))
                n_verified += result.is_correct
                n_tokens = int((completion_ids != tokenizer.pad_token_id).sum())
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
            batch_verified.append(n_verified)

        kept += sum(batch_verified)
        progress.update(len(batch))
        progress.set_postfix(
            verified=f"{kept}/{len(rows)}", accept=f"{100 * kept / len(rows):.0f}%"
        )

        if log_path is not None:
            stage("logging")
            for b, example in enumerate(batch):
                _append_jsonl(
                    log_path,
                    {
                        "example_id": batch_start + b,
                        "template_id": example.get("template_id"),
                        "branches": example.get("branches"),
                        "n_samples": samples,
                        "n_verified": batch_verified[b],
                        "prompt_tokens": prompt_lens[b],
                        "batch_size": len(batch),
                        # Whole batch time spread evenly: comparable across batch sizes.
                        "gen_seconds": round(gen_seconds / len(batch), 2),
                    },
                )
    progress.close()
    return rows


def main() -> None:
    # HF_TOKEN (and friends) from the repo's .env; variables already set in the shell win.
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="problems per generate call (x --samples sequences on the GPU at once)",
    )
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
        batch_size=args.batch_size,
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
