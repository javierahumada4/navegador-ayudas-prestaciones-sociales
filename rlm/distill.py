"""Generate verified IMV reasoning traces for cold-start SFT.

    Inspired in distilation methods like Sky-T1, OpenThoughts and DeepSeek's cold start.
    We take a strong model that writes solutions with visible reasoning, a verifier throws away the wrong ones,
    and what survives becomes SFT data. 
    Here the teacher is any model that can think in the
    ``<think>…</think><answer>…</answer>`` format 

    Run::

        uv run --extra distill python -m rlm.distill --data rlm/data/train.jsonl \
            --teacher Qwen/Qwen3-4B --samples 4 --output rlm/data/sft_traces.jsonl

    Backends: ``vllm`` (default, CUDA graphs + paged KV cache, much faster) or ``hf``
    (plain ``transformers.generate``, kept as a fallback).

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
import re
import time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any
import torch
from tqdm.auto import tqdm
from dotenv import load_dotenv

from rlm.data import load_domain_dataset
from rlm.rewards import extract_answer, has_valid_format, normalize_number
from rlm.verifier import Verifier, build_verifier

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from vllm import LLM

# Problems per generate call when --batch-size is not given. vLLM schedules the
# sequences itself, so it takes big batches; HF keeps every sequence's KV cache
# in memory at once, so 4 problems (16 sequences) is its ceiling on 16 GB.
DEFAULT_BATCH_SIZE = {"vllm": 64, "hf": 4}

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from vllm import LLM

# Problems per generate call when --batch-size is not given. vLLM schedules the
# sequences itself, so it takes big batches; HF keeps every sequence's KV cache
# in memory at once, so 4 problems (16 sequences) is its ceiling on 16 GB.
DEFAULT_BATCH_SIZE = {"vllm": 64, "hf": 4}

# Teacher-only system prompt for Qwen3 thinking mode. The student's R1-Zero prompt
# asks the model to write <think>/<answer> tags itself; a thinking model then opens
# its native <think> AND writes ours, which duplicated tags. Here the reasoning stays
# in the native block and the visible reply is only the answer tag.
TEACHER_THINKING_SYSTEM_PROMPT = (
    "Eres un experto en el Ingreso Mínimo Vital (IMV) y el Complemento de Ayuda para "
    "la Infancia (CAPI) de España. Razona paso a paso en tu bloque de pensamiento, "
    "aplicando exactamente las reglas que te dan. Cuando termines de pensar, tu "
    "respuesta visible debe ser ÚNICAMENTE <answer>IMPORTE</answer>, con el total "
    "mensual IMV+CAPI en euros, dos decimales y punto decimal, por ejemplo "
    "<answer>431.26</answer>. No escribas etiquetas <think> tú mismo."
)

# Without native thinking the teacher follows the R1-Zero format, but may answer with
# a bare number; starting its reply inside <think> makes it reason first.
NO_THINKING_PREFILL = "<think>\n"

_THINK_TAG = re.compile(r"</?think>")
_ANSWER_TAG = re.compile(r"</?answer>")
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _structure_trace(raw: str) -> tuple[str | None, str]:
    """
    Rebuild a teacher completion into the one canonical training format.

    The reasoning is everything before the first ``</think>`` (the end of the
    teacher's thinking) with any stray tags removed; the answer comes from the
    ``<answer>`` blocks after it. The result is always exactly
    ``<think>\\n{reasoning}\\n</think>\\n<answer>{amount}</answer>``, or ``None``.

    Args:
        raw (str): Teacher completion as decoded.

    Returns:
        trace (str | None): Canonical trace, ``None`` if it cannot be built.
        status (str): ``clean`` (already canonical up to whitespace), ``repaired``
                            (duplicated tags or extra text removed), or why it
                            was rejected: ``unclosed_think``, ``empty_reasoning``,
                            ``no_answer``, ``ambiguous_answer``.
    """
    if "</think>" not in raw:
        return None, "unclosed_think"  # truncated, or never reasoned
    head, tail = raw.split("</think>", 1)
    reasoning = _ANSWER_TAG.sub("", _THINK_TAG.sub("", head)).strip()
    if not reasoning:
        return None, "empty_reasoning"

    answers = _ANSWER_BLOCK.findall(tail)
    if not answers:
        return None, "no_answer"
    amounts = {normalize_number(answer) for answer in answers}
    if len(amounts) != 1 or None in amounts:
        return None, "ambiguous_answer"  # several different amounts: no hedging
    try:
        amount = Decimal(amounts.pop()).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None, "no_answer"

    trace = f"<think>\n{reasoning}\n</think>\n<answer>{amount}</answer>"
    assert has_valid_format(trace), trace[-200:]
    leftover = _THINK_TAG.sub("", _ANSWER_BLOCK.sub("", tail)).strip()
    clean = (
        head.count("<think>") <= 1
        and _ANSWER_TAG.search(head) is None
        and len(answers) == 1
        and "<think>" not in tail
        and "</think>" not in tail
        and not leftover
    )
    return trace, "clean" if clean else "repaired"

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

def _load_vllm(
    teacher: str, gpu_memory_utilization: float, max_model_len: int
) -> tuple[LLM, PreTrainedTokenizerBase]:
    """
    Load the teacher as a vLLM engine.

    Args:
        teacher (str): Teacher model name.
        gpu_memory_utilization (float): Fraction of the GPU vLLM may take
                            (weights + KV cache). The MIG slice is the whole GPU.
        max_model_len (int): Longest prompt + completion vLLM must fit.

    Returns:
        llm (LLM): vLLM engine ready to generate.
        tokenizer (PreTrainedTokenizerBase): Tokenizer of the teacher.
    """
    # CUDA is already initialized in this process by the time vLLM starts its engine
    # process, and a forked child cannot re-initialize it: start the child with spawn.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    llm = LLM(
        model=teacher,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=0,
    )
    return llm, llm.get_tokenizer()

def _append_jsonl(path: Path, record: dict) -> None:
    """
    Append one record to a JSON Lines file, flushing it to disk immediately.

    Args:
        path (Path): JSONL file.
        record (dict): Record to write as one line.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def _teacher_prompt(
    example: dict, use_rule_context: bool, thinking: bool
) -> list[dict]:
    """
    Build the teacher's messages: the row's prompt, with the rules prepended and,
    in thinking mode, the teacher-only system prompt.

    Args:
        example: Example taken from the dataset.
        use_rule_context (bool): Prepend ``rule_context`` if the row has one.
        thinking (bool): Qwen3 native thinking mode.

    Returns:
        prompt (List[dict]): Teacher messages; the row itself is not modified.
    """
    if use_rule_context and example.get("rule_context"):
        prompt = _prepend_rules(example)
    else:
        prompt = [dict(message) for message in example["prompt"]]
    if thinking:
        for message in prompt:
            if message.get("role") == "system":
                message["content"] = TEACHER_THINKING_SYSTEM_PROMPT
                break
    return prompt

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

def _generate_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    samples: int,
    max_new_tokens: int,
) -> tuple[list[dict], list[int]]:
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
        completions (list[dict]): ``text``, ``n_tokens`` and ``hit_max`` per
                            completion, ``samples`` consecutive entries per text,
                            in the order of ``texts``.
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
        first, first_lens = _generate_hf(
            model, tokenizer, texts[:half], samples, max_new_tokens
        )
        second, second_lens = _generate_hf(
            model, tokenizer, texts[half:], samples, max_new_tokens
        )
        return first + second, first_lens + second_lens

    # Left padding: every prompt ends at the same column, completions start after it.
    padded_len = inputs["input_ids"].shape[1]
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    completions = []
    for completion_ids in generated[:, padded_len:]:
        n_tokens = int((completion_ids != tokenizer.pad_token_id).sum())
        completions.append(
            {
                "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "n_tokens": n_tokens,
                "hit_max": n_tokens >= max_new_tokens,
            }
        )
    return completions, prompt_lens

def _generate_vllm(
    llm: LLM,
    texts: list[str],
    samples: int,
    max_new_tokens: int,
) -> tuple[list[dict], list[int]]:
    """
    Generate ``samples`` completions for each text with vLLM.

    Same sampling as the HF backend (temperature 0.7, top-p 0.95) and same output
    layout, so the rest of the pipeline does not know which backend ran.

    Args:
        llm (LLM): vLLM engine.
        texts (list[str]): Chat-formatted prompts.
        samples (int): Completions per prompt.
        max_new_tokens (int): Generation budget per completion.

    Returns:
        completions (list[dict]): ``text``, ``n_tokens`` and ``hit_max`` per
                            completion, ``samples`` consecutive entries per text.
        prompt_lens (list[int]): Prompt length in tokens per text.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        n=samples, temperature=0.7, top_p=0.95, max_tokens=max_new_tokens
    )
    outputs = llm.generate(texts, params, use_tqdm=False)
    completions = [
        {
            "text": completion.text,
            "n_tokens": len(completion.token_ids),
            "hit_max": completion.finish_reason == "length",
        }
        for output in outputs
        for completion in output.outputs
    ]
    return completions, [len(output.prompt_token_ids) for output in outputs]

def generate_traces(
    dataset: Any,
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    *,
    backend: str = "vllm",
    batch_size: int | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    teacher_uses_rule_context: bool = True,
    thinking: bool = True,
    log_path: Path | None = None,
    id_offset: int = 0,
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
        backend (str): ``"vllm"`` or ``"hf"``.
        batch_size (int | None): Problems per ``generate`` call, each yielding
                            ``samples`` sequences. ``None`` uses
                            ``DEFAULT_BATCH_SIZE[backend]``.
        gpu_memory_utilization (float): vLLM only, fraction of the GPU it may take.
        max_model_len (int | None): vLLM only, longest prompt + completion.
                            ``None`` means ``max_new_tokens + 3072`` (the
                            prompt with the exact rule sheet is ~2200 tokens).
        teacher_uses_rule_context (bool): Prepend each row's ``rule_context``
                            to the teacher prompt only.
        thinking (bool): Qwen3 native thinking mode, with the teacher-only
                            system prompt. ``False`` passes ``enable_thinking=False``
                            and prefills ``<think>``, so the teacher reasons in the
                            R1-Zero format of the row's system prompt.
        log_path (Path | None): Per-example metrics JSONL. ``None`` disables it.
        id_offset (int): Added to every ``example_id`` so ids stay the row
                            number in the full file when ``dataset`` is a shard.

    Returns:
        rows (list[dicts]): New dataset made of formated json traces.
    """

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[backend]

    if backend == "vllm":
        llm, tokenizer = _load_vllm(
            teacher, gpu_memory_utilization, max_model_len or max_new_tokens + 3072
        )

        def generate(texts: list[str]) -> tuple[list[dict], list[int]]:
            return _generate_vllm(llm, texts, samples, max_new_tokens)

    elif backend == "hf":
        model, tokenizer = _load_model(teacher)
        # Batched generation needs left padding so every completion starts at the same column.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def generate(texts: list[str]) -> tuple[list[dict], list[int]]:
            return _generate_hf(model, tokenizer, texts, samples, max_new_tokens)

    else:
        raise ValueError(f"Unknown backend {backend!r}: use 'vllm' or 'hf'.")

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
            _teacher_prompt(example, teacher_uses_rule_context, thinking)
            for example in batch
        ]
        stage("tokenizing")
        texts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                # Templates without this switch simply ignore it.
                enable_thinking=thinking,
            )
            + ("" if thinking else NO_THINKING_PREFILL)
            for prompt in prompts
        ]
        stage("generating")
        start = time.perf_counter()
        completions, prompt_lens = generate(texts)
        gen_seconds = time.perf_counter() - start

        stage("verifying")
        batch_verified: list[int] = []
        for b, example in enumerate(batch):
            i = id_offset + batch_start + b
            n_verified = 0
            for j in range(samples):
                completion = completions[b * samples + j]
                # The prefill is part of the prompt, not of the completion.
                raw = ("" if thinking else NO_THINKING_PREFILL) + completion["text"]
                trace, format_status = _structure_trace(raw)
                is_correct = (
                    trace is not None
                    and verifier.verify(trace, str(example["answer"])).is_correct
                )
                n_verified += is_correct
                rows.append(
                    {
                        "example_id": i,
                        "sample_idx": j,
                        "question": example["prompt"][-1]["content"],
                        "answer": str(example["answer"]),
                        "predicted": extract_answer(trace or raw),
                        "trace": trace,
                        "verified": is_correct,
                        "format_status": format_status,
                        "raw_valid_format": has_valid_format(raw),
                        "raw": raw,
                        "n_tokens": completion["n_tokens"],
                        "hit_max_tokens": completion["hit_max"],
                        "teacher": teacher,
                        "thinking": thinking,
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
                        "example_id": id_offset + batch_start + b,
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
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="problems per generate call (default: 64 for vllm, 4 for hf)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vllm: fraction of the GPU (MIG slice) for weights + KV cache",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vllm: longest prompt + completion (default: max-new-tokens + 3072)",
    )
    parser.add_argument("--verifier", default="imv")
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    parser.add_argument(
        "--log-path",
        default="rlm/data/logs/distill_metrics.jsonl",
        help="per-example metrics JSONL, overwritten at the start of each run",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="i/n: only the i-th of n contiguous slices of --data (1-based), so "
        "several sessions can split one run; outputs concatenate",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable Qwen3 native thinking; the teacher follows our <think><answer> format",
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
    id_offset = 0
    if args.shard:
        index, count = (int(part) for part in args.shard.split("/"))
        if not 1 <= index <= count:
            raise ValueError(f"--shard {args.shard}: need 1 <= i <= n")
        id_offset = (index - 1) * len(dataset) // count
        end = index * len(dataset) // count
        dataset = dataset.select(range(id_offset, end))
        print(f"Shard {args.shard}: rows {id_offset}..{end - 1} of --data")
    traces = generate_traces(
        dataset,
        args.teacher,
        args.samples,
        args.max_new_tokens,
        build_verifier(args.verifier),
        backend=args.backend,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        teacher_uses_rule_context=not args.no_teacher_rules,
        thinking=not args.no_thinking,
        log_path=log_path,
        id_offset=id_offset,
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
