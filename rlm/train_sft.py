"""Phase 1, step 2: cold-start supervised fine-tuning on verified reasoning traces.

This is the "SFT round 1" box of the DeepSeek pipeline. The input is the JSONL produced
by ``rlm/distill.py``: problems from your domain with a teacher-generated reasoning trace
that *passed your verifier*. The output is a LoRA adapter that already speaks the
``<think>…</think><answer>…</answer>`` format before we ever run GRPO.

Run::

    uv run python -m rlm.train_sft --data rlm/data/sft_traces.jsonl --output rlm/weights/sft_lora

What is decided for you: the trainer (``trl.SFTTrainer``), LoRA, the conversational
format (``prompt`` + ``completion`` columns, so that the loss is computed only on the
assistant turn). What you decide: the data, the base model, epochs, learning rate,
sequence length. Write down in EXPERIMENTS.md why you chose them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from rlm.data import build_prompt, get_imv_rule_context
from rlm.sft_selection import select_budgeted_traces


def select_verified_traces(dataset, max_per_problem: int):
    """Keep at most N short, distinct verified traces per question."""
    if max_per_problem <= 0:
        raise ValueError("max_per_problem must be positive")
    if "verified" in dataset.column_names:
        dataset = dataset.filter(lambda ex: bool(ex["verified"]))
    groups: dict[str, list[int]] = {}
    for idx, example in enumerate(dataset):
        groups.setdefault(example["question"], []).append(idx)
    selected = set()
    for indices in groups.values():
        ranked = sorted(indices, key=lambda i: (len(dataset[i]["trace"]), i))
        seen_traces: set[str] = set()
        for idx in ranked:
            trace = dataset[idx]["trace"]
            if trace in seen_traces:
                continue
            selected.add(idx)
            seen_traces.add(trace)
            if len(seen_traces) >= max_per_problem:
                break
    return dataset.select(sorted(selected))


def load_sft_dataset(path: str | Path, max_traces_per_problem: int = 2):
    """Turn distilled traces into TRL's prompt/completion conversational format.

    Expected JSONL fields: ``question``, ``trace`` (the full ``<think>…</think><answer>…</answer>``
    text) and ``answer``. Only traces marked ``verified: true`` are used.
    """
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = select_verified_traces(dataset, max_traces_per_problem)
    return format_sft_dataset(dataset)


def format_sft_dataset(dataset):
    """Format already selected raw teacher rows as conversational SFT examples."""
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(
                ex["question"],
                rule_context=ex.get("rule_context") or get_imv_rule_context(),
            ),
            "completion": [{
                "role": "assistant",
                "content": ex["trace"],
            }],
        },
        remove_columns=dataset.column_names,
    )


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw = load_dataset("json", data_files=str(args.data), split="train")
    report_file = Path(args.selection_report or Path(args.output).parent / "sft_selection.json")
    selected, coverage = select_budgeted_traces(
        raw, tokenizer,
        max_completion_tokens=args.max_completion_tokens,
        max_length=args.max_length,
        max_per_problem=args.max_traces_per_problem,
        min_problem_coverage=args.min_problem_coverage,
        min_families=args.min_families,
        report_path=report_file,
        warn_low_coverage=args.warn_low_coverage,
    )
    dataset = format_sft_dataset(selected)
    print(
        f"SFT selection: {len(dataset)} verified complete traces, "
        f"{coverage['covered_problems']}/{coverage['total_problems']} problems "
        f"({coverage['problem_coverage']:.1%}); "
        f"excluded={coverage['excluded_by_reason']}", flush=True,
    )
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eos_token=tokenizer.eos_token,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        # Completion-only loss uses the prompt/completion structure above.
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    print(f"adapter saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", required=True, help="JSONL with question / trace / answer / verified"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="rlm/weights/sft_lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192,
                        help="maximum total prompt + completion tokens")
    parser.add_argument("--max-completion-tokens", type=int, default=1024,
                        help="maximum SFT assistant completion tokens, shared with GRPO")
    parser.add_argument("--min-problem-coverage", type=float, default=0.70)
    parser.add_argument("--min-families", type=int, default=5)
    parser.add_argument("--warn-low-coverage", action="store_true",
                        help="continue SFT despite low problem/family coverage")
    parser.add_argument(
        "--max-traces-per-problem", type=int, default=2,
        help="maximum unique verified teacher traces used per problem in SFT",
    )
    parser.add_argument(
        "--selection-report", default=None,
        help="JSON report of verified, complete traces selected for SFT",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", default=None, help="checkpoint-XXX folder")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
