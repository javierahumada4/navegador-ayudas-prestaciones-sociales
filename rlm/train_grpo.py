"""Phase 1, step 3: GRPO for IMV/CAPI reasoning.

For IMV datasets the accuracy reward must use the same monetary semantics as the
oracle and verifier: Decimal + ROUND_HALF_UP to cents. GSM8K keeps the generic
exact numeric reward.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.rewards import (
    _completion_text,
    accuracy_reward,
    extract_answer,
    format_reward,
    imv_accuracy_reward,
    parse_imv_decimal,
)

_TWO_DECIMAL_ANSWER = re.compile(r"^\d+(?:\.\d{2})$")


def domain_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """Reward the clean machine-readable IMV answer contract: exactly `123.45`."""
    scores: list[float] = []
    for completion in completions:
        answer = extract_answer(_completion_text(completion))
        scores.append(1.0 if answer and _TWO_DECIMAL_ANSWER.fullmatch(answer.strip()) else 0.0)
    return scores


def zero_answer_rate(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """Monitor, not a reward: 1.0 when the model answers 0.

    About 30 % of the IMV problems have 0.00 as answer, so always answering zero is a
    cheap way to collect accuracy reward. It is registered with weight 0, so it never
    changes the advantages; TRL still logs its mean as ``rewards/zero_answer_rate/mean``.
    """
    rates: list[float] = []
    for completion in completions:
        value = parse_imv_decimal(extract_answer(_completion_text(completion)))
        rates.append(1.0 if value is not None and value == 0 else 0.0)
    return rates


def save_log_history(trainer, output_dir: str) -> Path:
    """Dump the per-step metrics (rewards, lengths, KL, clipping) for the training curves."""
    path = Path(output_dir) / "log_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    return path


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    is_gsm8k = args.data == "gsm8k"
    if is_gsm8k:
        dataset = load_gsm8k("train", n_examples=args.n_examples, seed=args.seed)
        reward_funcs = [format_reward, accuracy_reward]
        default_weights = [1.0, 2.0]
        monitors = []
    else:
        dataset = load_domain_dataset(args.data)
        # Critical: do not use generic accuracy_reward here. IMV money is rounded
        # HALF_UP to cents by the oracle and verifier.
        reward_funcs = [format_reward, imv_accuracy_reward, domain_reward]
        default_weights = [1.0, 2.0, 0.5]
        monitors = [zero_answer_rate]

    reward_weights = args.reward_weights or default_weights
    if len(reward_weights) != len(reward_funcs):
        names = ", ".join(f.__name__ for f in reward_funcs)
        raise ValueError(f"--reward-weights needs {len(reward_funcs)} values ({names})")
    print(
        "reward weights: "
        + ", ".join(f"{f.__name__}={w}" for f, w in zip(reward_funcs, reward_weights, strict=True))
    )
    # Monitors go last with weight 0: logged by TRL, ignored by the loss.
    reward_funcs = reward_funcs + monitors
    reward_weights = list(reward_weights) + [0.0] * len(monitors)

    print(f"{len(dataset)} training problems")
    # The loss forward materialises (batch, completion length, vocab) logits: with 150k vocab
    # and ~3k-token completions that is ~1.7 GiB per sequence in fp32. Generate the whole
    # group at once but run the loss on micro-batches, accumulating gradients over them.
    if (args.num_generations * args.grad_accum) % args.micro_batch_size:
        raise ValueError("--micro-batch-size must divide --num-generations x --grad-accum")
    accumulation = args.num_generations * args.grad_accum // args.micro_batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=args.epsilon,
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        logging_steps=1,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=2,
        model_init_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        reward_weights=reward_weights,
    )

    if args.init_adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    print(f"final adapter saved to {args.output}")
    print(f"training log -> {save_log_history(trainer, args.output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="gsm8k", help="'gsm8k' or path to domain JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--output", default="rlm/weights/final_rlm_lora")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="prompts (groups) per optimizer step"
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="completions per loss forward; lower it if the loss step runs out of memory",
    )
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO/GRPO clipping range")
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs="+",
        default=None,
        help="one weight per reward, in order: format accuracy [domain]; "
        "default 1.0 2.0 0.5 (GSM8K: 1.0 2.0)",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
