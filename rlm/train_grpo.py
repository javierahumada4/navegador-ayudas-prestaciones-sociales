"""Phase 1 GRPO training for the IMV/CAPI reasoning task."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.rewards import _completion_text, accuracy_reward, extract_answer, format_reward

IMV_FINAL_ANSWER = re.compile(r"^\d+(?:\.\d{2})$")


def domain_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """Reward the user-facing IMV answer contract.

    Accuracy already checks the amount. This third reward teaches the model to
    return a clean amount with exactly two decimals in the <answer> block, which
    is the output expected by the navigator and makes verification unambiguous.
    """
    rewards: list[float] = []
    for completion in completions:
        answer = extract_answer(_completion_text(completion))
        rewards.append(1.0 if answer and IMV_FINAL_ANSWER.fullmatch(answer.strip()) else 0.0)
    return rewards


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.data == "gsm8k":
        dataset = load_gsm8k("train", n_examples=args.n_examples, seed=args.seed)
    else:
        dataset = load_domain_dataset(args.data)
    print(f"{len(dataset)} training problems")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=0.2,
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        logging_steps=1,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=2,
        reward_weights=[1.0, 2.0, 0.5],
        model_init_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32},
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
        reward_funcs=[format_reward, accuracy_reward, domain_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    print(f"final adapter saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="rlm/data/train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--output", default="rlm/weights/final_rlm_lora")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
