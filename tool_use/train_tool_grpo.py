"""Continue SFT+reasoning-GRPO with GRPO specialised in calcular_imv tool calls.

This is an *agent/tool-use* rollout: TRL executes calcular_imv in the rollout,
reinserts the observation and generates the final assistant response. Unlike a
single-shot imitation objective, rewards depend on Pydantic-valid arguments,
the deterministic result, and the final numeric answer.

No test cases or answers are placed in prompts; reference values are available
only to reward functions. GPU training must be run on the DGX.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tool_use.evaluate_shared import load_shared_test, prompt_for_case
from tool_use.executor import SYSTEM_PROMPT_TEMPLATE
from tool_use.imv_tools import IMVCase
from tool_use.tool_grpo_rewards import (
    final_answer_reward, numeric_format_reward, tool_arguments_reward,
    tool_execution_reward, tool_selection_reward,
)
from tool_use.tools import default_registry

DEFAULT_WEIGHTS = [0.25, 1.0, 1.0, 2.0, 0.25]
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_ADAPTER = "${HOME}/clusters/dgx/arca-runs/rlm-final-v3/weights/grpo_lora"


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_train_test_disjoint(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    train_questions = {r["question"] for r in train}
    overlapping = sum(r["question"] in train_questions for r in test)
    if overlapping:
        raise ValueError(
            f"DATA LEAKAGE: {overlapping} test questions overlap the training set"
        )


def resolve_init_adapter(value: str, model: str) -> str:
    """Require a real final PEFT adapter compatible with the chosen base model."""
    adapter = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    conf_path = adapter / "adapter_config.json"
    if not conf_path.is_file():
        raise FileNotFoundError(
            f"Missing SFT+reasoning-GRPO adapter_config.json at {adapter}"
        )
    info = json.loads(conf_path.read_text(encoding="utf-8"))
    trained_base = info.get("base_model_name_or_path")
    if trained_base and trained_base != model:
        raise ValueError(
            f"Adapter base {trained_base!r} differs from --model {model!r}"
        )
    return str(adapter)



def calcular_imv(case: dict) -> dict:
    """Calculate monthly IMV+CAPI using the deterministic Spanish 2026 engine.

    The argument is a full synthetic benchmark case with application_date,
    applicant_id, persons (id, birth date, legal and effective residence),
    domicile (co_resident_ids, same_domicile_since), economic (annual income,
    net worth, other assets), and optional relationships and exceptions.
    Unmentioned optional facts use the simulator's neutral defaults. Use the
    same facts as the problem, without inventing different incomes or dates.

    Args:
        case: Structured IMV case JSON object, exactly as for calcular_imv at
            inference time. Requires application_date, applicant_id, persons,
            domicile, and economic with their required nested fields.

    Returns:
        Deterministic IMV/CAPI eligibility and total_monthly_eur.
    """
    record = default_registry().call("calcular_imv", {"case": case})
    if not record.ok:
        raise ValueError(record.error)
    return record.result


def prepare_training_examples(
    train_file: str | Path,
    test_file: str | Path,
    n_examples: int | None,
) -> list[dict[str, Any]]:
    """Build TRL conversational examples; hide oracle params from the model."""
    all_train = load_shared_test(train_file)
    test_cases = load_shared_test(test_file)
    check_train_test_disjoint(all_train, test_cases)
    if not all_train:
        raise ValueError("Training dataset is empty")
    if n_examples is not None and n_examples < 1:
        raise ValueError("--n-examples must be positive")
    samples = all_train if n_examples is None else all_train[:n_examples]
    prepared = []
    for i, row in enumerate(samples):
        if not isinstance(row.get("params"), dict):
            raise ValueError(f"Train row {i} lacks oracle params")
        # The generator includes private labels like _family, not part of IMVCase.
        facts = {key: val for key, val in row["params"].items()
                 if not key.startswith("_")}
        try:
            ref = IMVCase.model_validate(facts).model_dump(
                mode="json", by_alias=True
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid reference IMV case in train row {i}: {exc}") from exc
        prepared.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": prompt_for_case(row)},
            ],
            "answer": str(row["answer"]),
            "reference_case": json.dumps(ref, ensure_ascii=False),
        })
    return prepared


def training_metadata(
    *, base_model: str, train_file: str, test_file: str,
    adapter: str | None, steps: int, n_examples: int, max_completion_length: int,
    num_generations: int, seed: int, reward_weights: list[float],
) -> dict[str, Any]:
    return {
        "method": "trl.GRPOTrainer with tools=[calcular_imv]",
        "source": "sft_grpo",
        "base_model": base_model,
        "initial_adapter": adapter,
        "train_file": train_file,
        "train_sha256": file_hash(train_file),
        "heldout_file": test_file,
        "heldout_sha256": file_hash(test_file),
        "n_train_examples": n_examples,
        "steps": steps,
        "max_completion_length": max_completion_length,
        "num_generations": num_generations,
        "seed": seed,
        "reward_weights": reward_weights,
        "thinking": False,
        "max_tool_calling_iterations": 2,
        "note": "No checkpoint is implied by --dry-run; only trainer.save_model creates weights.",
    }


def preflight_trl_tools() -> None:
    """Fail before loading a 1.7B model if installed versions lack agent GRPO."""
    from transformers.utils import get_json_schema
    from trl import GRPOConfig, GRPOTrainer

    config_params = inspect.signature(GRPOConfig).parameters
    if "max_tool_calling_iterations" not in config_params:
        raise RuntimeError(
            "Installed trl.GRPOConfig lacks max_tool_calling_iterations. "
            "Upgrade to a release supporting multi-turn tools."
        )
    if "chat_template_kwargs" not in config_params:
        raise RuntimeError("Installed GRPOConfig lacks chat_template_kwargs")
    if "tools" not in inspect.signature(GRPOTrainer).parameters:
        raise RuntimeError("Installed GRPOTrainer lacks tools= agent rollouts")
    schema = get_json_schema(calcular_imv)
    if schema["function"]["name"] != "calcular_imv":
        raise RuntimeError("TRL training tool name differs from inference tool")
    if "case" not in schema["function"]["parameters"].get("properties", {}):
        raise RuntimeError("Tool has no case field in its schema")


def train(args: argparse.Namespace, prepared: list[dict[str, Any]], adapter: str | None) -> None:
    """Run the actual GPU GRPO training using TRL's native multi-turn tool loop."""
    preflight_trl_tools()
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from rlm.train_grpo import ZeroSignalMonitor, save_log_history

    base_model = args.model
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Some Qwen thinking adapters come from phase 1. For a controlled
    # comparison the tool-use training and ToolLoop inference both disable
    # thinking in the native Qwen tool-calling chat template.
    data = Dataset.from_list(prepared)
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if cuda else torch.float32
    )
    initial = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(initial, adapter, is_trainable=True)
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=(
            args.num_generations * args.grad_accum // args.micro_batch_size
        ),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=2,
        chat_template_kwargs={"enable_thinking": False},
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        epsilon=args.epsilon,
        bf16=cuda and dtype == torch.bfloat16,
        fp16=cuda and dtype == torch.float16,
        gradient_checkpointing=cuda,
        logging_steps=1,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=2,
        model_init_kwargs={"dtype": dtype},
        reward_weights=args.reward_weights,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=data,
        tools=[calcular_imv],
        peft_config=None,
        reward_funcs=[
            tool_selection_reward, tool_arguments_reward,
            tool_execution_reward, final_answer_reward, numeric_format_reward,
        ],
    )

    class AbortOnZeroSignal(TrainerCallback):
        def __init__(self, patience: int):
            self.monitor = ZeroSignalMonitor(patience)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if self.monitor.observe(state.global_step, logs):
                raise RuntimeError(
                    "Tool-use GRPO rewards have zero variance for "
                    f"{self.monitor.streak} logged steps. Inspect sampled calls, "
                    "JSON validity and reward logs before further training."
                )

    trainer.add_callback(AbortOnZeroSignal(args.zero_signal_patience))
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    finally:
        save_log_history(trainer, args.output)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    output = Path(args.output)
    if not (output / "adapter_config.json").is_file():
        raise RuntimeError("Training ended without a final PEFT adapter_config.json")
    print(f"Tool-use GRPO adapter saved: {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter", default=DEFAULT_ADAPTER,
                        help="Final LoRA of SFT followed by reasoning GRPO")
    parser.add_argument("--train-data", default="results/rlm_final/data/train.jsonl")
    parser.add_argument("--test-data", default="results/rlm_final/data/test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--zero-signal-patience", type=int, default=5)
    parser.add_argument("--reward-weights", type=float, nargs=5, default=DEFAULT_WEIGHTS)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--n-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="Verify installed TRL/Transformers tool API without loading model weights")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_generations < 2 or args.grad_accum < 1:
        parser.error("steps >= 1, num_generations >= 2, grad_accum >= 1 required")
    if args.micro_batch_size < 1 or (
        args.num_generations * args.grad_accum
    ) % args.micro_batch_size:
        parser.error("micro-batch-size must divide num-generations * grad-accum")
    if args.max_completion_length < 1:
        parser.error("max-completion-length must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("temperature > 0, top-p in (0,1], top-k >= 0 required")
    if args.learning_rate <= 0 or args.zero_signal_patience < 0:
        parser.error("learning-rate must be positive; zero-signal-patience >= 0")
    if any(weight < 0 for weight in args.reward_weights):
        parser.error("reward weights must be nonnegative")
    if not Path(args.train_data).is_file() or not Path(args.test_data).is_file():
        parser.error("Train and held-out test JSONL must exist")

    try:
        adapter = resolve_init_adapter(args.init_adapter, args.model)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    destination = Path(os.path.expandvars(os.path.expanduser(args.output))).resolve()
    source_adapter = Path(adapter)
    if destination == source_adapter or source_adapter in destination.parents:
        parser.error("Output must not overwrite the initial SFT+GRPO adapter")
    if (destination / "adapter_config.json").exists() and not args.resume_from_checkpoint:
        parser.error("Output already contains a trained adapter; choose another path")

    prepared = prepare_training_examples(
        args.train_data, args.test_data, args.n_examples
    )
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).exists():
        parser.error("resume-from-checkpoint does not exist")
    metadata = training_metadata(
        base_model=args.model,
        train_file=args.train_data, test_file=args.test_data,
        adapter=adapter, steps=args.steps, n_examples=len(prepared),
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations, seed=args.seed,
        reward_weights=args.reward_weights,
    )
    if args.dry_run or args.preflight:
        if args.preflight:
            preflight_trl_tools()
            metadata["trl_tool_preflight"] = "passed"
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "tool_grpo_run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output = str(destination)
    train(args, prepared, adapter)


if __name__ == "__main__":
    main()
