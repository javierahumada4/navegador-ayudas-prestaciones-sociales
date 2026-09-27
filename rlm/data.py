"""Dataset helpers for phase 1.

Domain JSONL keeps the ARCA contract: at least ``question`` and ``answer``.
Extra columns (``params``, ``branches``, ``rule_context``...) are intentionally
preserved so distillation/reward code can use them.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset


R1_ZERO_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

IMV_SYSTEM_PROMPT = (
    R1_ZERO_SYSTEM_PROMPT
    + " The domain is Spain's Ingreso Mínimo Vital (IMV) and CAPI. "
    "Reason carefully from the facts in the problem. The final <answer> must contain "
    "only the total monthly amount IMV+CAPI in euros, with exactly two decimal places "
    "and a decimal point, for example <answer>431.26</answer>."
)


def build_prompt(question: str, system_prompt: str = IMV_SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def gsm8k_final_answer(solution: str) -> str:
    return solution.split("####")[-1].strip().replace(",", "")


def load_gsm8k(split: str = "train", n_examples: int | None = None, seed: int = 0) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if n_examples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"], R1_ZERO_SYSTEM_PROMPT),
            "answer": gsm8k_final_answer(ex["answer"]),
        },
        remove_columns=dataset.column_names,
    )


def load_domain_dataset(
    path: str | Path, system_prompt: str = IMV_SYSTEM_PROMPT
) -> Dataset:
    dataset = load_dataset("json", data_files=str(path), split="train")
    if "question" not in dataset.column_names or "answer" not in dataset.column_names:
        raise ValueError("The domain dataset needs at least 'question' and 'answer' fields.")
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"], system_prompt),
            "answer": str(ex["answer"]),
        },
        remove_columns=["question"],
    )
