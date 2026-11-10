"""Dataset helpers for phase 1.

Domain JSONL keeps the ARCA contract: at least ``question`` and ``answer``.
Extra columns (``params``, ``branches``, ``rule_context``...) are intentionally
preserved so distillation/reward code can use them.
"""

from __future__ import annotations

from functools import lru_cache
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
    "Eres un asistente especializado en el Ingreso Mínimo Vital (IMV) y el "
    "Complemento de Ayuda para la Infancia (CAPI) de España. "
    "Resuelve cada caso a partir de los hechos proporcionados, sin inventar datos. "
    "Escribe el razonamiento en español dentro de <think>...</think> y la respuesta "
    "final dentro de <answer>...</answer>. "
    "En <answer> incluye únicamente el importe mensual total de IMV+CAPI en euros, "
    "con exactamente dos decimales y punto decimal, sin símbolo de moneda ni texto "
    "adicional, entre las etiquetas de apertura <answer> y cierre </answer>. "
    "Escribe un razonamiento breve y pertinente, y termina con la respuesta. "
    "No añadas texto fuera de esas etiquetas."
)


@lru_cache(maxsize=1)
def get_imv_rule_context() -> str:
    """Same versioned rule sheet used by the deterministic oracle and teacher."""
    from rlm.generate_problems import build_rule_context
    from rlm.imv_engine import load_ruleset

    return build_rule_context(load_ruleset())


def build_prompt(
    question: str, system_prompt: str = IMV_SYSTEM_PROMPT,
    *, rule_context: str | None = None,
) -> list[dict]:
    """If supplied, prepend the *exact* teacher rule sheet to the user's case."""
    user_content = f"{rule_context}\n\n{question}" if rule_context else question
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
    path: str | Path, system_prompt: str = IMV_SYSTEM_PROMPT,
    *, include_rules: bool = True,
) -> Dataset:
    dataset = load_dataset("json", data_files=str(path), split="train")
    if "question" not in dataset.column_names or "answer" not in dataset.column_names:
        raise ValueError("The domain dataset needs at least 'question' and 'answer' fields.")
    return dataset.map(
        lambda ex: {
            "prompt": build_prompt(
                ex["question"], system_prompt,
                rule_context=(ex.get("rule_context") or get_imv_rule_context())
                if include_rules else None,
            ),
            "answer": str(ex["answer"]),
        },
        remove_columns=["question"],
    )
