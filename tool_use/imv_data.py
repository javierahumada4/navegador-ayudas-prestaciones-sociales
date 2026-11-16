"""Shared IMV benchmark loading and training/test separation.

Only question and rule_context are provided to model inference. The oracle
answer and params are retained solely for scoring or the GRPO reward function.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_cases(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or
                    not isinstance(row.get("question"), str) or
                    not isinstance(row.get("answer"), (str, int, float))):
                raise ValueError(f"Invalid question/answer at {path}:{line_number}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No cases in {path}")
    return rows


def user_prompt(row: dict[str, Any]) -> str:
    context = row.get("rule_context")
    if not context:
        from rlm.generate_problems import build_rule_context
        from rlm.imv_engine import load_ruleset
        context = build_rule_context(load_ruleset())
    return f"{context}\n\n{row['question']}"


def assert_disjoint(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    questions = {row["question"] for row in train}
    duplicates = sum(row["question"] in questions for row in test)
    if duplicates:
        raise ValueError(f"DATA LEAKAGE: {duplicates} test questions overlap train")


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
