"""Reproducible 12-arm IMV experiment: four reasoning checkpoints x three tool modes.

The same held-out Phase-1 questions and reference amounts are used everywhere.
No ground-truth params/answers are sent to the model. Run on the DGX for GPU
inference, or --dry-run to inventory variants without loading model weights.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from rlm.rewards import (
    extract_unique_final_answer, has_valid_imv_format, imv_money,
    imv_numbers_match, parse_imv_decimal,
)
from tool_use.evaluate_shared import load_shared_test, prompt_for_case

MODEL = "Qwen/Qwen3-1.7B"
TWO_DECIMALS = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}\Z")
CHECKPOINTS = ("base", "sft", "grpo_zero", "sft_grpo")
TOOL_SUFFIXES = ("", "_tool", "_tool_grpo")
TOOL_ADAPTERS = {
    "base": "tool_grpo_base",
    "sft": "tool_grpo_sft",
    "grpo_zero": "tool_grpo_grpo_zero",
    "sft_grpo": "tool_grpo_sft_grpo",
}
ADAPTER_KEYS = CHECKPOINTS[1:] + tuple(TOOL_ADAPTERS.values())


@dataclass(frozen=True)
class Variant:
    name: str
    checkpoint: str
    with_tool: bool
    tool_grpo: bool
    adapter_key: str | None


VARIANTS = tuple(
    Variant(
        name=checkpoint + suffix,
        checkpoint=checkpoint,
        with_tool=bool(suffix),
        tool_grpo=suffix == "_tool_grpo",
        adapter_key=(
            TOOL_ADAPTERS[checkpoint] if suffix == "_tool_grpo"
            else checkpoint if checkpoint != "base" else None
        ),
    )
    for suffix in TOOL_SUFFIXES
    for checkpoint in CHECKPOINTS
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load checkpoint paths, never silently substituting one model for another."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("adapters"), dict):
        raise ValueError("Checkpoint config needs an 'adapters' object")
    unknown = set(config["adapters"]) - set(ADAPTER_KEYS)
    if unknown:
        raise ValueError(f"Unknown checkpoint keys: {sorted(unknown)}")
    return config


def resolve_adapter(variant: Variant, config: dict[str, Any]) -> str | None:
    if variant.adapter_key is None:
        return None
    value = config["adapters"].get(variant.adapter_key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing checkpoint '{variant.adapter_key}' for {variant.name}")
    adapter = Path(os.path.expandvars(os.path.expanduser(value)))
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{variant.name}: no adapter_config.json in {adapter}. "
            "Each key must reference the FINAL checkpoint for that combination."
        )
    return str(adapter)


def file_hash(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def check_test_is_held_out(test: list[dict[str, Any]], train_path: str | Path) -> None:
    train = load_shared_test(train_path)
    training_questions = {row["question"] for row in train}
    overlap = [row["question"] for row in test if row["question"] in training_questions]
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE: {len(overlap)} test questions are present in training set"
        )


def select_final_answer(raw: str) -> str | None:
    """Read ONE final value only; never take a number from a reasoning trace."""
    tagged = extract_unique_final_answer(raw)
    if tagged is not None:
        return tagged
    if any(tag in raw for tag in ("<think", "</think", "<answer", "</answer")):
        return None
    return raw.strip() or None


def absolute_error(predicted: str | None, expected: str) -> float | None:
    if predicted is None:
        return None
    # Match the verifier's single-number constraint, not a number from prose.
    from rlm.rewards import NUMBER_PATTERN

    if len(NUMBER_PATTERN.findall(predicted)) != 1:
        return None
    p, e = parse_imv_decimal(predicted), parse_imv_decimal(expected)
    if p is None or e is None:
        return None
    return float(abs(imv_money(p) - imv_money(e)))


def make_row(
    case: dict[str, Any],
    prediction: str | None,
    *,
    variant: Variant,
    sample: int,
    elapsed_ms: float,
    n_tokens: int | None,
    raw_generations: list[str],
    tool_calls: list[dict[str, Any]],
    valid_format: bool,
) -> dict[str, Any]:
    expected = str(case["answer"])
    tool_ok = any(
        x.get("name") == "calcular_imv" and x.get("ok") is True
        for x in tool_calls
    )
    return {
        "variant": variant.name,
        "question": case["question"],
        "expected": expected,
        "predicted": prediction,
        "numeric_correct": imv_numbers_match(prediction, expected),
        "absolute_error_eur": absolute_error(prediction, expected),
        "strict_format": valid_format,
        "tool_execution_ok": tool_ok if variant.with_tool else None,
        "tool_call_count": len(tool_calls) if variant.with_tool else None,
        "tool_error_count": (
            sum(not x.get("ok") for x in tool_calls) if variant.with_tool else None
        ),
        "family": (case.get("branches") or {}).get("family"),
        "sample": sample,
        "duration_ms": round(elapsed_ms, 3),
        "n_tokens": n_tokens,
        "raw_generations": raw_generations,
        "tool_calls": tool_calls,
    }


def summarize(rows: list[dict[str, Any]], variant: Variant) -> dict[str, Any]:
    n = len(rows)
    errors = [r["absolute_error_eur"] for r in rows if r["absolute_error_eur"] is not None]
    return {
        "n": n,
        "numeric_accuracy": sum(r["numeric_correct"] for r in rows) / n if n else None,
        "format_rate": sum(r["strict_format"] for r in rows) / n if n else None,
        "mae_eur": sum(errors) / len(errors) if len(errors) == n and n else None,
        "parseable_rate": len(errors) / n if n else None,
        "avg_duration_ms": sum(r["duration_ms"] for r in rows) / n if n else None,
        "avg_generated_tokens": (
            sum(r["n_tokens"] for r in rows) / n
            if n and all(r["n_tokens"] is not None for r in rows) else None
        ),
        "tool_execution_rate": (
            sum(r["tool_execution_ok"] for r in rows) / n
            if n and variant.with_tool else None
        ),
        "tool_error_rate": (
            sum(r["tool_error_count"] > 0 for r in rows) / n
            if n and variant.with_tool else None
        ),
    }


def run_variant(
    variant: Variant,
    cases: list[dict[str, Any]],
    *,
    adapter_path: str | None,
    model_id: str,
    max_new_tokens: int,
    max_turns: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load only one model at a time. Never pass oracle params to inference."""
    import torch

    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    if variant.with_tool:
        from tool_use.executor import ToolLoop

        runner = ToolLoop(
            brain="rlm" if adapter_path else "base",
            model_id=model_id, adapter_path=adapter_path,
            max_new_tokens=max_new_tokens, enable_thinking=False,
        )
    else:
        from rlm.inference import ReasoningModel

        runner = ReasoningModel(base_model=model_id, adapter_path=adapter_path)
        runner.load()

    try:
        for i, case in enumerate(cases):
            started = time.perf_counter()
            if variant.with_tool:
                trace: list[str] = []
                response = runner.run(
                    prompt_for_case(case), max_turns=max_turns, trace=trace
                )
                predicted = response.answer.strip()
                calls = [c.model_dump() for c in response.tool_calls]
                strict_format = TWO_DECIMALS.fullmatch(predicted) is not None
                # Token count computed from model outputs; not a billable API count.
                n_tokens = (
                    sum(len(runner.tokenizer.encode(g, add_special_tokens=False))
                        for g in trace)
                    if runner.tokenizer is not None else None
                )
                raw = trace
            else:
                completion, n_tokens = runner.generate(
                    case["question"], max_new_tokens=max_new_tokens,
                    do_sample=False, rule_context=case.get("rule_context"),
                )
                predicted = select_final_answer(completion)
                strict_format = has_valid_imv_format(completion)
                calls = []
                raw = [completion]
            row = make_row(
                case, predicted, variant=variant, sample=0,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                n_tokens=n_tokens, raw_generations=raw,
                tool_calls=calls, valid_format=strict_format,
            )
            row["id"] = i
            rows.append(row)
            print(
                f"[{variant.name}] {i+1}/{len(cases)} "
                f"numeric={'ok' if row['numeric_correct'] else 'fail'}",
                flush=True,
            )
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def summary_csv(path: Path, outputs: dict[str, dict[str, Any]]) -> None:
    fields = [
        "variant", "checkpoint", "with_tool", "tool_grpo", "status",
        "adapter", "n", "numeric_accuracy", "mae_eur", "parseable_rate",
        "format_rate", "tool_execution_rate", "tool_error_rate",
        "avg_duration_ms", "avg_generated_tokens", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            item = outputs[variant.name]
            writer.writerow({
                "variant": variant.name, "checkpoint": variant.checkpoint,
                "with_tool": variant.with_tool, "tool_grpo": variant.tool_grpo,
                "status": item["status"], "adapter": item.get("adapter"),
                **(item.get("summary") or {}), "error": item.get("error"),
            })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/rlm_final/data/test.jsonl")
    parser.add_argument("--train-data", default="results/rlm_final/data/train.jsonl",
                        help="Check exact question overlaps; omit with --train-data ''")
    parser.add_argument("--config", default="tool_use/bench/variants.example.json")
    parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), action="append")
    parser.add_argument("--all", action="store_true", help="Evaluate all 12 configurations")
    parser.add_argument("--out-dir", default="reports/imv_12_variants")
    parser.add_argument("--n-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Inventory adapters; no GPU")
    args = parser.parse_args(argv)
    if args.all and args.variant:
        parser.error("Choose either --all or --variant")
    if not args.all and not args.variant:
        parser.error("Provide --all or at least one --variant")
    if args.n_examples is not None and args.n_examples < 1:
        parser.error("--n-examples must be positive")
    if args.max_new_tokens < 1 or args.max_turns < 1:
        parser.error("Token and turn limits must be positive")
    data = Path(args.data)
    cases = load_shared_test(data, args.n_examples)
    if not cases:
        parser.error("Test dataset is empty")
    if args.train_data:
        check_test_is_held_out(cases, args.train_data)
    config = load_config(args.config)
    model_id = config.get("base_model", MODEL)
    if not isinstance(model_id, str) or not model_id:
        parser.error("Invalid base_model")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chosen = set(VARIANT_BY_NAME) if args.all else set(args.variant)
    outputs: dict[str, dict[str, Any]] = {}
    meta = {
        "data": str(data), "data_sha256": file_hash(data),
        "train_data": args.train_data or None,
        "train_sha256": file_hash(args.train_data) if args.train_data else None,
        "n_examples": len(cases), "base_model": model_id, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "max_turns": args.max_turns,
        "decoding": "greedy", "samples_per_problem": 1,
        "scoring": "one unambiguous final IMV+CAPI amount; HALF_UP to cents",
        "note": "Tool mode uses a different system prompt and multi-turn budget.",
    }
    for variant in VARIANTS:
        result: dict[str, Any] = {
            "status": "not_selected" if variant.name not in chosen else "pending",
            "adapter": None, "summary": None,
        }
        if variant.name in chosen:
            try:
                adapter = resolve_adapter(variant, config)
                result["adapter"] = adapter
                if args.dry_run:
                    result["status"] = "ready"
                else:
                    rows = run_variant(
                        variant, cases, adapter_path=adapter, model_id=model_id,
                        max_new_tokens=args.max_new_tokens, max_turns=args.max_turns,
                        seed=args.seed,
                    )
                    result["status"] = "completed"
                    result["summary"] = summarize(rows, variant)
                    (out / f"{variant.name}.json").write_text(
                        json.dumps(
                            {"configuration": {**meta, "variant": variant.name,
                                               "adapter": adapter},
                             "summary": result["summary"], "rows": rows},
                            indent=2, ensure_ascii=False,
                        ), encoding="utf-8",
                    )
            except (ValueError, FileNotFoundError) as exc:
                if not args.all:
                    parser.error(str(exc))
                result["status"] = "missing_checkpoint"
                result["error"] = str(exc)
        outputs[variant.name] = result
        # Incremental output survives interruption of later GPU variants.
        (out / "manifest.json").write_text(
            json.dumps({"configuration": meta, "variants": outputs},
                       indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_csv(out / "comparison.csv", outputs)
        print(f"{variant.name}: {result['status']}", flush=True)
    print(f"Results: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
