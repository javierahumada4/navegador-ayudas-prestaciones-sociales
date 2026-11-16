"""Five-arm evaluation on the same frozen Phase-1 IMV/CAPI test.

Arms: base, SFT, SFT+reasoning-GRPO, SFT+GRPO+calcular_imv,
and SFT+GRPO+tool-GRPO+calcular_imv. No model is trained here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.rewards import (
    NUMBER_PATTERN, extract_unique_final_answer, imv_money,
    imv_numbers_match, parse_imv_decimal,
)
from tool_use.imv_data import assert_disjoint, file_hash, read_cases, user_prompt

AMOUNT = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}\Z")
BASE_MODEL = "Qwen/Qwen3-1.7B"


@dataclass(frozen=True)
class Variant:
    name: str
    adapter_key: str | None
    with_tool: bool


VARIANTS = (
    Variant("base", None, False),
    Variant("sft", "sft", False),
    Variant("sft_grpo", "sft_grpo", False),
    Variant("sft_grpo_tool", "sft_grpo", True),
    Variant("sft_grpo_tool_grpo", "tool_grpo", True),
)
VARIANT_BY_NAME = {v.name: v for v in VARIANTS}
ADAPTER_KEYS = {"sft", "sft_grpo", "tool_grpo"}


def load_config(path: str | Path) -> dict[str, Any]:
    conf = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(conf, dict) or not isinstance(conf.get("adapters"), dict):
        raise ValueError("Config needs an adapters object")
    extra = set(conf["adapters"]) - ADAPTER_KEYS
    if extra:
        raise ValueError(f"Unknown adapter keys: {sorted(extra)}")
    model = conf.get("base_model", BASE_MODEL)
    if not isinstance(model, str) or not model:
        raise ValueError("Invalid base_model")
    return conf


def resolve_adapter(variant: Variant, config: dict[str, Any]) -> str | None:
    if variant.adapter_key is None:
        return None
    raw = config["adapters"].get(variant.adapter_key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{variant.name}: missing '{variant.adapter_key}' checkpoint")
    path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    cfg = path / "adapter_config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"{variant.name}: missing {cfg}")
    info = json.loads(cfg.read_text(encoding="utf-8"))
    trained_base = info.get("base_model_name_or_path")
    if trained_base and trained_base != config.get("base_model", BASE_MODEL):
        raise ValueError(f"{variant.name}: incompatible adapter base {trained_base}")
    return str(path)


def final_answer_from_reasoning(raw: str) -> str | None:
    """Do not mine numbers from intermediate calculations or invalid tags."""
    answer = extract_unique_final_answer(raw)
    if answer is not None:
        return answer
    if any(tag in raw for tag in ("<think", "</think", "<answer", "</answer")):
        return None
    return raw.strip() or None


def error_eur(predicted: str | None, expected: str) -> float | None:
    if predicted is None or len(NUMBER_PATTERN.findall(predicted)) != 1:
        return None
    p, e = parse_imv_decimal(predicted), parse_imv_decimal(expected)
    if p is None or e is None:
        return None
    return float(abs(imv_money(p) - imv_money(e)))


def score_case(
    case: dict[str, Any], predicted: str | None,
    *, variant: Variant, raw: list[str], calls: list[dict[str, Any]],
    duration_ms: float, tokens: int | None,
) -> dict[str, Any]:
    reference = str(case["answer"])
    executed = any(
        c.get("name") == "calcular_imv" and c.get("ok") is True for c in calls
    )
    return {
        "variant": variant.name,
        "question": case["question"],
        "family": (case.get("branches") or {}).get("family"),
        "expected": reference,
        "predicted": predicted,
        "numeric_correct": imv_numbers_match(predicted, reference),
        "absolute_error_eur": error_eur(predicted, reference),
        "strict_numeric_format": bool(predicted and AMOUNT.fullmatch(predicted)),
        "tool_executed": executed if variant.with_tool else None,
        "tool_call_count": len(calls) if variant.with_tool else None,
        "tool_error_count": sum(not c.get("ok") for c in calls) if variant.with_tool else None,
        "elapsed_ms": round(duration_ms, 2),
        "generated_tokens": tokens,
        "raw_generations": raw,
        "tool_calls": calls,
    }


def summarize(rows: list[dict[str, Any]], variant: Variant) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("Cannot summarize zero examples")
    errors = [r["absolute_error_eur"] for r in rows if r["absolute_error_eur"] is not None]
    return {
        "n": n,
        "numeric_accuracy": sum(r["numeric_correct"] for r in rows) / n,
        "parseable_rate": len(errors) / n,
        "mae_eur": sum(errors) / n if len(errors) == n else None,
        "strict_numeric_format_rate": sum(r["strict_numeric_format"] for r in rows) / n,
        "tool_execution_rate": (
            sum(r["tool_executed"] for r in rows) / n if variant.with_tool else None
        ),
        "tool_error_rate": (
            sum(r["tool_error_count"] > 0 for r in rows) / n if variant.with_tool else None
        ),
        "avg_elapsed_ms": sum(r["elapsed_ms"] for r in rows) / n,
        "avg_generated_tokens": (
            sum(r["generated_tokens"] for r in rows) / n
            if all(r["generated_tokens"] is not None for r in rows) else None
        ),
    }


def run_variant(
    variant: Variant, cases: list[dict[str, Any]], *,
    model_id: str, adapter: str | None, max_new_tokens: int,
    max_turns: int, seed: int,
) -> list[dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    if variant.with_tool:
        from tool_use.executor import ToolLoop

        runner = ToolLoop(
            brain="rlm", model_id=model_id, adapter_path=adapter,
            max_new_tokens=max_new_tokens, enable_thinking=False,
        )
    else:
        from rlm.inference import ReasoningModel

        runner = ReasoningModel(base_model=model_id, adapter_path=adapter)
        runner.load()
    rows: list[dict[str, Any]] = []
    try:
        for i, case in enumerate(cases):
            started = time.perf_counter()
            if variant.with_tool:
                trace: list[str] = []
                response = runner.run(user_prompt(case), max_turns=max_turns, trace=trace)
                predicted = response.answer.strip()
                calls = [c.model_dump() for c in response.tool_calls]
                token_count = (
                    sum(len(runner.tokenizer.encode(raw, add_special_tokens=False))
                        for raw in trace)
                    if runner.tokenizer is not None else None
                )
                generated = trace
            else:
                raw, token_count = runner.generate(
                    case["question"], do_sample=False,
                    max_new_tokens=max_new_tokens,
                    rule_context=case.get("rule_context"),
                )
                predicted = final_answer_from_reasoning(raw)
                calls = []
                generated = [raw]
            row = score_case(
                case, predicted, variant=variant, raw=generated, calls=calls,
                duration_ms=(time.perf_counter() - started) * 1000,
                tokens=token_count,
            )
            row["case_index"] = i
            rows.append(row)
            print(f"{variant.name} {i+1}/{len(cases)}: "
                  f"{'correct' if row['numeric_correct'] else 'incorrect'}", flush=True)
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def save_comparison(output: Path, records: dict[str, dict[str, Any]]) -> None:
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        names = (
            "variant", "status", "adapter", "n", "numeric_accuracy",
            "parseable_rate", "mae_eur", "strict_numeric_format_rate",
            "tool_execution_rate", "tool_error_rate", "avg_elapsed_ms",
            "avg_generated_tokens", "error",
        )
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for variant in VARIANTS:
            row = records.get(variant.name, {"status": "not_started"})
            writer.writerow({
                "variant": variant.name, "status": row["status"],
                "adapter": row.get("adapter"),
                **(row.get("summary") or {}), "error": row.get("error"),
            })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/rlm_final/data/test.jsonl")
    parser.add_argument("--train-data", default="results/rlm_final/data/train.jsonl")
    parser.add_argument("--config", default="tool_use/bench/imv_five_models.example.json")
    parser.add_argument("--out-dir", default="reports/imv_five_models")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_BY_NAME))
    parser.add_argument("--n-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.all == bool(args.variant):
        parser.error("Choose --all OR one/more --variant")
    if args.n_examples is not None and args.n_examples <= 0:
        parser.error("--n-examples must be positive")
    if args.max_new_tokens <= 0 or args.max_turns <= 0:
        parser.error("Token and turn limits must be positive")
    cases = read_cases(args.data, args.n_examples)
    train = read_cases(args.train_data)
    assert_disjoint(train, cases)
    config = load_config(args.config)
    base_model = config.get("base_model", BASE_MODEL)
    chosen = set(VARIANT_BY_NAME) if args.all else set(args.variant)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data": str(args.data), "data_sha256": file_hash(args.data),
        "train_data": str(args.train_data),
        "train_sha256": file_hash(args.train_data),
        "n_examples": len(cases), "base_model": base_model,
        "max_new_tokens": args.max_new_tokens, "max_turns": args.max_turns,
        "seed": args.seed, "do_sample": False,
        "metric": "one final IMV+CAPI amount, ROUND_HALF_UP to cents",
        "note": "Tool mode has a separate prompt, schema and multi-turn budget.",
    }
    results: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        item: dict[str, Any] = {"status": "not_selected"}
        if variant.name in chosen:
            try:
                adapter = resolve_adapter(variant, config)
                item["adapter"] = adapter
                if args.dry_run:
                    item["status"] = "ready"
                else:
                    rows = run_variant(
                        variant, cases, model_id=base_model, adapter=adapter,
                        max_new_tokens=args.max_new_tokens, max_turns=args.max_turns,
                        seed=args.seed,
                    )
                    summary = summarize(rows, variant)
                    item.update({"status": "completed", "summary": summary})
                    (output / f"{variant.name}.json").write_text(
                        json.dumps({
                            "configuration": {**metadata, "variant": variant.name,
                                               "adapter": adapter},
                            "summary": summary, "rows": rows,
                        }, ensure_ascii=False, indent=2), encoding="utf-8",
                    )
            except (ValueError, FileNotFoundError) as exc:
                if not args.all:
                    parser.error(str(exc))
                item.update({"status": "missing_checkpoint", "error": str(exc)})
        results[variant.name] = item
        (output / "manifest.json").write_text(
            json.dumps({"configuration": metadata, "variants": results},
                       ensure_ascii=False, indent=2), encoding="utf-8",
        )
        save_comparison(output, results)
        print(f"{variant.name}: {item['status']}", flush=True)
    print(f"CSV: {output / 'comparison.csv'}")


if __name__ == "__main__":
    main()
