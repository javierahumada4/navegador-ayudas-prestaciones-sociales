"""Stop GRPO unless the trained SFT adapter can finish its IMV answers.

The readiness check uses training questions, never held-out test or OOD
problems: it is a functional gate, not an estimate of generalisation.
Always write the JSON report even if the model is not ready.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def select_diverse_problems(problems: list[dict], limit: int) -> list[dict]:
    """Pick one case per family per round in original dataset order."""
    if limit <= 0:
        raise ValueError("number of SFT check examples must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for problem in problems:
        grouped[(problem.get("branches") or {}).get("family") or "other"].append(problem)
    selected: list[dict] = []
    round_idx = 0
    while len(selected) < limit and any(round_idx < len(group) for group in grouped.values()):
        for group in grouped.values():
            if len(selected) >= limit:
                break
            if round_idx < len(group):
                selected.append(group[round_idx])
        round_idx += 1
    return selected


def readiness_failures(summary: dict, *, min_format_rate: float = 0.75,
                       max_truncated_rate: float = 0.25) -> list[str]:
    """Require most SFT generations to finish with all machine-readable tags."""
    failures = []
    if summary["n_generations"] == 0:
        failures.append("no generations")
    if summary["format_rate"] < min_format_rate:
        failures.append(
            f"format_rate={summary['format_rate']:.3f} < required {min_format_rate:.3f}"
        )
    if summary["truncated_rate"] > max_truncated_rate:
        failures.append(
            f"truncated_rate={summary['truncated_rate']:.3f} > allowed {max_truncated_rate:.3f}"
        )
    return failures


def run_check(args) -> dict:
    import torch
    from rlm.evaluate import evaluate_model, load_problems, summarize
    from rlm.verifier import build_verifier

    if not Path(args.adapter).is_dir():
        raise FileNotFoundError(f"SFT adapter directory not found: {args.adapter}")
    all_problems = load_problems(args.data, None)
    problems = select_diverse_problems(all_problems, args.n_examples)
    if not problems:
        raise ValueError(f"No training questions in {args.data}")
    verifier = build_verifier("imv")
    torch.manual_seed(args.seed)
    rows = evaluate_model(
        args.model, args.adapter, problems, verifier,
        max_new_tokens=args.max_new_tokens,
        samples=1, do_sample=True, temperature=args.temperature,
        batch_size=args.batch_size, seed=args.seed,
    )
    summary = summarize(rows)
    failures = readiness_failures(
        summary, min_format_rate=args.min_format_rate,
        max_truncated_rate=args.max_truncated_rate,
    )
    report = {
        "purpose": "SFT completion readiness on training problems; NOT a held-out evaluation",
        "model": args.model,
        "adapter": args.adapter,
        "min_format_rate": args.min_format_rate,
        "max_truncated_rate": args.max_truncated_rate,
        "max_new_tokens": args.max_new_tokens,
        "ready": not failures,
        "failures": failures,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SFT readiness: format={summary['format_rate']:.1%}, "
          f"truncated={summary['truncated_rate']:.1%}, "
          f"pass@1={summary['pass@1']:.1%} (training cases); report={output}", flush=True)
    if failures:
        raise RuntimeError(
            "GRPO blocked because SFT cannot reliably finish its answers: "
            + "; ".join(failures) + f". See {output}."
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="TRAIN JSONL only; no test/ood leakage")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--min-format-rate", type=float, default=0.75)
    parser.add_argument("--max-truncated-rate", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.n_examples <= 0 or args.max_new_tokens <= 0 or args.batch_size <= 0:
        parser.error("n-examples, max-new-tokens and batch-size must be positive")
    if not (0 <= args.min_format_rate <= 1 and 0 <= args.max_truncated_rate <= 1):
        parser.error("readiness thresholds must be between 0 and 1")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    return run_check(args)


if __name__ == "__main__":
    main()
