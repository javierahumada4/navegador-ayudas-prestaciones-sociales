"""End-to-end IMV/CAPI phase 1: generator, verified traces, LoRA SFT, GRPO, evaluation.

The launcher itself needs only Python's standard library. uv prepares independent
environments for vLLM (distill) and Transformers/TRL/PEFT (train).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("generate", "distill", "sft", "grpo", "evaluate")


@dataclass
class Step:
    title: str
    command: list[str]
    env_path: str | None = None


@dataclass
class Stage:
    name: str
    steps: list[Step]
    outputs: list[Path]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/phase1-rebuild")
    parser.add_argument("--from-stage", choices=ORDER, default="generate")
    parser.add_argument("--through-stage", choices=ORDER, default="evaluate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace selected stage outputs")
    parser.add_argument("--smoke", action="store_true", help="small real GPU run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--ood-size", type=int, default=100)
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--teacher-backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--teacher-samples", type=int, default=4)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=2048)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-gpu-memory", type=float, default=0.85)
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft-epochs", type=float, default=2.0)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-grad-accum", type=int, default=8)
    parser.add_argument("--sft-max-length", type=int, default=4096)
    parser.add_argument("--grpo-steps", type=int, default=300)
    parser.add_argument("--grpo-generations", type=int, default=4)
    parser.add_argument("--grpo-micro-batch-size", type=int, default=1)
    parser.add_argument("--grpo-completion-length", type=int, default=768)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if ORDER.index(args.from_stage) > ORDER.index(args.through_stage):
        parser.error("--from-stage must precede --through-stage")
    if args.smoke:
        args.train_size, args.test_size, args.ood_size = 24, 8, 8
        args.teacher_samples, args.teacher_max_new_tokens = 2, 1024
        args.sft_epochs, args.grpo_steps, args.grpo_generations = 1.0, 2, 2
    for key in ("train_size", "test_size", "ood_size", "teacher_samples",
                "teacher_max_new_tokens", "grpo_steps", "grpo_generations",
                "sft_batch_size", "sft_grad_accum", "grpo_micro_batch_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    return args


def build_plan(args):
    run = (ROOT / args.run_dir).resolve()
    data, weights, reports, logs = (run / x for x in ("data", "weights", "reports", "logs"))
    train, test, ood = (data / x for x in ("train.jsonl", "test.jsonl", "test_ood.jsonl"))
    traces = data / "sft_traces.jsonl"
    sft, grpo = weights / "sft_lora", weights / "grpo_lora"
    env_distill, env_train = str(run / ".venv-distill"), str(run / ".venv-train")

    def py(module, *args):
        return [sys.executable, "-m", module, *(str(x) for x in args)]

    def uv(extra, module, *args):
        return ["uv", "run", "--extra", extra, "python", "-m", module,
                *(str(x) for x in args)]

    generation = [
        Step(f"generate {split}", py("rlm.generate_problems",
             "--n", n, "--split", split, "--seed", args.seed, "--out", out))
        for split, n, out in (
            ("train", args.train_size, train),
            ("test", args.test_size, test),
            ("ood", args.ood_size, ood),
        )
    ]
    generation.append(Step("validate splits", py(
        "rlm.validate_data", "--train", train, "--test", test, "--ood", ood,
        "--out", reports / "dataset_validation.json"
    )))
    result = [
        Stage("generate", generation, [
            train, test, ood, reports / "dataset_validation.json"
        ]),
        Stage("distill", [
            Step("teacher traces", uv(
                "distill", "rlm.distill", "--data", train, "--teacher", args.teacher,
                "--backend", args.teacher_backend, "--samples", args.teacher_samples,
                "--max-new-tokens", args.teacher_max_new_tokens,
                "--batch-size", args.teacher_batch_size,
                "--gpu-memory-utilization", args.teacher_gpu_memory,
                "--verifier", "imv", "--output", traces,
                "--log-path", logs / "distillation.jsonl"
            ), env_distill),
            Step("validate traces", py(
                "rlm.validate_data", "--train", train, "--traces", traces,
                "--out", reports / "distillation_validation.json"
            )),
        ], [traces, logs / "distillation.jsonl",
            reports / "distillation_validation.json"]),
        Stage("sft", [
            Step("supervised LoRA", uv(
                "train", "rlm.train_sft", "--data", traces, "--model", args.student,
                "--output", sft, "--epochs", args.sft_epochs,
                "--batch-size", args.sft_batch_size, "--grad-accum", args.sft_grad_accum,
                "--max-length", args.sft_max_length
            ), env_train),
        ], [sft]),
        Stage("grpo", [
            Step("GRPO from SFT", uv(
                "train", "rlm.train_grpo", "--data", train,
                "--model", args.student, "--init-adapter", sft, "--output", grpo,
                "--steps", args.grpo_steps, "--num-generations", args.grpo_generations,
                "--micro-batch-size", args.grpo_micro_batch_size,
                "--max-completion-length", args.grpo_completion_length,
                "--seed", args.seed
            ), env_train),
        ], [grpo]),
    ]
    adapters = ["base=none", f"sft={sft}", f"grpo={grpo}"]
    evaluation = [
        Step(f"evaluate {name}", uv(
            "train", "rlm.evaluate", "--data", input_path, "--model", args.student,
            "--adapters", *adapters, "--verifier", "imv", "--samples", 1,
            "--batch-size", args.eval_batch_size,
            "--max-new-tokens", args.eval_max_new_tokens,
            "--seed", args.seed, "--out", reports / f"eval_{name}.json"
        ), env_train)
        for name, input_path in (("test", test), ("ood", ood))
    ]
    evaluation.append(Step("GRPO curves", uv(
        "train", "rlm.curves", "--runs", f"grpo={grpo}",
        "--out", reports / "grpo_curves.png"
    ), env_train))
    result.append(Stage("evaluate", evaluation, [
        reports / "eval_test.json", reports / "eval_ood.json",
        reports / "grpo_curves.png"
    ]))
    return result[ORDER.index(args.from_stage):ORDER.index(args.through_stage) + 1]


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    for stage in plan:
        print(f"\n== {stage.name.upper()} ==")
        for step in stage.steps:
            prefix = (f"UV_PROJECT_ENVIRONMENT={shlex.quote(step.env_path)} "
                      if step.env_path else "")
            print(prefix + shlex.join(step.command))
    if args.dry_run:
        return
    if shutil.which("uv") is None and any(
        step.env_path for stage in plan for step in stage.steps
    ):
        raise SystemExit("Install uv: https://docs.astral.sh/uv/")
    run = (ROOT / args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    manifest = run / "manifest.json"
    record = (json.loads(manifest.read_text(encoding="utf-8"))
              if manifest.exists() else {"configuration": vars(args), "steps": []})
    for stage in plan:
        existing = [p for p in stage.outputs if p.exists()]
        if existing and not args.force:
            raise SystemExit(
                f"{stage.name}: output already exists ({existing[0]}). "
                "Use a fresh --run-dir or --force to replace selected stages."
            )
        if args.force:
            for path in existing:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for step in stage.steps:
            environment = dict(os.environ)
            if step.env_path:
                environment["UV_PROJECT_ENVIRONMENT"] = step.env_path
            started = datetime.now(timezone.utc).isoformat()
            status = "failed"
            try:
                print(f"\nRUN {stage.name}: {step.title}", flush=True)
                subprocess.run(step.command, cwd=ROOT, env=environment, check=True)
                status = "ok"
            finally:
                record["steps"].append({
                    "stage": stage.name, "step": step.title,
                    "command": step.command, "environment": step.env_path,
                    "started_at": started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                })
                manifest.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
    print(f"\nArtifacts and manifest: {run}")


if __name__ == "__main__":
    main()
