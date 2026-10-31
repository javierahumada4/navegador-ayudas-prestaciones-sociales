"""A staged execution plan must be valid without a GPU or downloaded model."""
from rlm.pipeline import build_plan, parse_args


def test_plan_wires_all_stages_and_isolates_uv_extras():
    plan = build_plan(parse_args(["--dry-run", "--run-dir", "runs/test-plan"]))
    assert [stage.name for stage in plan] == ["generate", "distill", "sft", "grpo", "evaluate"]
    commands = {step.title: step.command for stage in plan for step in stage.steps}
    assert "--split" in commands["generate train"]
    assert "--init-adapter" in commands["GRPO from SFT"]
    assert "--traces" in commands["validate traces"]
    assert "--adapters" in commands["evaluate test"]
    assert "--runs" in commands["GRPO curves"]
    assert plan[1].steps[0].env_path != plan[2].steps[0].env_path


def test_smoke_has_small_datasets_and_short_training():
    args = parse_args(["--smoke", "--dry-run"])
    assert (args.train_size, args.test_size, args.ood_size) == (24, 8, 8)
    assert args.grpo_steps == 2
    assert len(build_plan(args)) == 5


def test_resume_from_sft_does_not_generate_again():
    plan = build_plan(parse_args(["--from-stage", "sft", "--through-stage", "grpo"]))
    assert [stage.name for stage in plan] == ["sft", "grpo"]
