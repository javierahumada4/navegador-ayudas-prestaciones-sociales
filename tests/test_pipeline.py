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
    assert "--validation-report" in commands["supervised LoRA"]
    assert "--adapter" in commands["SFT readiness gate"]
    assert plan[1].steps[0].env_path != plan[2].steps[0].env_path


def test_smoke_has_small_datasets_and_short_training():
    args = parse_args(["--smoke", "--dry-run"])
    assert (args.train_size, args.test_size, args.ood_size) == (24, 8, 8)
    assert args.grpo_steps == 2
    assert args.grpo_generations == 4
    assert len(build_plan(args)) == 5


def test_resume_from_sft_does_not_generate_again():
    plan = build_plan(parse_args(["--from-stage", "sft", "--through-stage", "grpo"]))
    assert [stage.name for stage in plan] == ["sft", "grpo"]


def test_resume_must_be_a_trainable_stage():
    import pytest

    with pytest.raises(SystemExit):
        parse_args(["--from-stage", "generate", "--resume"])
    with pytest.raises(SystemExit):
        parse_args(["--from-stage", "sft", "--resume", "--force"])


def test_latest_checkpoint_uses_complete_highest_step(tmp_path):
    from rlm.pipeline import latest_checkpoint

    for n, complete in ((50, True), (100, True), (150, False)):
        folder = tmp_path / f"checkpoint-{n}"
        folder.mkdir()
        if complete:
            (folder / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert latest_checkpoint(tmp_path).name == "checkpoint-100"


def test_resume_grpo_picks_checkpoint_and_preserves_earlier_stages(tmp_path):
    args = parse_args([
        "--run-dir", str(tmp_path),
        "--from-stage", "grpo", "--through-stage", "grpo", "--resume",
    ])
    output = tmp_path / "weights" / "grpo_lora" / "checkpoint-30"
    output.mkdir(parents=True)
    (output / "trainer_state.json").write_text("{}", encoding="utf-8")
    plan = build_plan(args)
    assert len(plan) == 1
    assert [step.title for step in plan[0].steps] == [
        "SFT readiness gate", "GRPO from SFT"
    ]
    command = plan[0].steps[-1].command
    assert command[command.index("--resume-from-checkpoint") + 1] == str(output)


def test_smoke_uses_longer_teacher_budget_and_quality_gate():
    args = parse_args(["--smoke", "--teacher-max-new-tokens", "3072"])
    assert args.teacher_max_new_tokens == 3072
    assert args.teacher_max_model_len == 8192
    assert (args.min_verified_traces, args.min_unique_trace_problems,
            args.min_verified_families) == (6, 4, 3)
    cmds = {step.title: step.command for stage in build_plan(args) for step in stage.steps}
    assert cmds["teacher traces"][cmds["teacher traces"].index("--max-new-tokens") + 1] == "3072"
    assert "--min-verified-traces" in cmds["validate traces"]
    assert "--zero-signal-patience" in cmds["GRPO from SFT"]


def test_full_dataset_quality_gate_is_stronger():
    args = parse_args(["--dry-run"])
    assert (args.min_verified_traces, args.min_unique_trace_problems,
            args.min_verified_families) == (100, 50, 5)
    assert args.grpo_completion_length == 1024
    assert args.eval_max_new_tokens == 1536


def test_optional_teacher_non_native_thinking_mode():
    plan = build_plan(parse_args(["--smoke", "--teacher-no-thinking"]))
    command = plan[1].steps[0].command
    assert "--no-thinking" in command
    assert "--no-thinking" not in build_plan(parse_args(["--smoke"]))[1].steps[0].command


def test_teacher_seed_and_expected_sample_count_travel_through_pipeline():
    args = parse_args(["--dry-run", "--smoke", "--seed", "31"])
    commands = {step.title: step.command for stage in build_plan(args) for step in stage.steps}
    teacher = commands["teacher traces"]
    validate = commands["validate traces"]
    assert teacher[teacher.index("--seed") + 1] == "31"
    assert validate[validate.index("--expected-trace-samples") + 1] == "2"


def test_grpo_must_pass_sft_readiness_and_uses_qwen_thinking_sampling():
    args = parse_args(["--smoke", "--dry-run"])
    plan = build_plan(args)
    assert [step.title for step in plan[3].steps] == [
        "SFT readiness gate", "GRPO from SFT"
    ]
    cmds = {step.title: step.command for stage in plan for step in stage.steps}
    readiness = cmds["SFT readiness gate"]
    assert readiness[readiness.index("--data") + 1].endswith("data/train.jsonl")
    assert "--min-format-rate" in readiness
    assert "--max-truncated-rate" in readiness
    grpo = cmds["GRPO from SFT"]
    for flag, expected in (("--num-generations", "4"), ("--temperature", "0.6"),
                           ("--top-p", "0.95"), ("--top-k", "20")):
        assert grpo[grpo.index(flag) + 1] == expected
    assert any(str(path).endswith("sft_readiness.json") for path in plan[3].outputs)


def test_explicit_grpo_sampling_options_forwarded():
    args = parse_args(["--grpo-generations", "8", "--grpo-temperature", "0.7",
                       "--grpo-top-p", "0.9", "--grpo-top-k", "40"])
    grpo = build_plan(args)[3].steps[-1].command
    assert grpo[grpo.index("--num-generations") + 1] == "8"
    assert grpo[grpo.index("--temperature") + 1] == "0.7"
    assert grpo[grpo.index("--top-p") + 1] == "0.9"
    assert grpo[grpo.index("--top-k") + 1] == "40"
