"""A staged execution plan must be valid without a GPU or downloaded model."""
import pytest

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
    assert "--selection-report" in commands["supervised LoRA"]
    assert "--validation-report" not in commands["supervised LoRA"]
    assert not any("sft_input_validation.json" in str(p)
                   for stage in plan for p in stage.outputs)
    assert [step.title for step in plan[3].steps] == ["GRPO from SFT"]
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
    assert [step.title for step in plan[0].steps] == ["GRPO from SFT"]
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
    assert args.completion_tokens == 1024
    assert args.teacher_samples == 8
    assert args.min_sft_problem_coverage == 0.70


def test_distillation_cannot_disable_native_thinking():
    teacher = build_plan(parse_args(["--smoke"]))[1].steps[0].command
    assert "--no-thinking" not in teacher
    with pytest.raises(SystemExit):
        parse_args(["--teacher-no-thinking"])


def test_teacher_seed_and_expected_sample_count_travel_through_pipeline():
    args = parse_args(["--dry-run", "--smoke", "--seed", "31"])
    commands = {step.title: step.command for stage in build_plan(args) for step in stage.steps}
    teacher = commands["teacher traces"]
    validate = commands["validate traces"]
    assert teacher[teacher.index("--seed") + 1] == "31"
    assert validate[validate.index("--expected-trace-samples") + 1] == "2"


def test_grpo_runs_directly_after_sft_with_qwen_thinking_sampling():
    args = parse_args(["--smoke", "--dry-run"])
    plan = build_plan(args)
    assert [step.title for step in plan[3].steps] == ["GRPO from SFT"]
    assert not any("sft_readiness" in str(path) for path in plan[3].outputs)
    commands = {step.title: step.command for stage in plan for step in stage.steps}
    assert not any("check_sft" in " ".join(cmd) for cmd in commands.values())
    grpo = commands["GRPO from SFT"]
    assert grpo[grpo.index("--init-adapter") + 1].endswith("/weights/sft_lora")
    for flag, expected in (("--num-generations", "4"), ("--temperature", "0.6"),
                           ("--top-p", "0.95"), ("--top-k", "20")):
        assert grpo[grpo.index(flag) + 1] == expected


def test_explicit_grpo_sampling_options_forwarded():
    args = parse_args(["--grpo-generations", "8", "--grpo-temperature", "0.7",
                       "--grpo-top-p", "0.9", "--grpo-top-k", "40"])
    grpo = build_plan(args)[3].steps[-1].command
    assert grpo[grpo.index("--num-generations") + 1] == "8"
    assert grpo[grpo.index("--temperature") + 1] == "0.7"
    assert grpo[grpo.index("--top-p") + 1] == "0.9"
    assert grpo[grpo.index("--top-k") + 1] == "40"


def test_sft_caps_teacher_traces_and_evaluates_each_ood_axis():
    args = parse_args(["--smoke", "--sft-max-traces-per-problem", "1"])
    plan = build_plan(args)
    commands = {step.title: step.command for stage in plan for step in stage.steps}
    sft = commands["supervised LoRA"]
    assert sft[sft.index("--max-traces-per-problem") + 1] == "1"
    assert {"generate ood", "generate ood_threshold",
            "generate ood_wording"} <= set(commands)
    assert {"evaluate ood", "evaluate ood_threshold",
            "evaluate ood_wording"} <= set(commands)
    assert not any("ood_composition" in title for title in commands)
    validation = commands["validate splits"]
    assert validation.count("--ood-extra") == 2
    for axis in ("ood_threshold", "ood_wording"):
        assert axis in validation
        assert any(str(p).endswith("eval_" + axis + ".json")
                   for p in plan[-1].outputs)
    assert not any("ood_composition" in str(p)
                   for stage in plan for p in stage.outputs)

def test_shared_completion_budget_is_forwarded_to_all_stages():
    args = parse_args(["--completion-tokens", "1024"])
    cmds = {step.title: step.command for stage in build_plan(args) for step in stage.steps}
    for title, flag in (
        ("supervised LoRA", "--max-completion-tokens"),
        ("GRPO from SFT", "--max-completion-length"),
        ("evaluate test", "--max-new-tokens"),
    ):
        command = cmds[title]
        assert command[command.index(flag) + 1] == "1024"
    sft = cmds["supervised LoRA"]
    assert sft[sft.index("--max-length") + 1] == "8192"
    assert sft[sft.index("--min-problem-coverage") + 1] == "0.7"
    assert "--min-families" in sft
    assert any(str(p).endswith("sft_selection.json") for p in build_plan(args)[2].outputs)


def test_shared_completion_budget_must_be_positive():
    with pytest.raises(SystemExit):
        parse_args(["--completion-tokens", "0"])


def test_quality_gates_warn_by_default_without_disabling_integrity_validation():
    args = parse_args(["--dry-run"])
    assert args.quality_gates == "warn"
    cmds = {step.title: step.command for stage in build_plan(args) for step in stage.steps}
    assert "--warn-low-coverage" in cmds["validate traces"]
    assert "--warn-low-coverage" in cmds["supervised LoRA"]
    assert cmds["GRPO from SFT"][
        cmds["GRPO from SFT"].index("--zero-signal-patience") + 1
    ] == "0"
    assert "--warn-low-coverage" not in cmds["validate splits"]


def test_quality_gates_strict_restores_threshold_failures():
    args = parse_args(["--quality-gates", "strict", "--grpo-zero-signal-patience", "7"])
    cmds = {step.title: step.command for stage in build_plan(args) for step in stage.steps}
    assert "--warn-low-coverage" not in cmds["validate traces"]
    assert "--warn-low-coverage" not in cmds["supervised LoRA"]
    assert cmds["GRPO from SFT"][
        cmds["GRPO from SFT"].index("--zero-signal-patience") + 1
    ] == "7"


def test_full_pipeline_defaults_to_50_examples_per_remaining_ood():
    args = parse_args(["--dry-run"])
    assert (args.train_size, args.test_size, args.ood_size) == (800, 200, 50)
    plan = build_plan(args)
    commands = {step.title: step.command for stage in plan for step in stage.steps}
    for split in ("ood", "ood_threshold", "ood_wording"):
        generate = commands["generate " + split]
        assert generate[generate.index("--n") + 1] == "50"
        assert "evaluate " + split in commands
    assert len([title for title in commands if title.startswith("evaluate ")]) == 4
    assert len([title for title in commands if title.startswith("generate ")]) == 5


def test_smoke_ood_size_and_override_remain_available():
    assert parse_args(["--smoke"]).ood_size == 8
    assert parse_args(["--ood-size", "20"]).ood_size == 20


def test_removed_sft_readiness_flags_are_rejected():
    for flag in ("--sft-check-examples", "--sft-min-format-rate",
                 "--sft-max-truncated-rate"):
        with pytest.raises(SystemExit):
            parse_args([flag, "1"])
