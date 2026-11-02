"""Check the SFT readiness thresholds without requiring any GPU/model."""
from rlm.check_sft import readiness_failures, select_diverse_problems


def test_diverse_selector_rotates_through_families():
    problems = [
        {"id": i, "branches": {"family": fam}}
        for i, fam in enumerate(("ordinary", "ordinary", "individual", "ordinary",
                                  "disability", "disability"))
    ]
    selected = select_diverse_problems(problems, 4)
    assert [p["id"] for p in selected] == [0, 2, 4, 1]


def test_readiness_blocks_incomplete_smoke():
    summary = {"n_generations": 8, "format_rate": 0.125, "truncated_rate": 0.875}
    assert len(readiness_failures(summary)) == 2


def test_readiness_accepts_completed_sft_and_rejects_edge():
    good = {"n_generations": 8, "format_rate": 0.875, "truncated_rate": 0.125}
    assert readiness_failures(good) == []
    low = {"n_generations": 8, "format_rate": 0.625, "truncated_rate": 0.25}
    assert any("format_rate" in reason for reason in readiness_failures(low))
    assert readiness_failures({"n_generations": 0, "format_rate": 0, "truncated_rate": 0})
