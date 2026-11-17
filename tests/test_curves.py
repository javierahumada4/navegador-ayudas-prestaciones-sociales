"""Checks for the training-curve helpers."""

import json

from rlm.curves import find_history, load_history, plot_runs, smooth

HISTORY = [
    {"step": 1, "reward": 0.5, "rewards/format_reward/mean": 1.0, "completions/mean_length": 900},
    {"step": 2, "reward": 1.5, "rewards/format_reward/mean": 1.0, "completions/mean_length": 800},
    {"step": 2, "train_runtime": 10.0, "train_loss": 0.1},
]


def test_load_history_from_log_and_latest_checkpoint(tmp_path):
    (tmp_path / "log_history.json").write_text(json.dumps(HISTORY))
    assert [e["step"] for e in load_history(tmp_path)] == [1, 2]

    run = tmp_path / "run"
    for step in (2, 10):
        ckpt = run / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(json.dumps({"log_history": HISTORY[:1]}))
    assert find_history(run).parent.name == "checkpoint-10"
    assert len(load_history(run)) == 1


def test_smooth_is_a_trailing_mean():
    assert smooth([0, 2, 4], 1) == [0, 2, 4]
    assert smooth([0, 2, 4], 2) == [0, 1, 3]


def test_plot_skips_missing_metrics(tmp_path):
    out = tmp_path / "curves.png"
    drawn = plot_runs({"run": load_history_list()}, out, window=1)
    assert out.exists()
    assert "Entropía" not in drawn and "Recompensa total" in drawn


SFT_HISTORY = [
    {"step": 10, "loss": 0.9, "mean_token_accuracy": 0.75, "grad_norm": 1.2, "learning_rate": 2e-4},
    {"step": 20, "loss": 0.6, "mean_token_accuracy": 0.82, "grad_norm": 0.8, "learning_rate": 1e-4},
]


def test_plot_sft_panels(tmp_path):
    out = tmp_path / "sft.png"
    drawn = plot_runs({"sft": SFT_HISTORY}, out, window=1, panel_set="sft")
    assert out.exists()
    assert "Pérdida de entrenamiento" in drawn and "Exactitud por token" in drawn
    assert "Recompensa total" not in drawn and "Entropía" not in drawn


def load_history_list():
    return [e for e in HISTORY if "train_runtime" not in e]
