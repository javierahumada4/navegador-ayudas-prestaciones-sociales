"""GRPO must abort long zero-signal runs rather than spending GPU hours blindly."""
import pytest

pytest.importorskip("datasets")

from rlm.train_grpo import ZeroSignalMonitor  # noqa: E402


def test_five_uninformative_steps_trigger_abort():
    monitor = ZeroSignalMonitor(patience=5)
    for step in range(1, 5):
        assert not monitor.observe(step, {"frac_reward_zero_std": 1.0})
    assert monitor.observe(5, {"frac_reward_zero_std": 1.0})


def test_recovery_resets_streak_and_duplicate_logs_do_not_count():
    monitor = ZeroSignalMonitor(patience=3)
    assert not monitor.observe(1, {"train/frac_reward_zero_std": 1.0})
    assert not monitor.observe(1, {"train/frac_reward_zero_std": 1.0})
    assert monitor.streak == 1
    assert not monitor.observe(2, {"frac_reward_zero_std": 0.5})
    assert monitor.streak == 0
    assert not monitor.observe(3, {"frac_reward_zero_std": 1.0})
    assert not monitor.observe(4, {"frac_reward_zero_std": 1.0})
    assert monitor.observe(5, {"frac_reward_zero_std": 1.0})


def test_no_metric_and_disabled_monitor_never_abort():
    monitor = ZeroSignalMonitor(patience=0)
    for step in range(1, 10):
        assert not monitor.observe(step, {"frac_reward_zero_std": 1.0})
    active = ZeroSignalMonitor(patience=2)
    assert not active.observe(1, {"reward": 0.0})
    assert active.streak == 0
