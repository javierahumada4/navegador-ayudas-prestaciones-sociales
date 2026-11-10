"""A released TRL upgrade must not silently break the training argument contract."""
from dataclasses import fields

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from trl import GRPOConfig, SFTConfig  # noqa: E402


def test_sft_arguments_supported_by_installed_trl():
    supported = {field.name for field in fields(SFTConfig)}
    required = {
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "max_length",
        "model_init_kwargs", "completion_only_loss", "bf16", "gradient_checkpointing", "save_steps",
        "save_strategy", "save_total_limit",
    }
    assert required <= supported


def test_grpo_arguments_supported_by_installed_trl():
    supported = {field.name for field in fields(GRPOConfig)}
    required = {
        "output_dir", "max_steps", "learning_rate", "per_device_train_batch_size",
        "gradient_accumulation_steps", "num_generations",
        "max_completion_length", "temperature", "top_p", "top_k", "beta", "epsilon",
        "reward_weights", "model_init_kwargs", "log_completions",
        "num_completions_to_print",
    }
    assert required <= supported
