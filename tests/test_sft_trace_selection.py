"""SFT keeps short verified traces without adding intermediate output fields."""
from rlm.train_sft import _remove_legacy_component_tags, select_verified_traces


def test_sft_strips_old_component_blocks():
    trace = (
        "<think><imv>999.00</imv> <capi>9.00</capi>"
        "Aplico las reglas.</think><answer>57.50</answer>"
    )
    cleaned = _remove_legacy_component_tags(trace)
    assert cleaned == "<think>  Aplico las reglas.</think><answer>57.50</answer>"
    assert _remove_legacy_component_tags(cleaned) == cleaned


def test_sft_rejects_unpaired_old_component_blocks():
    import pytest
    with pytest.raises(ValueError, match="Unmatched"):
        _remove_legacy_component_tags(
            "<think><imv>99.00</think><answer>0.00</answer>"
        )


def test_sft_keeps_shortest_distinct_traces_with_limit():
    from datasets import Dataset
    rows = [
        {"question": "A", "trace": "xxx" * 20, "verified": True},
        {"question": "A", "trace": "short", "verified": True},
        {"question": "A", "trace": "short", "verified": True},
        {"question": "A", "trace": "medium", "verified": True},
        {"question": "B", "trace": "unique", "verified": True},
        {"question": "B", "trace": "no", "verified": False},
    ]
    dataset = select_verified_traces(Dataset.from_list(rows), 2)
    assert list(dataset["trace"]) == ["short", "medium", "unique"]
    assert all(dataset["verified"])


def test_invalid_sft_cap_rejected():
    import pytest
    from datasets import Dataset
    with pytest.raises(ValueError, match="positive"):
        select_verified_traces(Dataset.from_list([{
            "question": "A", "trace": "x", "verified": True,
        }]), 0)
