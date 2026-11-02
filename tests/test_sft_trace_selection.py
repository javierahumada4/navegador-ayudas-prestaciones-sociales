"""SFT keeps short verified traces without adding intermediate output fields."""
from rlm.train_sft import select_verified_traces


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
