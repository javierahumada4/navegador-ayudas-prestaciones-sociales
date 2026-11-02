"""SFT selects diverse short traces and adds oracle-verifiable components."""
from rlm.generate_problems import IMVProblemGenerator
from rlm.imv_engine import evaluate_imv_case
from rlm.train_sft import _trace_with_verified_components, select_verified_traces


def test_sft_adds_verifiable_components_once():
    case = IMVProblemGenerator().generate(1, "train", seed=8)[0].params
    trace = "<think>Aplico la normativa.</think><answer>0.00</answer>"
    out = _trace_with_verified_components(trace, case)
    expected = evaluate_imv_case(case)
    assert f"<imv>{expected['imv_monthly_eur']:.2f}</imv>" in out
    assert f"<capi>{expected['capi_monthly_eur']:.2f}</capi>" in out
    assert out.endswith("<answer>0.00</answer>")
    assert _trace_with_verified_components(out, case) == out


def test_legacy_teacher_component_claims_cannot_override_the_oracle():
    case = IMVProblemGenerator().generate(1, "train", seed=9)[0].params
    wrong = (
        "<think><imv>999999.00</imv> <capi>999999.00</capi>"
        "Justifico las partidas.</think><answer>0.00</answer>"
    )
    out = _trace_with_verified_components(wrong, case)
    expected = evaluate_imv_case(case)
    assert "<imv>999999.00</imv>" not in out
    assert "<capi>999999.00</capi>" not in out
    assert out.count("<imv>") == out.count("<capi>") == 1
    assert f"<imv>{expected['imv_monthly_eur']:.2f}</imv>" in out
    assert f"<capi>{expected['capi_monthly_eur']:.2f}</capi>" in out


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
