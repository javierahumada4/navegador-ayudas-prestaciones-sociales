"""GRPO reward shaping must be verifiable and resistant to fake numbers."""
from rlm.generate_problems import IMVProblemGenerator
from rlm.imv_engine import evaluate_imv_case
from rlm.train_grpo import (
    _claimed_component, capi_component_reward, imv_component_reward,
)


def _case():
    return IMVProblemGenerator().generate(1, "train", seed=41)[0].params


def test_components_reward_exact_oracle_values():
    case = _case()
    value = evaluate_imv_case(case)
    trace = (
        "<think>\n<imv>" + f"{value['imv_monthly_eur']:.2f}" + "</imv> "
        "<capi>" + f"{value['capi_monthly_eur']:.2f}" + "</capi>\n"
    )
    # A deliberately unfinished reasoning trace still receives partial signal.
    assert imv_component_reward([], [trace], params=[case]) == [1.0]
    assert capi_component_reward([], [trace], params=[case]) == [1.0]


def test_component_reward_rejects_wrong_or_duplicated_numbers():
    case = _case()
    assert imv_component_reward([], ["<think><imv>999999.00</imv>"], params=[case]) == [0.0]
    assert capi_component_reward([], ["<answer><capi>0.00</capi></answer>"], params=[case]) == [0.0]
    assert _claimed_component("<think><imv>0.00</imv><imv>0.00</imv>", "imv") is None
    assert _claimed_component("<think>El IMV es 0.00", "imv") is None
    assert _claimed_component("<think><imv>0,00</imv>", "imv") is None


def test_component_reward_requires_case_params():
    import pytest
    with pytest.raises(ValueError, match="params"):
        imv_component_reward([], ["<think><imv>0.00</imv>"], params=[None])
    with pytest.raises(ValueError, match="one params"):
        capi_component_reward([], ["<think>"], params=[])
