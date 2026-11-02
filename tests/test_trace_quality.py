"""Protect the Spanish-language SFT contract without heavyweight dependencies."""
import pytest

from rlm.trace_quality import is_predominantly_english_reasoning


def test_clear_english_reasoning_is_flagged():
    english = (
        "<think>Okay, first we need to check the amount of income for the "
        "family. The user has given the income and the number of members. "
        "We should calculate the monthly amount and then we can check whether "
        "the user has the right to the benefit. The answer is based on the "
        "given income and the monthly amount.</think><answer>431.26</answer>"
    )
    assert is_predominantly_english_reasoning(english)


def test_spanish_or_brief_numerical_reasoning_is_not_flagged():
    spanish = (
        "<think>El importe mensual se calcula con los ingresos anuales. "
        "La renta garantizada de la familia supera los ingresos. "
        "Por tanto, corresponde el importe indicado.</think>"
        "<answer>431.26</answer>"
    )
    assert not is_predominantly_english_reasoning(spanish)
    assert not is_predominantly_english_reasoning(
        "<think>RG - ingreso/12 = 431.26</think><answer>431.26</answer>"
    )


def test_verified_english_trace_is_rejected_by_split_validator(tmp_path):
    from rlm.generate_problems import IMVProblemGenerator
    from rlm.validate_data import validate_traces
    import json
    from dataclasses import asdict

    p = IMVProblemGenerator().generate(1, "train", seed=17)[0]
    train = tmp_path / "train.jsonl"
    trace_file = tmp_path / "traces.jsonl"
    train.write_text(json.dumps({**asdict(p), "split": "train"}) + "\n")
    thought = (
        "First we need to check the family and calculate the income for the "
        "user. The amount is given by the rules. Then we can check the "
        "monthly amount and the income and finally we should answer with "
        "the total amount for the family and the user."
    )
    trace_file.write_text(
        json.dumps({
            "question": p.question, "answer": p.answer, "sample_idx": 0,
            "verified": True,
            "trace": f"<think>{thought}</think><answer>{p.answer}</answer>",
        }) + "\n"
    )
    with pytest.raises(ValueError, match="English verified reasoning"):
        validate_traces(trace_file, train)
