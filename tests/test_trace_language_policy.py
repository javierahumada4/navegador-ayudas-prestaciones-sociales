"""Verify that language does not affect correctness checks."""

def test_verified_english_trace_is_accepted_by_split_validator(tmp_path):
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
    report = validate_traces(trace_file, train)
    assert report["verified"] == 1
    assert report["rejected_by_reason"] == {}
