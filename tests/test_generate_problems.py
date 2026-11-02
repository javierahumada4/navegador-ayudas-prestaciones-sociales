"""Regression tests for the real IMV/CAPI generator, not the shipping example."""
import json

from rlm.generate_problems import IMVProblemGenerator, describe
from rlm.imv_engine import expected_answer
from rlm.verifier import IMVAmountVerifier

GEN = IMVProblemGenerator()


def test_generation_is_deterministic_and_deduplicated():
    first = GEN.generate(40, "train", seed=31)
    second = GEN.generate(40, "train", seed=31)
    assert [p.question for p in first] == [p.question for p in second]
    assert [p.answer for p in first] == [p.answer for p in second]
    assert len({GEN.key(p.params) for p in first}) == 40


def test_oracle_and_verifier_agree_on_money():
    verifier = IMVAmountVerifier()
    for problem in GEN.generate(45, "train", seed=11):
        assert problem.answer == expected_answer(problem.params)
        assert verifier.is_correct(problem.answer, problem.answer)
        assert len(problem.answer.rsplit(".", 1)[-1]) == 2


def test_ood_capi_only_is_held_out():
    train = GEN.generate(60, "train", seed=5)
    test = GEN.generate(20, "test", seed=5)
    ood = GEN.generate(20, "ood", seed=5)
    assert all(p.branches["family"] != "capi_only" for p in train + test)
    assert all(p.branches["family"] == "capi_only" for p in ood)
    fingerprints = {GEN.key(p.params) for p in train}
    assert fingerprints.isdisjoint({GEN.key(p.params) for p in test + ood})


def test_templates_and_no_label_leak():
    stats = describe(GEN.generate(100, "train", seed=9))
    assert stats["n_templates"] >= 3
    assert stats["answer_leaked_in_statement"] == 0


def test_fingerprint_ignores_key_order():
    params = GEN.generate(1, "train", seed=13)[0].params
    assert GEN.key(params) == GEN.key(dict(reversed(list(params.items()))))
    assert json.loads(GEN.key(params)) == params


def test_people_and_household_dates_are_chronologically_possible():
    from datetime import date

    for split, seed in (("train", 9), ("test", 15), ("ood", 33)):
        for problem in GEN.generate(200, split, seed=seed):
            params = problem.params
            app = date.fromisoformat(params["application_date"])
            births = [date.fromisoformat(p["date_of_birth"]) for p in params["persons"]]
            together = date.fromisoformat(params["domicile"]["same_domicile_since"])
            assert max(births) <= together <= app
            for person in params["persons"]:
                born = date.fromisoformat(person["date_of_birth"])
                residence = date.fromisoformat(
                    person["continuous_legal_effective_residence_since"]
                )
                assert born <= residence <= app
