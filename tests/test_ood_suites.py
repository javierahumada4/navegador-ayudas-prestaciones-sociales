"""Additional OOD suites have disjoint parameters and clean labels."""
import json
from collections import Counter
from dataclasses import asdict
from decimal import Decimal

import pytest

from rlm.generate_problems import (
    IMVProblemGenerator, TRAIN_FAMILIES, OOD_FAMILIES, balanced_family_schedule,
)
from rlm.imv_engine import economic_limits, evaluate_imv_case, guaranteed_income_monthly, build_unit
from rlm.validate_data import validate_splits


def test_balanced_train_and_test_cover_each_family():
    gen = IMVProblemGenerator()
    for split in ("train", "test"):
        rows = gen.generate(100, split, seed=17)
        counts = Counter(r.branches["family"] for r in rows)
        assert set(counts) == set(TRAIN_FAMILIES)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_schedule_has_no_duplicate_or_missing_families_when_n_large():
    import random
    schedule = balanced_family_schedule(101, TRAIN_FAMILIES, random.Random(12))
    counts = Counter(schedule)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_imv_boundary_cases_are_one_cent_from_statutory_cutoffs():
    gen = IMVProblemGenerator()
    rows = gen.generate(180, "train", seed=27)
    boundaries = [r for r in rows if r.branches["family"] == "boundary"]
    assert boundaries
    for row in boundaries:
        kind = row.params["_boundary_type"]
        side = row.params["_boundary_side"]
        assert side in (-1, 0, 1)
        case = row.params
        limits = economic_limits(case)
        amount = Decimal(str(case["economic"]["countable_income_annual_eur"]))
        if kind == "capi_income":
            assert amount == limits["capi_income"] + Decimal(side) / 100
        elif kind == "net_worth":
            assert Decimal(str(case["economic"]["net_worth_eur"])) == (
                limits["net_worth"] + Decimal(side) / 100
            )
        elif kind == "non_corporate_assets":
            assert Decimal(str(case["economic"]["non_corporate_assets_eur"])) == (
                limits["assets"] + Decimal(side) / 100
            )
        elif kind == "capi_net_worth":
            assert Decimal(str(case["economic"]["net_worth_eur"])) == (
                limits["capi_net_worth"] + Decimal(side) / 100
            )
        elif kind == "imv_minimum":
            guaranteed = guaranteed_income_monthly(case, build_unit(case), gen.rules)
            gap = guaranteed - amount / 12
            assert gap == Decimal("10.00") - Decimal(side) / 100


def test_ood_suites_are_separate_and_oracle_verified(tmp_path):
    gen = IMVProblemGenerator()
    paths = {split: tmp_path / (split + ".jsonl") for split in (
        "train", "test", "ood", "ood_threshold", "ood_wording"
    )}
    fingerprints = set()
    for split, path in paths.items():
        rows = gen.generate(30 if split in {"train", "test"} else 15, split, seed=13)
        with path.open("w", encoding="utf-8") as f:
            for problem in rows:
                assert problem.answer == f"{evaluate_imv_case(problem.params)['total_monthly_eur']:.2f}"
                assert problem.branches["family"] == (
                    OOD_FAMILIES[split] if split in OOD_FAMILIES
                    else problem.branches["family"]
                )
                if split == "ood_wording":
                    assert problem.template_id >= 100
                fingerprint = gen.key(problem.params)
                assert fingerprint not in fingerprints
                fingerprints.add(fingerprint)
                record = asdict(problem)
                record["split"] = split
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    result = validate_splits(
        paths["train"], paths["test"], paths["ood"],
        {name: paths[name] for name in OOD_FAMILIES if name != "ood"},
    )
    assert set(result) == set(paths)
    assert all(result[name]["count"] > 0 for name in paths)


def test_validation_rejects_ood_family_leakage(tmp_path):
    gen = IMVProblemGenerator()
    train, test, ood = (tmp_path / s for s in ("train", "test", "ood"))
    for path, split in ((train, "train"), (test, "test"), (ood, "ood")):
        p = gen.generate(3, split, seed=2)
        with path.open("w", encoding="utf-8") as f:
            for problem in p:
                row = asdict(problem)
                row["split"] = split
                if split == "test":
                    row["branches"]["family"] = "wording_ood"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="leaked"):
        validate_splits(train, test, ood)
