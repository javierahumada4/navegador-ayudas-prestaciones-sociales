from copy import deepcopy

from rlm.generate_problems import IMVProblemGenerator
from rlm.imv_engine import evaluate_imv_case, expected_answer, load_ruleset


def reference_case():
    return {
        "application_date": "2026-09-26",
        "applicant_id": "p1",
        "persons": [
            {
                "id": "p1",
                "date_of_birth": "1985-05-10",
                "emancipated": False,
                "legal_residence_in_spain": True,
                "effective_residence_in_spain": True,
                "continuous_legal_effective_residence_since": "2010-01-01",
                "gender_violence_victim": False,
                "trafficking_victim": False,
                "homeless": False,
                "parents_or_guardians_deceased": False,
                "disability_percent": 0,
                "dependency_grade": 0,
                "permanent_disability_degree": "none",
                "active_company_director": False,
                "residential_service": "none",
                "benefits": [],
            },
            {
                "id": "p2",
                "date_of_birth": "1984-01-01",
                "emancipated": False,
                "legal_residence_in_spain": True,
                "effective_residence_in_spain": True,
                "continuous_legal_effective_residence_since": "2010-01-01",
                "gender_violence_victim": False,
                "trafficking_victim": False,
                "homeless": False,
                "parents_or_guardians_deceased": False,
                "disability_percent": 0,
                "dependency_grade": 0,
                "permanent_disability_degree": "none",
                "active_company_director": False,
                "residential_service": "none",
                "benefits": [],
            },
            {
                "id": "p3",
                "date_of_birth": "2010-06-01",
                "emancipated": False,
                "legal_residence_in_spain": True,
                "effective_residence_in_spain": True,
                "continuous_legal_effective_residence_since": "2010-06-01",
                "gender_violence_victim": False,
                "trafficking_victim": False,
                "homeless": False,
                "parents_or_guardians_deceased": False,
                "disability_percent": 0,
                "dependency_grade": 0,
                "permanent_disability_degree": "none",
                "active_company_director": False,
                "residential_service": "none",
                "benefits": [],
            },
        ],
        "domicile": {
            "co_resident_ids": ["p1", "p2", "p3"],
            "same_domicile_since": "2010-06-01",
            "caregiver_family_only": True,
        },
        "relationships": [
            {"person_a": "p1", "person_b": "p2", "type": "spouse"},
            {
                "person_a": "p1",
                "person_b": "p3",
                "type": "parent_child",
                "degree": 1,
                "custody": "shared",
            },
            {
                "person_a": "p2",
                "person_b": "p3",
                "type": "parent_child",
                "degree": 1,
                "custody": "shared",
            },
        ],
        "applicant_history": {"independent_from_parents_since": "2010-01-01"},
        "social_security_registration_periods": [],
        "economic": {
            "countable_income_annual_eur": 9600.00,
            "net_worth_eur": 5000.00,
            "non_corporate_assets_eur": 5000.00,
        },
    }


def test_reference_case_matches_project_example():
    result = evaluate_imv_case(reference_case())
    assert result["imv_eligible"]
    assert result["guaranteed_income_monthly_eur"] == 1173.76
    assert result["imv_monthly_eur"] == 373.76
    assert result["capi_eligible"]
    assert result["capi_monthly_eur"] == 57.50
    assert result["total_monthly_eur"] == 431.26
    assert expected_answer(reference_case()) == "431.26"


def test_disability_threshold_is_in_ruleset_not_input_schema():
    case = reference_case()
    case["persons"][1]["disability_percent"] = 40

    current = evaluate_imv_case(case)
    assert current["guaranteed_income_monthly_eur"] == 1173.76

    future_rules = load_ruleset()
    future_rules["amounts"]["disability_min_percent"] = 40
    changed = evaluate_imv_case(case, future_rules)
    assert changed["guaranteed_income_monthly_eur"] == 1335.15


def test_capi_can_be_granted_without_imv():
    case = reference_case()
    case["economic"]["countable_income_annual_eur"] = 18000
    result = evaluate_imv_case(case)
    assert not result["imv_eligible"]
    assert result["capi_eligible"]
    assert result["total_monthly_eur"] == 57.50


def test_net_worth_equality_excludes_imv():
    case = reference_case()
    # 2 adults + 1 minor => annex-II multiplier 1.8.
    case["economic"]["net_worth_eur"] = 26409.60 * 1.8
    result = evaluate_imv_case(case)
    assert not result["imv_eligible"]
    assert "NET_WORTH_LIMIT_EXCEEDED" in result["failed_requirements"]


def test_asset_equality_is_allowed():
    case = reference_case()
    case["economic"]["non_corporate_assets_eur"] = 52819.20 * 1.8
    result = evaluate_imv_case(case)
    assert "NON_CORPORATE_ASSET_LIMIT_EXCEEDED" not in result["failed_requirements"]


def test_monoparental_exclusive_custody_adds_22_percent():
    case = reference_case()
    case["persons"] = [case["persons"][0], case["persons"][2]]
    case["domicile"]["co_resident_ids"] = ["p1", "p3"]
    case["relationships"] = [
        {
            "person_a": "p1",
            "person_b": "p3",
            "type": "parent_child",
            "degree": 1,
            "custody": "exclusive",
        }
    ]
    case["economic"]["countable_income_annual_eur"] = 0
    result = evaluate_imv_case(case)
    assert result["guaranteed_income_monthly_eur"] == 1115.07


def test_active_company_director_excludes_both():
    case = reference_case()
    case["persons"][0]["active_company_director"] = True
    result = evaluate_imv_case(case)
    assert not result["imv_eligible"]
    assert not result["capi_eligible"]


def test_ood_generator_is_capi_only_family():
    problems = IMVProblemGenerator().generate(20, "ood", seed=17)
    assert {p.branches["family"] for p in problems} == {"capi_only"}
    assert all(p.branches["imv_eligible"] == "False" for p in problems)
    assert all(p.branches["capi_eligible"] == "True" for p in problems)


def test_generator_has_no_answer_leakage_in_sample():
    problems = IMVProblemGenerator().generate(100, "train", seed=7)
    assert sum(p.answer in p.question for p in problems) == 0
