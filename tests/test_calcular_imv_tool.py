"""Regression tests for the phase-2 deterministic IMV calculation tool."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from tool_use.imv_tools import calcular_imv
from tool_use.tools import default_registry


@pytest.fixture
def case():
    def person(pid, birth):
        return {
            "id": pid,
            "date_of_birth": birth,
            "emancipated": False,
            "legal_residence_in_spain": True,
            "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": "2010-01-01"
            if pid != "p3" else "2010-06-01",
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
        }

    return {
        "application_date": "2026-09-26",
        "applicant_id": "p1",
        "persons": [
            person("p1", "1985-05-10"),
            person("p2", "1984-01-01"),
            person("p3", "2010-06-01"),
        ],
        "domicile": {
            "co_resident_ids": ["p1", "p2", "p3"],
            "same_domicile_since": "2010-06-01",
            "caregiver_family_only": True,
        },
        "relationships": [
            {"person_a": "p1", "person_b": "p2", "type": "spouse"},
            {"person_a": "p1", "person_b": "p3", "type": "parent_child", "custody": "shared", "degree": 1},
            {"person_a": "p2", "person_b": "p3", "type": "parent_child", "custody": "shared", "degree": 1},
        ],
        "applicant_history": {"independent_from_parents_since": "2010-01-01"},
        "social_security_registration_periods": [],
        "economic": {
            "countable_income_annual_eur": 9600.00,
            "net_worth_eur": 5000.00,
            "non_corporate_assets_eur": 5000.00,
        },
    }


def test_tool_is_registered_with_nested_json_schema():
    registry = default_registry()
    assert "calcular_imv" in registry.tools
    assert set(registry.tools) == {"calcular_imv"}
    definition = registry.tools["calcular_imv"].definition["function"]
    assert definition["parameters"]["required"] == ["case"]
    assert "IMVCase" in str(definition["parameters"])
    assert registry.tools["calcular_imv"].requires_confirmation is False


def test_complete_case_returns_reference_breakdown(case):
    record = default_registry().call("calcular_imv", {"case": case})
    assert record.ok, record.error
    assert record.result["imv_eligible"]
    assert record.result["imv_monthly_eur"] == 373.76
    assert record.result["capi_eligible"]
    assert record.result["capi_monthly_eur"] == 57.50
    assert record.result["total_monthly_eur"] == 431.26
    assert record.result["failed_requirements"] == []
    assert record.latency_ms >= 0


def test_capi_only_case(case):
    case["economic"]["countable_income_annual_eur"] = 18000.0
    record = default_registry().call("calcular_imv", {"case": case})
    assert record.ok, record.error
    assert not record.result["imv_eligible"]
    assert record.result["capi_eligible"]
    assert record.result["total_monthly_eur"] == 57.50


@pytest.mark.parametrize(
    ("edit", "expected_error"),
    [
        (lambda c: c["economic"].pop("net_worth_eur"), "net_worth_eur"),
        (lambda c: c["persons"][0].pop("legal_residence_in_spain"), "legal_residence_in_spain"),
        (lambda c: c["economic"].update({"countable_income_annual_eur": -5}), "countable_income_annual_eur"),
        (lambda c: c.update({"applicant_id": "unknown"}), "applicant_id"),
        (lambda c: c["domicile"].update({"co_resident_ids": ["p1", "unknown"]}), "co_resident_ids"),
        (lambda c: c["persons"][0].update({"date_of_birth": "2030-01-01"}), "date_of_birth"),
    ],
)
def test_missing_or_inconsistent_facts_are_rejected(case, edit, expected_error):
    edit(case)
    record = default_registry().call("calcular_imv", {"case": case})
    assert not record.ok
    assert expected_error in record.error


def test_optional_benchmark_facts_are_not_required(case):
    """Phase-1 questions omit many neutral conditions; the tool does not demand them."""
    simplified = deepcopy(case)
    for person in simplified["persons"]:
        for name in (
            "emancipated", "gender_violence_victim", "trafficking_victim",
            "homeless", "parents_or_guardians_deceased", "disability_percent",
            "dependency_grade", "permanent_disability_degree",
            "active_company_director", "residential_service", "benefits",
        ):
            person.pop(name)
    simplified["domicile"].pop("caregiver_family_only")
    simplified.pop("applicant_history")
    simplified.pop("social_security_registration_periods")
    for relation in simplified["relationships"]:
        relation.pop("degree", None)
        relation.pop("custody", None)
    result = default_registry().call("calcular_imv", {"case": simplified})
    assert result.ok, result.error
    assert result.result["total_monthly_eur"] == 431.26


def test_explicit_pension_and_director_still_change_result(case):
    case["persons"][0]["benefits"] = [
        {"type": "pension", "monthly_eur_with_extra_payments": 1000.0}
    ]
    pension = default_registry().call("calcular_imv", {"case": case})
    assert pension.ok, pension.error
    assert pension.result["total_monthly_eur"] == 231.26
    case["persons"][0]["active_company_director"] = True
    director = default_registry().call("calcular_imv", {"case": case})
    assert director.ok, director.error
    assert director.result["total_monthly_eur"] == 0.0


def test_unsupported_year_is_rejected_by_engine(case):
    case["application_date"] = "2027-01-01"
    record = default_registry().call("calcular_imv", {"case": case})
    assert not record.ok
    assert "outside the loaded ruleset" in record.error


def test_direct_tool_call_validates_the_case(case):
    assert calcular_imv(case=deepcopy(case))["total_monthly_eur"] == 431.26
    del case["economic"]["net_worth_eur"]
    with pytest.raises(ValidationError):
        calcular_imv(case=case)


def test_tool_cannot_mutate_user_case(case):
    original = deepcopy(case)
    record = default_registry().call("calcular_imv", {"case": case})
    assert record.ok, record.error
    assert case == original



def test_minimal_benchmark_case_does_not_require_full_personal_checklist(case):
    """An incomplete *optional* questionnaire still yields the benchmark result."""
    reduced = deepcopy(case)
    reduced.pop("applicant_history")
    reduced.pop("social_security_registration_periods")
    reduced.pop("relationships")
    reduced["domicile"].pop("caregiver_family_only")
    for person in reduced["persons"]:
        for field in (
            "emancipated", "gender_violence_victim", "trafficking_victim",
            "homeless", "parents_or_guardians_deceased", "disability_percent",
            "dependency_grade", "permanent_disability_degree",
            "active_company_director", "residential_service", "benefits",
        ):
            person.pop(field)
    # A household with no declared legal relationships means separate units:
    # keep the original relationship facts to test only optional omissions.
    reduced["relationships"] = deepcopy(case["relationships"])
    result = default_registry().call("calcular_imv", {"case": reduced})
    assert result.ok, result.error
    assert result.result["total_monthly_eur"] == 431.26
