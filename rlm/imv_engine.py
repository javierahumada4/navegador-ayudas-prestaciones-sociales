"""Deterministic IMV/CAPI reference engine for phase 1.

This module is deliberately pure Python + stdlib: it is the oracle used by the
problem generator and the unit tests.  The model never calls it at inference
time; it exists to create trustworthy labels and verifiable rewards.

Scope:
- new IMV/CAPI applications
- 2026 ruleset
- normalized inputs (external-law concepts already resolved)
- ordinary units of convivencia plus the common monoparental/disability cases

The ruleset contains every threshold/amount that can change over time.  Input
fields store facts (dates, ages, percentages, amounts), not conclusions such as
"disability_65_or_more".
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import calendar
import json
from pathlib import Path
from typing import Any

CENT = Decimal("0.01")


class IMVInputError(ValueError):
    """Raised when a normalized IMV case is internally inconsistent."""


def D(value: Any) -> Decimal:
    return Decimal(str(value))


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def age_on(birth_date: str | date, on_date: str | date) -> int:
    birth = parse_date(birth_date) if isinstance(birth_date, str) else birth_date
    when = parse_date(on_date) if isinstance(on_date, str) else on_date
    return when.year - birth.year - ((when.month, when.day) < (birth.month, birth.day))


def add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def load_ruleset(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name("rulesets") / "imv_2026.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _annex_ii_multiplier(rules: dict[str, Any], adults: int, minors: int) -> Decimal:
    table = rules["patrimony"]["annex_ii"]
    candidates: list[str]
    if adults == 1:
        candidates = [f"1,{minors}"] if minors < 4 else ["1,4+"]
    elif adults == 2:
        candidates = [f"2,{minors}"] if minors < 3 else ["2,3+"]
    elif adults == 3:
        candidates = [f"3,{minors}"] if minors < 2 else ["3,2+"]
    elif adults == 4:
        candidates = ["4,0"] if minors == 0 else ["4,1+"]
    else:
        candidates = ["other"]
    return D(table.get(candidates[0], table["other"]))


def _base_guaranteed_monthly(rules: dict[str, Any], n_members: int) -> Decimal:
    amounts = rules["amounts"]
    base = D(amounts["single_adult_monthly_eur"])
    if n_members <= 1:
        return base
    multiplier = min(
        Decimal("1") + D(amounts["additional_member_pct"]) * Decimal(n_members - 1),
        D(amounts["max_household_multiplier"]),
    )
    return money(base * multiplier)


def _is_minor(person: dict[str, Any], app_date: date) -> bool:
    return age_on(person["date_of_birth"], app_date) < 18


def _qualifying_edge(relation: dict[str, Any]) -> bool:
    kind = relation["type"]
    if kind in {"spouse", "registered_partner", "parent_child", "adoption"}:
        return True
    if kind in {"consanguinity", "affinity"}:
        return int(relation.get("degree", 99)) <= 2
    if kind in {"pre_adoptive_guard", "permanent_foster_care"}:
        return True
    return False


def build_unit(case: dict[str, Any]) -> list[str]:
    """Build the applicant's ordinary unit of convivencia from co-residence + legal links.

    The upstream normalizer supplies the people currently co-residing at the same
    domicile and legally normalized relationship facts.  The IMV engine decides
    which connected component forms the unit.
    """
    applicant_id = case["applicant_id"]
    persons = {p["id"]: p for p in case["persons"]}
    if applicant_id not in persons:
        raise IMVInputError("applicant_id is not present in persons")

    co_residents = set(case["domicile"]["co_resident_ids"])
    if applicant_id not in co_residents:
        raise IMVInputError("applicant must be included in domicile.co_resident_ids")
    unknown = co_residents - persons.keys()
    if unknown:
        raise IMVInputError(f"unknown co-residents: {sorted(unknown)}")

    adjacency = {pid: set() for pid in co_residents}
    for relation in case.get("relationships", []):
        if not _qualifying_edge(relation):
            continue
        a, b = relation["person_a"], relation["person_b"]
        if a in co_residents and b in co_residents:
            adjacency[a].add(b)
            adjacency[b].add(a)

    component: set[str] = set()
    stack = [applicant_id]
    while stack:
        current = stack.pop()
        if current in component:
            continue
        component.add(current)
        stack.extend(adjacency[current] - component)
    return sorted(component)


def _has_parent_child_relation(case: dict[str, Any], adult_id: str, minor_id: str) -> bool:
    for relation in case.get("relationships", []):
        if relation["type"] != "parent_child":
            continue
        pair = {relation["person_a"], relation["person_b"]}
        if pair == {adult_id, minor_id}:
            return True
    return False


def _custody(case: dict[str, Any], adult_id: str, minor_id: str) -> str:
    for relation in case.get("relationships", []):
        if relation["type"] == "parent_child":
            pair = {relation["person_a"], relation["person_b"]}
            if pair == {adult_id, minor_id}:
                return relation.get("custody", "none")
    return "none"


def is_monoparental(case: dict[str, Any], unit: list[str], rules: dict[str, Any]) -> bool:
    """Common art. 13 monoparental configurations used by the phase-1 task."""
    persons = {p["id"]: p for p in case["persons"]}
    app_date = parse_date(case["application_date"])
    adults = [pid for pid in unit if not _is_minor(persons[pid], app_date)]
    minors = [pid for pid in unit if _is_minor(persons[pid], app_date)]
    if not minors:
        return False

    # One adult with at least one minor under exclusive custody.
    if len(adults) == 1:
        adult = adults[0]
        if any(
            _has_parent_child_relation(case, adult, minor)
            and _custody(case, adult, minor) == "exclusive"
            for minor in minors
        ):
            return True
        # Woman victim of gender violence with her minor descendants.
        if persons[adult].get("gender_violence_victim", False) and all(
            _has_parent_child_relation(case, adult, minor) for minor in minors
        ):
            return True
        # Sole permanent foster/pre-adoptive guardian.
        for relation in case.get("relationships", []):
            if relation["type"] in {"pre_adoptive_guard", "permanent_foster_care"}:
                if adult in {relation["person_a"], relation["person_b"]}:
                    other = relation["person_b"] if relation["person_a"] == adult else relation["person_a"]
                    if other in minors:
                        return True

    # Equivalent monoparental household: parents/grandparents/guardians/fosterers
    # with minors, where one adult has degree-3 dependency / qualifying disability.
    if len(adults) >= 2:
        qualifying_degree = int(rules["monoparental"]["dependency_min_grade"]) if "monoparental" in rules else 3
        qualifying_perm = {"absolute", "great_disability"}
        if any(
            int(persons[a].get("dependency_grade", 0)) >= qualifying_degree
            or persons[a].get("permanent_disability_degree", "none") in qualifying_perm
            for a in adults
        ):
            # The phase-1 normalizer marks whether every adult belongs to the
            # parent/grandparent/guardian/fosterer set. This is a relationship fact,
            # not an IMV eligibility conclusion.
            return bool(case["domicile"].get("caregiver_family_only", False))

    return False


def guaranteed_income_monthly(
    case: dict[str, Any], unit: list[str], rules: dict[str, Any]
) -> Decimal:
    persons = {p["id"]: p for p in case["persons"]}
    app_date = parse_date(case["application_date"])
    base = D(rules["amounts"]["single_adult_monthly_eur"])
    amount = _base_guaranteed_monthly(rules, len(unit))

    if len(unit) > 1 and is_monoparental(case, unit, rules):
        amount += base * D(rules["amounts"]["monoparental_pct"])

    if any(
        D(persons[pid].get("disability_percent", 0))
        >= D(rules["amounts"]["disability_min_percent"])
        for pid in unit
    ):
        amount += base * D(rules["amounts"]["disability_pct"])

    return money(amount)


def _registration_months(case: dict[str, Any], applicant_id: str, start: date, end: date) -> int:
    """Count approximate covered calendar months in normalized registration periods."""
    total_days = 0
    for period in case.get("social_security_registration_periods", []):
        if period["person_id"] != applicant_id:
            continue
        p0 = max(parse_date(period["from"]), start)
        p1 = min(parse_date(period["to"]) if period.get("to") else end, end)
        if p0 <= p1:
            total_days += (p1 - p0).days + 1
    return total_days // 30


def _individual_independence_ok(case: dict[str, Any], rules: dict[str, Any]) -> bool:
    persons = {p["id"]: p for p in case["persons"]}
    applicant = persons[case["applicant_id"]]
    app_date = parse_date(case["application_date"])
    age = age_on(applicant["date_of_birth"], app_date)
    req = rules["requirements"]

    # External legal statuses that exempt the ordinary independence test.
    if any(
        applicant.get(key, False)
        for key in (
            "gender_violence_victim",
            "trafficking_victim",
            "homeless",
            "parents_or_guardians_deceased",
        )
    ):
        return True

    independent_since_raw = case.get("applicant_history", {}).get("independent_from_parents_since")
    if not independent_since_raw:
        return False
    independent_since = parse_date(independent_since_raw)

    if age < int(req["independence_under_age"]):
        months_needed = int(req["independence_under_age_months"])
        start = add_months(app_date, -months_needed)
        if independent_since > start:
            return False
        return _registration_months(case, case["applicant_id"], start, app_date) >= int(
            req["independence_under_age_ss_months"]
        )

    start = add_months(app_date, -int(req["independence_age_30_plus_months"]))
    return independent_since <= start


def _non_economic_eligibility(
    case: dict[str, Any], unit: list[str], rules: dict[str, Any]
) -> tuple[bool, list[str]]:
    persons = {p["id"]: p for p in case["persons"]}
    applicant = persons[case["applicant_id"]]
    app_date = parse_date(case["application_date"])
    req = rules["requirements"]
    failures: list[str] = []

    # Residence for every member of the resulting unit.
    residence_cutoff = add_months(app_date, -int(req["minimum_continuous_residence_months"]))
    for pid in unit:
        p = persons[pid]
        if not p.get("legal_residence_in_spain", False) or not p.get(
            "effective_residence_in_spain", False
        ):
            failures.append("RESIDENCE_NOT_MET")
            break
        since = parse_date(p["continuous_legal_effective_residence_since"])
        if since > residence_cutoff and not (
            p.get("gender_violence_victim", False) or p.get("trafficking_victim", False)
        ):
            failures.append("RESIDENCE_NOT_MET")
            break

    # Holder age and independence.
    applicant_age = age_on(applicant["date_of_birth"], app_date)
    if len(unit) == 1:
        if applicant_age < int(req["individual_min_age"]) and not (
            applicant.get("gender_violence_victim", False)
            or applicant.get("trafficking_victim", False)
        ):
            failures.append("APPLICANT_AGE_NOT_MET")
        if not _individual_independence_ok(case, rules):
            failures.append("INDEPENDENCE_REQUIREMENT_NOT_MET")
    else:
        if applicant_age < int(req["individual_min_age"]):
            # A younger adult/emancipated holder is accepted when responsible for
            # a minor in the unit.
            has_minor_child = any(
                _is_minor(persons[pid], app_date)
                and _has_parent_child_relation(case, case["applicant_id"], pid)
                for pid in unit
            )
            if not has_minor_child and not applicant.get("emancipated", False):
                failures.append("APPLICANT_AGE_NOT_MET")

        formation_cutoff = add_months(app_date, -int(req["minimum_household_formation_months"]))
        same_since = parse_date(case["domicile"]["same_domicile_since"])
        recent_birth = any(
            _is_minor(persons[pid], app_date)
            and parse_date(persons[pid]["date_of_birth"]) > formation_cutoff
            for pid in unit
        )
        if same_since > formation_cutoff and not recent_birth:
            failures.append("HOUSEHOLD_FORMATION_PERIOD_NOT_MET")

    # Permanent residential services exclude except selected statutory exceptions.
    if any(
        persons[pid].get("residential_service", "none") == "permanent"
        and not (
            persons[pid].get("gender_violence_victim", False)
            or persons[pid].get("trafficking_victim", False)
        )
        for pid in unit
    ):
        failures.append("PERMANENT_RESIDENTIAL_SERVICE_EXCLUSION")

    return (not failures), failures


def _economic_limits(
    case: dict[str, Any], unit: list[str], rules: dict[str, Any]
) -> dict[str, Decimal]:
    persons = {p["id"]: p for p in case["persons"]}
    app_date = parse_date(case["application_date"])
    adults = sum(not _is_minor(persons[pid], app_date) for pid in unit)
    minors = len(unit) - adults
    annex_ii = _annex_ii_multiplier(rules, adults, minors)

    annual_single = D(rules["amounts"]["single_adult_monthly_eur"]) * Decimal(12)
    single_net = annual_single * D(
        rules["patrimony"]["single_adult_net_worth_multiplier_annual"]
    )
    single_assets = annual_single * D(
        rules["patrimony"]["single_adult_non_corporate_assets_multiplier_annual"]
    )

    base_annual = _base_guaranteed_monthly(rules, len(unit)) * Decimal(12)
    return {
        "net_worth": money(single_net * annex_ii),
        "assets": money(single_assets * annex_ii),
        "capi_income": money(base_annual * D(rules["capi"]["income_multiplier"])),
        "capi_net_worth": money(
            single_net * annex_ii * D(rules["capi"]["net_worth_multiplier"])
        ),
    }


def economic_limits(case: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Decimal]:
    """Public helper used by the synthetic problem generator for boundary sampling."""
    rules = rules or load_ruleset()
    unit = build_unit(case)
    return _economic_limits(case, unit, rules)


def capi_monthly(case: dict[str, Any], unit: list[str], rules: dict[str, Any]) -> Decimal:
    persons = {p["id"]: p for p in case["persons"]}
    app_date = parse_date(case["application_date"])
    jan1 = date(app_date.year, 1, 1)
    total = Decimal("0")
    for pid in unit:
        p = persons[pid]
        if not _is_minor(p, app_date):
            continue
        age = age_on(p["date_of_birth"], jan1)
        for band in rules["capi"]["age_bands"]:
            if age < int(band["max_age_exclusive"]):
                total += D(band["monthly_eur"])
                break
    return money(total)


def evaluate_imv_case(
    case: dict[str, Any], rules: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the deterministic reference result for one normalized IMV case."""
    rules = rules or load_ruleset()
    app_date = parse_date(case["application_date"])
    if not (parse_date(rules["effective_from"]) <= app_date <= parse_date(rules["effective_to"])):
        raise IMVInputError("application_date is outside the loaded ruleset")

    unit = build_unit(case)
    persons = {p["id"]: p for p in case["persons"]}
    non_economic_ok, failures = _non_economic_eligibility(case, unit, rules)
    guaranteed = guaranteed_income_monthly(case, unit, rules)
    limits = _economic_limits(case, unit, rules)

    econ = case["economic"]
    annual_income = D(econ["countable_income_annual_eur"])
    monthly_income = annual_income / Decimal(12)
    net_worth = D(econ["net_worth_eur"])
    assets = D(econ["non_corporate_assets_eur"])

    active_director = any(persons[pid].get("active_company_director", False) for pid in unit)
    if active_director:
        failures.append("ACTIVE_COMPANY_DIRECTOR")

    net_ok = net_worth < limits["net_worth"]
    if not net_ok:
        failures.append("NET_WORTH_LIMIT_EXCEEDED")

    assets_ok = assets <= limits["assets"]
    if not assets_ok:
        failures.append("NON_CORPORATE_ASSET_LIMIT_EXCEEDED")

    raw_imv = guaranteed - monthly_income
    income_ok = raw_imv >= D(rules["amounts"]["minimum_imv_monthly_eur"])
    if not income_ok:
        failures.append("IMV_INCOME_LIMIT_EXCEEDED")

    pension_monthly = sum(
        D(benefit.get("monthly_eur_with_extra_payments", 0))
        for pid in unit
        for benefit in persons[pid].get("benefits", [])
        if benefit.get("type") in {"pension", "older_worker_unemployment_subsidy"}
    )
    pension_cap_ok = pension_monthly < guaranteed
    if not pension_cap_ok:
        failures.append("PENSION_CAP_EXCLUDES_IMV")

    imv_eligible = (
        non_economic_ok
        and not active_director
        and net_ok
        and assets_ok
        and income_ok
        and pension_cap_ok
    )
    imv_amount = Decimal("0")
    if imv_eligible:
        imv_amount = raw_imv
        if pension_monthly:
            imv_amount = min(imv_amount, guaranteed - pension_monthly)
        imv_amount = money(max(imv_amount, Decimal("0")))

    minors = [pid for pid in unit if _is_minor(persons[pid], app_date)]
    capi_income_ok = annual_income < limits["capi_income"]
    capi_net_ok = net_worth < limits["capi_net_worth"]
    capi_assets_ok = assets <= limits["assets"]

    capi_eligible = (
        bool(minors)
        and non_economic_ok
        and not active_director
        and capi_income_ok
        and capi_net_ok
        and capi_assets_ok
    )
    if minors and not capi_income_ok:
        failures.append("CAPI_INCOME_LIMIT_EXCEEDED")
    if minors and not capi_net_ok:
        failures.append("CAPI_NET_WORTH_LIMIT_EXCEEDED")
    if minors and not capi_assets_ok:
        failures.append("CAPI_NON_CORPORATE_ASSET_LIMIT_EXCEEDED")

    capi_amount = capi_monthly(case, unit, rules) if capi_eligible else Decimal("0")
    total = money(imv_amount + capi_amount)

    # Stable deterministic failure ordering.
    failures = list(dict.fromkeys(failures))
    return {
        "status": "eligible" if (imv_eligible or capi_eligible) else "ineligible",
        "unit_member_ids": unit,
        "imv_eligible": imv_eligible,
        "guaranteed_income_monthly_eur": float(guaranteed),
        "imv_monthly_eur": float(imv_amount),
        "capi_eligible": capi_eligible,
        "capi_monthly_eur": float(capi_amount),
        "total_monthly_eur": float(total),
        "failed_requirements": failures,
    }


def expected_answer(case: dict[str, Any], rules: dict[str, Any] | None = None) -> str:
    """Canonical answer used by phase-1 JSONL files and the verifier."""
    result = evaluate_imv_case(case, rules)
    return f"{result['total_monthly_eur']:.2f}"
