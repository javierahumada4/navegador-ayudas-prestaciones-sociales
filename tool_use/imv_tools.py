"""Deterministic IMV/CAPI calculator for the shared phase-1 numerical benchmark.

The model maps the provided synthetic problem to an IMV case; the tool does not
interview the user or check whether every real-life fact was supplied. Optional
conditions not mentioned in these generated cases use the simulator's neutral
defaults. A real-world intake/verification workflow belongs to phase 4.
Core dates, residence, household and economic figures remain required to run
the oracle; Pydantic retains basic type and consistency validation.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from rlm.imv_engine import evaluate_imv_case
from tool_use.registry import tool


class CaseFacts(BaseModel):
    """Do not allow unknown fact keys or quietly coerce missing values."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Benefit(CaseFacts):
    type: str = Field(min_length=1, description="Benefit kind, e.g. pension.")
    monthly_eur_with_extra_payments: float = Field(ge=0, allow_inf_nan=False)


class Person(CaseFacts):
    id: str = Field(min_length=1)
    date_of_birth: date
    emancipated: StrictBool = False
    legal_residence_in_spain: StrictBool
    effective_residence_in_spain: StrictBool
    continuous_legal_effective_residence_since: date
    gender_violence_victim: StrictBool = False
    trafficking_victim: StrictBool = False
    homeless: StrictBool = False
    parents_or_guardians_deceased: StrictBool = False
    disability_percent: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    dependency_grade: int = Field(default=0, ge=0, le=3)
    permanent_disability_degree: Literal["none", "absolute", "great_disability"] = "none"
    active_company_director: StrictBool = False
    residential_service: Literal["none", "temporary", "permanent"] = "none"
    benefits: list[Benefit] = Field(
        default_factory=list,
        description="Pensions/subsidies mentioned in the synthetic problem.",
    )


class Domicile(CaseFacts):
    co_resident_ids: list[str] = Field(min_length=1)
    same_domicile_since: date
    caregiver_family_only: StrictBool = False


class Relationship(CaseFacts):
    person_a: str = Field(min_length=1)
    person_b: str = Field(min_length=1)
    type: Literal[
        "spouse",
        "registered_partner",
        "parent_child",
        "adoption",
        "consanguinity",
        "affinity",
        "pre_adoptive_guard",
        "permanent_foster_care",
        "unrelated",
    ]
    degree: int = Field(default=99, ge=1)
    custody: Literal["exclusive", "shared", "none"] = "none"

class ApplicantHistory(CaseFacts):
    independent_from_parents_since: date | None = Field(
        default=None, description="Independence date when mentioned in the problem."
    )


class SocialSecurityPeriod(CaseFacts):
    person_id: str = Field(min_length=1)
    from_date: date = Field(alias="from")
    to_date: date | None = Field(alias="to")

    @model_validator(mode="after")
    def check_period(self):
        if self.to_date is not None and self.to_date < self.from_date:
            raise ValueError("registration 'to' cannot precede 'from'")
        return self


class EconomicFacts(CaseFacts):
    countable_income_annual_eur: float = Field(ge=0, allow_inf_nan=False)
    net_worth_eur: float = Field(ge=0, allow_inf_nan=False)
    non_corporate_assets_eur: float = Field(ge=0, allow_inf_nan=False)


class OtherParentInstitution(CaseFacts):
    kind: Literal["prison", "hospital"]
    continuous_months: int = Field(ge=0)


class IMVCase(CaseFacts):
    """Input format already used by rlm.imv_engine.evaluate_imv_case."""

    application_date: date = Field(description="Application date within the 2026 ruleset.")
    applicant_id: str = Field(min_length=1)
    persons: list[Person] = Field(min_length=1)
    domicile: Domicile
    relationships: list[Relationship] = Field(default_factory=list)
    applicant_history: ApplicantHistory = Field(default_factory=ApplicantHistory)
    social_security_registration_periods: list[SocialSecurityPeriod] = Field(
        default_factory=list
    )
    economic: EconomicFacts
    other_parent_institution: OtherParentInstitution | None = None

    @model_validator(mode="after")
    def check_internal_consistency(self):
        ids = [person.id for person in self.persons]
        known_ids = set(ids)
        if len(ids) != len(known_ids):
            raise ValueError("persons contains duplicate ids")
        if self.applicant_id not in known_ids:
            raise ValueError("applicant_id must refer to a person")
        residents = self.domicile.co_resident_ids
        if len(residents) != len(set(residents)):
            raise ValueError("domicile.co_resident_ids contains duplicates")
        if self.applicant_id not in residents:
            raise ValueError("applicant must be a co-resident")
        if not set(residents).issubset(known_ids):
            raise ValueError("co_resident_ids contains unknown persons")
        if self.domicile.same_domicile_since > self.application_date:
            raise ValueError("same_domicile_since cannot be after application_date")
        for person in self.persons:
            if person.date_of_birth > self.application_date:
                raise ValueError(f"date_of_birth for {person.id} is in the future")
            if person.continuous_legal_effective_residence_since > self.application_date:
                raise ValueError(f"residence start for {person.id} is in the future")
        for relation in self.relationships:
            if relation.person_a not in known_ids or relation.person_b not in known_ids:
                raise ValueError("relationship refers to an unknown person")
            if relation.person_a == relation.person_b:
                raise ValueError("relationship cannot refer to the same person twice")
        for period in self.social_security_registration_periods:
            if period.person_id not in known_ids:
                raise ValueError("registration period refers to an unknown person")
        return self


@tool
def calcular_imv(case: IMVCase) -> dict:
    """Calculate IMV+CAPI from synthetic 2026 benchmark facts.

    Use for numerical IMV/CAPI problems. Translate the facts in the question
    into the structured case; omitted *optional* exceptions use the benchmark
    simulator defaults (e.g. no disability or pension). The tool does not
    perform a data-completeness interview. Keep the provided household, dates,
    income, patrimony, residence and custody exact. This is an orientation
    calculation and not an official eligibility decision.

    Args:
        case: Structured facts from the IMV problem: people, dates, residence,
            economic values, relevant family relationships, and mentioned
            exceptions. Numeric calculations are performed by the engine.

    Returns:
        Engine-produced status, member IDs, IMV and CAPI eligibility and monthly
        amounts, total_monthly_eur and failed_requirements.
    """
    # ToolRegistry passes a validated dict to the function, including date
    # objects and Python field names. Re-validate direct calls as well, then
    # serialize dates and aliases exactly as the reference engine expects.
    normalized = IMVCase.model_validate(case).model_dump(mode="json", by_alias=True)
    return evaluate_imv_case(normalized)
