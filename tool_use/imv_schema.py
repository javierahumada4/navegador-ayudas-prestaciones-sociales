"""Validated domain contract for the deterministic IMV/CAPI calculator.

The synthetic phase-1 engine expects normalized legal facts, not free-form guesses.
Missing mandatory information is an error. Optional flags default to neutral values
only for the controlled phase-1 benchmark; real-world intake must verify them.
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class Facts(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Benefit(Facts):
    type: Literal["pension", "older_worker_unemployment_subsidy", "other"]
    monthly_eur_with_extra_payments: float = Field(ge=0, allow_inf_nan=False)


class Person(Facts):
    id: str = Field(min_length=1)
    date_of_birth: date
    legal_residence_in_spain: StrictBool
    effective_residence_in_spain: StrictBool
    continuous_legal_effective_residence_since: date
    emancipated: StrictBool = False
    gender_violence_victim: StrictBool = False
    trafficking_victim: StrictBool = False
    homeless: StrictBool = False
    parents_or_guardians_deceased: StrictBool = False
    disability_percent: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    dependency_grade: int = Field(default=0, ge=0, le=3)
    permanent_disability_degree: Literal["none", "absolute", "great_disability"] = "none"
    active_company_director: StrictBool = False
    residential_service: Literal["none", "temporary", "permanent"] = "none"
    benefits: list[Benefit] = Field(default_factory=list)


class Domicile(Facts):
    co_resident_ids: list[str] = Field(min_length=1)
    same_domicile_since: date
    caregiver_family_only: StrictBool = False


class Relationship(Facts):
    person_a: str
    person_b: str
    type: Literal[
        "spouse", "registered_partner", "parent_child", "adoption",
        "consanguinity", "affinity", "pre_adoptive_guard",
        "permanent_foster_care", "unrelated"
    ]
    degree: int = Field(default=99, ge=1)
    custody: Literal["exclusive", "shared", "none"] = "none"


class ApplicantHistory(Facts):
    independent_from_parents_since: date | None = None


class RegistrationPeriod(Facts):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    person_id: str
    from_date: date = Field(alias="from")
    to_date: date | None = Field(default=None, alias="to")

    @model_validator(mode="after")
    def valid_period(self):
        if self.to_date is not None and self.to_date < self.from_date:
            raise ValueError("registration period ends before it starts")
        return self


class EconomicFacts(Facts):
    countable_income_annual_eur: float = Field(ge=0, allow_inf_nan=False)
    net_worth_eur: float = Field(ge=0, allow_inf_nan=False)
    non_corporate_assets_eur: float = Field(ge=0, allow_inf_nan=False)


class OtherParentInstitution(Facts):
    kind: Literal["prison", "hospital"]
    continuous_months: int = Field(ge=0)


class IMVCase(Facts):
    """A complete normalized 2026 IMV application; dates are YYYY-MM-DD."""

    application_date: date
    applicant_id: str = Field(min_length=1)
    persons: list[Person] = Field(min_length=1)
    domicile: Domicile
    economic: EconomicFacts
    relationships: list[Relationship] = Field(default_factory=list)
    applicant_history: ApplicantHistory = Field(default_factory=ApplicantHistory)
    social_security_registration_periods: list[RegistrationPeriod] = Field(default_factory=list)
    other_parent_institution: OtherParentInstitution | None = None

    @model_validator(mode="after")
    def consistent(self):
        # The same person written twice with identical facts carries no ambiguity: keep one.
        # Qwen3-1.7B does this for people living alone. Same id with different facts is
        # still an error, because then it is unclear which facts are true.
        unique: list[Person] = []
        for person in self.persons:
            if person not in unique:
                unique.append(person)
        self.persons = unique
        ids = [p.id for p in self.persons]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate person ids")
        known = set(ids)
        if self.applicant_id not in known:
            raise ValueError("applicant_id not found")
        residents = self.domicile.co_resident_ids
        if len(residents) != len(set(residents)):
            raise ValueError("duplicate residents")
        if self.applicant_id not in residents or not set(residents) <= known:
            raise ValueError("invalid co_resident_ids")
        if self.domicile.same_domicile_since > self.application_date:
            raise ValueError("domicile date after application")
        for p in self.persons:
            if p.date_of_birth > self.application_date:
                raise ValueError("birth date after application")
            if p.continuous_legal_effective_residence_since > self.application_date:
                raise ValueError("residence date after application")
        for rel in self.relationships:
            if rel.person_a not in known or rel.person_b not in known:
                raise ValueError("relationship refers to unknown person")
            if rel.person_a == rel.person_b:
                raise ValueError("self relationship")
        for period in self.social_security_registration_periods:
            if period.person_id not in known:
                raise ValueError("registration refers to unknown person")
        return self
