"""Deterministic Spanish-language presentation of the 2026 IMV engine output.

This tool never invokes a language model or performs its own eligibility
calculation. It only renders the *validated* structured response of
`rlm.imv_engine.evaluate_imv_case` using fixed Spanish text templates.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from tool_use.registry import tool

CENT = Decimal("0.01")

# Labels describe engine checks, not additional rules or official decisions.
FAILED_CHECK_LABELS = {
    "RESIDENCE_NOT_MET": "residencia",
    "APPLICANT_AGE_NOT_MET": "edad de la persona solicitante",
    "INDEPENDENCE_REQUIREMENT_NOT_MET": "independencia de la persona solicitante",
    "HOUSEHOLD_FORMATION_PERIOD_NOT_MET": "antigüedad de la unidad de convivencia",
    "PERMANENT_RESIDENTIAL_SERVICE_EXCLUSION": "estancia permanente en servicio residencial",
    "ACTIVE_COMPANY_DIRECTOR": "administración activa de una sociedad",
    "NET_WORTH_LIMIT_EXCEEDED": "límite de patrimonio neto del IMV",
    "NON_CORPORATE_ASSET_LIMIT_EXCEEDED": "límite de activos no societarios",
    "IMV_INCOME_LIMIT_EXCEEDED": "umbral de ingresos del IMV",
    "PENSION_CAP_EXCLUDES_IMV": "límite relacionado con pensiones",
    "CAPI_INCOME_LIMIT_EXCEEDED": "umbral de ingresos del CAPI",
    "CAPI_NET_WORTH_LIMIT_EXCEEDED": "límite de patrimonio neto del CAPI",
    "CAPI_NON_CORPORATE_ASSET_LIMIT_EXCEEDED": "límite de activos del CAPI",
}


class IMVCalculation(BaseModel):
    """Exactly the fields returned by the deterministic IMV/CAPI engine."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["eligible", "ineligible"]
    unit_member_ids: list[str] = Field(min_length=1)
    imv_eligible: StrictBool
    guaranteed_income_monthly_eur: Decimal = Field(ge=0, allow_inf_nan=False)
    imv_monthly_eur: Decimal = Field(ge=0, allow_inf_nan=False)
    capi_eligible: StrictBool
    capi_monthly_eur: Decimal = Field(ge=0, allow_inf_nan=False)
    total_monthly_eur: Decimal = Field(ge=0, allow_inf_nan=False)
    failed_requirements: list[str]

    @model_validator(mode="after")
    def validate_result(self):
        if self.status != (
            "eligible" if (self.imv_eligible or self.capi_eligible) else "ineligible"
        ):
            raise ValueError("status contradicts engine eligibility flags")
        if not self.imv_eligible and self.imv_monthly_eur != 0:
            raise ValueError("IMV amount is nonzero when IMV is ineligible")
        if not self.capi_eligible and self.capi_monthly_eur != 0:
            raise ValueError("CAPI amount is nonzero when CAPI is ineligible")
        if self.total_monthly_eur.quantize(CENT, rounding=ROUND_HALF_UP) != (
            self.imv_monthly_eur + self.capi_monthly_eur
        ).quantize(CENT, rounding=ROUND_HALF_UP):
            raise ValueError("total does not match IMV + CAPI amounts")
        return self


def _eur(amount: Decimal) -> str:
    """Deterministic Spanish monetary formatting without locale dependencies."""
    value = f"{amount.quantize(CENT, rounding=ROUND_HALF_UP):,.2f}"
    return value.replace(",", "_").replace(".", ",").replace("_", ".")


def render_imv_result(result: IMVCalculation) -> str:
    """Render one fixed narrative from engine fields, with no model-generated text."""
    members = len(result.unit_member_ids)
    unit = "persona" if members == 1 else "personas"
    outcome = (
        "La simulación estima una prestación"
        if result.status == "eligible"
        else "La simulación no reconoce prestación"
    )
    text = (
        f"Para una unidad de convivencia de {members} {unit}, {outcome.lower()}. "
        f"El importe mensual estimado es de {_eur(result.total_monthly_eur)} €: "
        f"{_eur(result.imv_monthly_eur)} € de IMV y "
        f"{_eur(result.capi_monthly_eur)} € de CAPI. "
        f"El cálculo indica IMV {'aplicable' if result.imv_eligible else 'no aplicable'} "
        f"y CAPI {'aplicable' if result.capi_eligible else 'no aplicable'}."
    )
    if result.failed_requirements:
        reasons = [
            FAILED_CHECK_LABELS.get(code, f"código de comprobación {code}")
            for code in result.failed_requirements
        ]
        text += " Comprobaciones desfavorables registradas por el motor: " + "; ".join(reasons) + "."
    return (
        text + " Esta información es orientativa y no constituye una resolución "
        "administrativa de la Seguridad Social."
    )


@tool
def explicar_imv(resultado: IMVCalculation) -> str:
    """Convert a calculated IMV/CAPI result to plain Spanish with a fixed template.

    Use only with the complete structured response already returned by
    calcular_imv. Never invent monetary values or eligibility facts. This
    function does not calculate eligibility or call an LLM; it preserves the
    exact engine figures and includes a non-official-estimate disclaimer.

    Args:
        resultado: Entire structured response from calcular_imv, including
            eligibility, IMV/CAPI amounts, household IDs and failed checks.

    Returns:
        Deterministic natural-language explanation of the engine result.
    """
    # Validate direct calls too. ToolRegistry already checks model arguments.
    validated = IMVCalculation.model_validate(resultado)
    return render_imv_result(validated)
