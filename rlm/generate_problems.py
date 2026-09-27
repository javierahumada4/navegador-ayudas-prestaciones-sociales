"""Generate verifiable IMV/CAPI reasoning problems for phase 1.

This replaces the shipping example from the ARCA template.  Ground truth comes
from rlm.imv_engine, so the same deterministic implementation is used to label
train/test data and to test the verifier.

Recommended:
    uv run python -m rlm.generate_problems --n 800 --split train --out rlm/data/train.jsonl
    uv run python -m rlm.generate_problems --n 200 --split test --out rlm/data/test.jsonl
    uv run python -m rlm.generate_problems --n 100 --split ood --out rlm/data/test_ood.jsonl

The OOD split deliberately contains the "CAPI-only" family (no IMV because
income is too high for IMV, but still below CAPI's broader thresholds).  That
family is absent from train/test.
"""

from __future__ import annotations

import argparse
import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from rlm.imv_engine import (
    age_on,
    economic_limits,
    evaluate_imv_case,
    expected_answer,
    guaranteed_income_monthly,
    load_ruleset,
)

APP_DATE = date(2026, 9, 26)

def _D(value: Any) -> Decimal:
    return Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _eur_es(value: Decimal | str | float | int) -> str:
    """Spanish display format, e.g. Decimal('26409.60') -> '26.409,60 €'."""
    value = _money(_D(value))
    raw = f"{value:,.2f}"
    return raw.replace(",", "X").replace(".", ",").replace("X", ".") + " €"


def _annex_ii_multiplier_for_key(rules: dict[str, Any], key: str) -> Decimal:
    return _D(rules["patrimony"]["annex_ii"][key])


def build_rule_context(rules: dict[str, Any]) -> str:
    """Build the teacher rule sheet from the same ruleset used by the oracle.

    Keeping this generated rather than handwritten is intentional: the teacher
    and the deterministic reference implementation must never disagree because a
    threshold was updated in one place but not the other.
    """
    amounts = rules["amounts"]
    patrimony = rules["patrimony"]
    capi = rules["capi"]
    mono = rules["monoparental"]

    base_monthly = _D(amounts["single_adult_monthly_eur"])
    base_annual = base_monthly * Decimal(12)
    single_net = base_annual * _D(patrimony["single_adult_net_worth_multiplier_annual"])
    single_assets = base_annual * _D(
        patrimony["single_adult_non_corporate_assets_multiplier_annual"]
    )

    # Annex-II table. Each row is (human label, annex-II key).
    compositions = [
        ("1 adulto", "1,0"),
        ("1 adulto + 1 menor", "1,1"),
        ("1 adulto + 2 menores", "1,2"),
        ("1 adulto + 3 menores", "1,3"),
        ("1 adulto + 4 o más menores", "1,4+"),
        ("2 adultos", "2,0"),
        ("2 adultos + 1 menor", "2,1"),
        ("2 adultos + 2 menores", "2,2"),
        ("2 adultos + 3 o más menores", "2,3+"),
        ("3 adultos", "3,0"),
        ("3 adultos + 1 menor", "3,1"),
        ("3 adultos + 2 o más menores", "3,2+"),
        ("4 adultos", "4,0"),
        ("4 adultos + 1 o más menores", "4,1+"),
        ("otros", "other"),
    ]

    patrimony_lines = []
    for label, key in compositions:
        mult = _annex_ii_multiplier_for_key(rules, key)
        net = _money(single_net * mult)
        assets = _money(single_assets * mult)
        patrimony_lines.append(
            f"  - {label}: patrimonio neto < {_eur_es(net)}; "
            f"activos no societarios <= {_eur_es(assets)}."
        )

    # CAPI uses 300% of Annex-I ordinary guaranteed income (without the 22%
    # monoparental/disability supplements) and 150% of Annex-II net worth.
    capi_compositions = [
        ("1 adulto + 1 menor", 2, "1,1"),
        ("1 adulto + 2 menores", 3, "1,2"),
        ("1 adulto + 3 menores", 4, "1,3"),
        ("1 adulto + 4 o más menores", 5, "1,4+"),
        ("2 adultos + 1 menor", 3, "2,1"),
        ("2 adultos + 2 menores", 4, "2,2"),
        ("2 adultos + 3 o más menores", 5, "2,3+"),
        ("3 adultos + 1 menor", 4, "3,1"),
        ("3 adultos + 2 o más menores", 5, "3,2+"),
        ("4 adultos + 1 o más menores", 5, "4,1+"),
        ("otros con menores", 5, "other"),
    ]
    capi_lines = []
    add_pct = _D(amounts["additional_member_pct"])
    max_mult = _D(amounts["max_household_multiplier"])
    for label, member_count, annex_key in capi_compositions:
        annex_i = min(Decimal(1) + add_pct * Decimal(member_count - 1), max_mult)
        income_limit = _money(
            base_annual * annex_i * _D(capi["income_multiplier"])
        )
        net_mult = _annex_ii_multiplier_for_key(rules, annex_key)
        capi_net = _money(
            single_net * net_mult * _D(capi["net_worth_multiplier"])
        )
        capi_lines.append(
            f"  - {label}: ingresos anuales < {_eur_es(income_limit)}; "
            f"patrimonio neto < {_eur_es(capi_net)}."
        )

    disability_amount = _money(base_monthly * _D(amounts["disability_pct"]))
    mono_amount = _money(base_monthly * _D(amounts["monoparental_pct"]))
    disability_min = amounts["disability_min_percent"]
    inst_months = mono["other_parent_institution_min_months"]
    dep_grade = mono["dependency_min_grade"]

    return f"""REGLAS EXACTAS IMV/CAPI 2026 PARA RESOLVER ESTOS PROBLEMAS

1) RENTA GARANTIZADA E IMV
- Base de 1 adulto: {_eur_es(base_monthly)}/mes.
- Por cada miembro adicional: +{_D(amounts['additional_member_pct']) * 100:.0f}% de la base, hasta un máximo del {_D(amounts['max_household_multiplier']) * 100:.0f}% de la base.
- Importes ordinarios: 1 miembro {_eur_es(base_monthly)}/mes; 2 miembros {_eur_es(base_monthly * Decimal('1.30'))}/mes; 3 miembros {_eur_es(base_monthly * Decimal('1.60'))}/mes; 4 miembros {_eur_es(base_monthly * Decimal('1.90'))}/mes; 5 o más {_eur_es(base_monthly * Decimal('2.20'))}/mes.
- IMV mensual = renta garantizada final (incluidos complementos) - ingresos computables anuales/12.
- Solo hay IMV si la diferencia es >= {_eur_es(amounts['minimum_imv_monthly_eur'])}/mes. La igualdad a 10 € sí cumple.

2) DISCAPACIDAD
- Si el beneficiario individual tiene discapacidad reconocida >= {disability_min}%, o si cualquier miembro de la unidad tiene discapacidad reconocida >= {disability_min}%, se añade una sola vez +{_D(amounts['disability_pct']) * 100:.0f}% de la base de un adulto = {_eur_es(disability_amount)}/mes.

3) COMPLEMENTO MONOPARENTAL
- Añade una sola vez +{_D(amounts['monoparental_pct']) * 100:.0f}% de la base de un adulto = {_eur_es(mono_amount)}/mes cuando concurra alguno de estos supuestos:
  a) un solo adulto con uno o más descendientes hasta segundo grado menores, con guarda y custodia exclusiva;
  b) un solo adulto con menores en acogimiento familiar permanente o guarda con fines de adopción, siendo el único acogedor/guardador;
  c) el otro progenitor, guardador o acogedor está en prisión o centro hospitalario durante un periodo ininterrumpido >= {inst_months} meses;
  d) conviven exclusivamente progenitores/abuelos/guardadores/acogedores y menores, y uno de los adultos tiene dependencia grado >= {dep_grade}, incapacidad permanente absoluta o gran invalidez;
  e) unidad formada exclusivamente por una mujer víctima de violencia de género y sus descendientes hasta segundo grado menores bajo guarda/custodia, o menores en acogimiento permanente/guarda preadoptiva.

4) TOPE POR PENSIONES Y SUBSIDIO PARA MAYORES DE 52 AÑOS
- Suma, para todos los miembros de la unidad, las pensiones contributivas o no contributivas de la Seguridad Social y, en su caso, el subsidio de desempleo para mayores de 52 años. Usa el importe mensual conjunto incluyendo la parte proporcional de pagas extraordinarias.
- En los enunciados del dataset esta magnitud aparece como «Pensiones/subsidios sujetos al tope». Es un dato adicional al campo de «Ingresos computables anuales» y se usa para aplicar este tope específico.
- Si pensiones/subsidios mensuales >= renta garantizada mensual aplicable: NO hay derecho al IMV por este motivo. La igualdad excluye.
- Si 0 < pensiones/subsidios mensuales < renta garantizada y se cumplen los demás requisitos: primero calcula el IMV por renta = renta garantizada - ingresos computables anuales/12; después aplica IMV final = min(IMV por renta, renta garantizada - pensiones/subsidios mensuales).
- Este tope limita o excluye el IMV. El CAPI se decide por sus propios requisitos y umbrales del apartado CAPI.

5) PATRIMONIO Y TEST DE ACTIVOS PARA IMV
- Vivienda habitual excluida. El patrimonio neto debe ser ESTRICTAMENTE menor que el límite: igualdad => no elegible.
- Los activos no societarios deben ser <= al límite: solo superar el límite => no elegible.
""" + "\n".join(patrimony_lines) + f"""

6) CAPI (COMPLEMENTO DE AYUDA PARA LA INFANCIA)
- Requiere al menos un menor en la unidad. Puede concederse aunque el IMV sea 0.
- Ingresos: ESTRICTAMENTE < 300% del umbral ordinario del Anexo I (sin sumar los complementos de monoparentalidad/discapacidad).
- Patrimonio neto: ESTRICTAMENTE < 150% del límite del Anexo II.
- Activos no societarios: se aplica el MISMO límite del test de activos anterior (<= límite).
- Límites exactos de renta y patrimonio CAPI por composición:
""" + "\n".join(capi_lines) + f"""
- Cuantía CAPI por cada menor, según edad a 1 de enero: 115,00 €/mes si <3 años; 80,50 €/mes si 3-5 años; 57,50 €/mes si 6-17 años.

7) OTROS REQUISITOS USADOS EN EL DATASET
- Residencia legal y efectiva continuada: CADA miembro de la unidad debe cumplir en general >= {rules['requirements']['minimum_continuous_residence_months']} meses. Comprueba la fecha individual de cada miembro; basta con que uno no cumpla para que falle este requisito.
- Beneficiario individual: edad mínima general {rules['requirements']['individual_min_age']} años.
- Si el beneficiario individual es menor de {rules['requirements']['independence_under_age']} años: debe haber vivido en domicilio distinto al de sus progenitores/tutores durante al menos los {rules['requirements']['independence_under_age_months']} meses inmediatamente anteriores y acreditar al menos {rules['requirements']['independence_under_age_ss_months']} meses de alta en Seguridad Social dentro de ese periodo.
- Si el beneficiario individual tiene {rules['requirements']['independence_under_age']} años o más, el oracle de este dataset exige domicilio distinto al de progenitores/tutores durante al menos {rules['requirements']['independence_age_30_plus_months']} meses anteriores.
- Unidad de convivencia: constituida en general >= {rules['requirements']['minimum_household_formation_months']} meses.
- Ser administrador de derecho de una sociedad mercantil activa excluye tanto IMV como CAPI.

8) CONVENCIÓN EXACTA DE REDONDEO DEL ORACLE
- Toda operación de redondeo monetario usa ROUND_HALF_UP a 2 decimales (céntimos). Ejemplo: 474,005 -> 474,01.
- Renta garantizada ordinaria de la unidad: aplica el multiplicador por número de miembros y redondea a céntimos HALF_UP.
- Complementos de monoparentalidad/discapacidad: se calculan como porcentaje de la base de un adulto; NO se redondea cada complemento por separado. Se suman a la renta garantizada ordinaria y entonces se redondea la renta garantizada final a céntimos HALF_UP.
- Ingreso mensual para IMV: ingresos computables anuales / 12, SIN redondear antes de restarlo.
- IMV por renta: renta garantizada final - ingreso mensual. Comprueba sobre ese valor sin redondear si la diferencia es >= 10,00 €.
- Si hay pensiones/subsidios sujetos al tope, aplica el mínimo indicado en el apartado 4 antes del redondeo final del IMV.
- IMV final: redondea a céntimos HALF_UP después de aplicar el tope de pensiones, si existe.
- CAPI: suma las cuantías fijas correspondientes a todos los menores y redondea el total CAPI a céntimos HALF_UP.
- TOTAL mensual pedido: suma el IMV final ya redondeado + el CAPI total ya redondeado y redondea de nuevo a céntimos HALF_UP.
- Los límites monetarios calculados por el oracle (patrimonio, activos y umbrales CAPI) también se redondean a céntimos HALF_UP antes de compararlos con el caso.

La respuesta final pedida en el dataset es TOTAL mensual = IMV + CAPI, con exactamente dos decimales."""



@dataclass
class Problem:
    question: str
    answer: str
    params: dict[str, Any]
    template_id: int
    branches: dict[str, str] = field(default_factory=dict)
    rule_context: str = ""


class ProblemGenerator(ABC):
    name: str = "generator"

    @abstractmethod
    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        ...

    @abstractmethod
    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        ...

    def key(self, params: dict[str, Any]) -> str:
        return json.dumps(params, sort_keys=True, ensure_ascii=False)

    def generate(self, n: int, split: str, seed: int = 0) -> list[Problem]:
        rng = random.Random(f"{seed}-{split}")
        seen: set[str] = set()
        problems: list[Problem] = []
        attempts = 0
        while len(problems) < n and attempts < 300 * n:
            attempts += 1
            params = self.sample_params(rng, split)
            fingerprint = self.key(params)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            answer, branches = self.solve(params)
            question, template_id = self.render(params, rng)
            problems.append(Problem(question, answer, params, template_id, branches, self.rule_context))
        if len(problems) < n:
            raise RuntimeError(
                f"only {len(problems)} unique problems after {attempts} attempts"
            )
        return problems


def _date_for_age(age: int, rng: random.Random) -> str:
    # Always before application date birthday ambiguity: sample month/day safely.
    year = APP_DATE.year - age - rng.choice([0, 1])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    # Adjust until exact age is the requested one.
    candidate = date(year, month, day)
    actual = APP_DATE.year - candidate.year - (
        (APP_DATE.month, APP_DATE.day) < (candidate.month, candidate.day)
    )
    if actual < age:
        candidate = date(candidate.year - 1, month, day)
    elif actual > age:
        candidate = date(candidate.year + 1, month, day)
    return candidate.isoformat()


def _person(
    pid: str,
    age: int,
    rng: random.Random,
    *,
    disability: int = 0,
    residence_since: str = "2018-01-01",
) -> dict[str, Any]:
    return {
        "id": pid,
        "date_of_birth": _date_for_age(age, rng),
        "emancipated": False,
        "legal_residence_in_spain": True,
        "effective_residence_in_spain": True,
        "continuous_legal_effective_residence_since": residence_since,
        "gender_violence_victim": False,
        "trafficking_victim": False,
        "homeless": False,
        "parents_or_guardians_deceased": False,
        "disability_percent": disability,
        "dependency_grade": 0,
        "permanent_disability_degree": "none",
        "active_company_director": False,
        "residential_service": "none",
        "benefits": [],
    }


def _fmt_eur(value: float) -> str:
    if abs(float(value)) < 0.005:
        return "0 €"
    return f"{float(value):.2f} €"


class IMVProblemGenerator(ProblemGenerator):
    name = "imv_2026"

    def __init__(self) -> None:
        self.rules = load_ruleset()
        self.rule_context = build_rule_context(self.rules)

    def _base_case(
        self, rng: random.Random, adults: int, minors: int, family: str
    ) -> dict[str, Any]:
        persons: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        applicant_age = rng.randint(24, 55)
        if family == "individual":
            applicant_age = rng.choice([24, 27, 31, 42, 58])

        persons.append(_person("p1", applicant_age, rng))
        next_id = 2

        # Other adults.
        for i in range(adults - 1):
            pid = f"p{next_id}"
            next_id += 1
            persons.append(_person(pid, rng.randint(24, 65), rng))
            relationships.append(
                {
                    "person_a": "p1",
                    "person_b": pid,
                    "type": "spouse" if i == 0 else "consanguinity",
                    "degree": None if i == 0 else rng.choice([1, 2]),
                }
            )

        # Minors.
        minor_ids = []
        for _ in range(minors):
            pid = f"p{next_id}"
            next_id += 1
            age = rng.choice([1, 2, 4, 5, 7, 10, 14, 17])
            persons.append(_person(pid, age, rng))
            minor_ids.append(pid)
            relationships.append(
                {
                    "person_a": "p1",
                    "person_b": pid,
                    "type": "parent_child",
                    "degree": 1,
                    # Non-monoparental synthetic cases always make the
                    # custody fact explicit.  A one-adult household can still
                    # have shared/non-exclusive custody with a parent who does
                    # not live in the unit; this prevents the statement from
                    # being ambiguous about the 22% monoparental complement.
                    "custody": "shared",
                }
            )

        case = {
            "application_date": APP_DATE.isoformat(),
            "applicant_id": "p1",
            "persons": persons,
            "domicile": {
                "co_resident_ids": [p["id"] for p in persons],
                "same_domicile_since": "2023-01-01",
                "caregiver_family_only": True,
            },
            "relationships": relationships,
            "applicant_history": {
                "independent_from_parents_since": "2022-01-01",
            },
            "social_security_registration_periods": [
                {
                    "person_id": "p1",
                    "from": "2023-01-01",
                    "to": None,
                }
            ],
            "economic": {
                "countable_income_annual_eur": 0.0,
                "net_worth_eur": 0.0,
                "non_corporate_assets_eur": 0.0,
            },
            "_family": family,
        }
        return case

    def sample_params(self, rng: random.Random, split: str) -> dict[str, Any]:
        if split == "ood":
            family = "capi_only"
        else:
            family = rng.choices(
                [
                    "ordinary",
                    "individual",
                    "monoparental",
                    "disability",
                    "income_fail",
                    "residence_fail",
                    "patrimony_fail",
                    "director_fail",
                ],
                weights=[20, 13, 14, 10, 12, 8, 12, 6],
            )[0]

        if family == "individual":
            adults, minors = 1, 0
        elif family == "monoparental":
            adults, minors = 1, rng.randint(1, 3)
        elif family == "capi_only":
            adults, minors = rng.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
        else:
            adults = rng.randint(1, 3)
            minors = rng.randint(0, 3)
            # One-adult + minor cases outside the monoparental family remain
            # non-monoparental because _base_case marks custody as shared.

        case = self._base_case(rng, adults, minors, family)

        if family == "monoparental":
            for rel in case["relationships"]:
                if rel["type"] == "parent_child":
                    rel["custody"] = "exclusive"

        if family == "disability":
            chosen = rng.choice(case["persons"])
            chosen["disability_percent"] = rng.choice([65, 70, 80])

        if family == "residence_fail":
            chosen = rng.choice(case["persons"])
            chosen["continuous_legal_effective_residence_since"] = rng.choice(
                ["2026-01-15", "2026-03-01", "2026-06-01"]
            )

        if family == "director_fail":
            rng.choice(case["persons"])["active_company_director"] = True

        # Sample economics relative to the actual guaranteed amount and limits.
        unit = [p["id"] for p in case["persons"]]
        guaranteed = guaranteed_income_monthly(case, unit, self.rules)
        limits = economic_limits(case, self.rules)

        if family == "income_fail":
            # Difference is below the statutory minimum 10 €/month.
            monthly_income = max(0.0, float(guaranteed) - rng.uniform(0.0, 9.5))
        elif family == "capi_only":
            # Above ordinary IMV threshold, but below the CAPI 300% ceiling.
            monthly_income = float(guaranteed) + rng.uniform(50.0, 300.0)
            capi_monthly_limit = float(limits["capi_income"]) / 12
            monthly_income = min(monthly_income, capi_monthly_limit - 50.0)
        else:
            # Usually eligible by income; some other branch may still exclude.
            gap = rng.uniform(40.0, min(650.0, float(guaranteed)))
            monthly_income = max(0.0, float(guaranteed) - gap)

        case["economic"]["countable_income_annual_eur"] = round(monthly_income * 12, 2)

        if family == "patrimony_fail":
            if rng.random() < 0.5:
                case["economic"]["net_worth_eur"] = float(limits["net_worth"])
            else:
                case["economic"]["non_corporate_assets_eur"] = round(
                    float(limits["assets"]) + rng.uniform(100, 5000), 2
                )
        else:
            case["economic"]["net_worth_eur"] = round(
                rng.uniform(0, max(1.0, float(limits["net_worth"]) * 0.45)), 2
            )
            case["economic"]["non_corporate_assets_eur"] = round(
                rng.uniform(0, max(1.0, float(limits["assets"]) * 0.45)), 2
            )

        # Occasionally exercise pension cap without making it a held-out family.
        if split != "ood" and family == "ordinary" and rng.random() < 0.15:
            case["persons"][0]["benefits"] = [
                {
                    "type": "pension",
                    "monthly_eur_with_extra_payments": round(
                        rng.uniform(150.0, float(guaranteed) * 0.6), 2
                    ),
                }
            ]

        return case

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        result = evaluate_imv_case(params, self.rules)
        branches = {
            "family": params["_family"],
            "status": result["status"],
            "imv_eligible": str(result["imv_eligible"]),
            "capi_eligible": str(result["capi_eligible"]),
            "has_disability_complement": str(
                any(
                    p["disability_percent"]
                    >= int(self.rules["amounts"]["disability_min_percent"])
                    for p in params["persons"]
                )
            ),
            "monoparental": str(
                params["_family"] == "monoparental"
            ),
        }
        return expected_answer(params, self.rules), branches

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        result = evaluate_imv_case(params, self.rules)
        persons = {p["id"]: p for p in params["persons"]}
        unit = result["unit_member_ids"]
        app_date = date.fromisoformat(params["application_date"])
        jan1 = date(app_date.year, 1, 1)

        # Use exactly the same age calculation as the oracle.  For minors we
        # expose both ages because CAPI uses age on 1 January while the
        # beneficiary/minor tests use age on the application date.
        adults = [pid for pid in unit if age_on(persons[pid]["date_of_birth"], app_date) >= 18]
        minors = [pid for pid in unit if pid not in adults]

        person_lines = []
        for pid in unit:
            p = persons[pid]
            age_application = age_on(p["date_of_birth"], app_date)
            disability = (
                f", discapacidad reconocida del {p['disability_percent']} %"
                if p["disability_percent"]
                else ""
            )
            if pid in minors:
                age_jan1 = age_on(p["date_of_birth"], jan1)
                person_lines.append(
                    f"{pid}: nacido el {p['date_of_birth']}, {age_application} años "
                    f"a fecha de solicitud y {age_jan1} años a 1 de enero de "
                    f"{app_date.year}{disability}"
                )
            else:
                person_lines.append(
                    f"{pid}: nacido el {p['date_of_birth']}, {age_application} años "
                    f"a fecha de solicitud{disability}"
                )

        relationship_bits = []
        custody_labels = {
            "shared": "compartida (no exclusiva)",
            "exclusive": "exclusiva",
            "none": "no indicada",
        }
        for rel in params["relationships"]:
            if rel["type"] == "parent_child":
                custody = rel.get("custody", "none")
                relationship_bits.append(
                    f"{rel['person_a']} y {rel['person_b']} tienen relación progenitor-hijo "
                    f"con custodia {custody_labels.get(custody, custody)}"
                )
            elif rel["type"] == "spouse":
                relationship_bits.append(f"{rel['person_a']} y {rel['person_b']} son cónyuges")
            elif rel["type"] == "consanguinity":
                relationship_bits.append(
                    f"{rel['person_a']} y {rel['person_b']} tienen parentesco de "
                    f"{rel.get('degree', 1)}º grado"
                )

        # Residence is a per-member requirement in the oracle.  Never collapse
        # it to one aggregate date, otherwise residence_fail can hide the very
        # person that makes the case ineligible.
        residence_bits = []
        for pid in unit:
            p = persons[pid]
            legal_effective = (
                p.get("legal_residence_in_spain", False)
                and p.get("effective_residence_in_spain", False)
            )
            residence_bits.append(
                f"{pid}: {'sí' if legal_effective else 'no'}, desde "
                f"{p['continuous_legal_effective_residence_since']}"
            )

        # Independence facts are only relevant for an individual beneficiary,
        # and every fact used by _individual_independence_ok must be observable
        # in the natural-language problem.
        independence_text = ""
        if len(unit) == 1:
            independent_since = params.get("applicant_history", {}).get(
                "independent_from_parents_since"
            )
            periods = [
                period
                for period in params.get("social_security_registration_periods", [])
                if period.get("person_id") == params["applicant_id"]
            ]
            if periods:
                period_bits = []
                for period in periods:
                    end = period.get("to") or params["application_date"]
                    period_bits.append(f"{period['from']} a {end}")
                ss_text = "; ".join(period_bits)
            else:
                ss_text = "ninguno"
            independence_text = (
                "Independencia del solicitante: domicilio distinto al de sus "
                f"progenitores/tutores desde {independent_since or 'fecha no acreditada'}. "
                f"Periodos de alta en Seguridad Social: {ss_text}. "
            )

        econ = params["economic"]
        benefits = sum(
            b.get("monthly_eur_with_extra_payments", 0)
            for pid in unit
            for b in persons[pid].get("benefits", [])
        )

        facts = (
            f"Solicitud a fecha {params['application_date']}. Solicitante: {params['applicant_id']}. "
            f"Conviven {len(unit)} personas en el mismo domicilio desde "
            f"{params['domicile']['same_domicile_since']}. "
            f"Personas: {'; '.join(person_lines)}. "
            f"Relaciones: {'; '.join(relationship_bits) if relationship_bits else 'ninguna relevante'}. "
            f"Residencia legal y efectiva continuada por miembro: {'; '.join(residence_bits)}. "
            f"{independence_text}"
            f"Ingresos computables anuales de la unidad: "
            f"{_fmt_eur(econ['countable_income_annual_eur'])}. "
            f"Patrimonio neto sin vivienda habitual: {_fmt_eur(econ['net_worth_eur'])}. "
            f"Activos no societarios sin vivienda habitual: "
            f"{_fmt_eur(econ['non_corporate_assets_eur'])}. "
            f"{'Hay un administrador de una sociedad mercantil activa. ' if any(p['active_company_director'] for p in params['persons']) else ''}"
            f"{f'Pensiones/subsidios sujetos al tope: {_fmt_eur(benefits)}/mes. ' if benefits else ''}"
        )

        templates = [
            facts
            + "Determina primero si hay derecho al IMV y/o al CAPI y calcula el total mensual a percibir. "
            "Responde solo con el importe total en euros, con dos decimales.",
            "Caso para resolver sobre IMV 2026. "
            + facts
            + "¿Cuál sería la suma mensual de IMV y CAPI? Da únicamente el número con dos decimales.",
            facts
            + "Aplica las reglas del IMV y del complemento de ayuda para la infancia. "
            "Si no corresponde ninguna de las dos prestaciones, indica cero euros. "
            "Indica el total mensual con dos decimales.",
            "Analiza esta solicitud de IMV/CAPI: "
            + facts
            + "Calcula el importe mensual final (IMV + CAPI) y contesta únicamente con la cifra.",
        ]
        idx = rng.randrange(len(templates))
        return templates[idx], idx


def describe(problems: list[Problem]) -> dict[str, Any]:
    counters: dict[str, Counter] = {}
    for problem in problems:
        for branch, value in problem.branches.items():
            counters.setdefault(branch, Counter())[value] += 1
    leaked = sum(1 for p in problems if p.answer in p.question)
    return {
        "n_problems": len(problems),
        "n_templates": len({p.template_id for p in problems}),
        "branches": {k: dict(v) for k, v in counters.items()},
        "answer_leaked_in_statement": leaked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--split", choices=["train", "test", "ood"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="rlm/data/train.jsonl")
    parser.add_argument(
        "--with-rules",
        action="store_true",
        help="prepend the IMV 2026 rule sheet to the student question",
    )
    args = parser.parse_args()

    generator = IMVProblemGenerator()
    problems = generator.generate(args.n, args.split, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for problem in problems:
            row = asdict(problem)
            if args.with_rules:
                row["question"] = f"{problem.rule_context}\n\n{problem.question}"
            row["split"] = args.split
            row["label_source"] = "imv_reference_engine"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(describe(problems), indent=2, ensure_ascii=False))
    print(f"\n{len(problems)} problemas -> {out}")
    print(f"Ejemplo:\n{problems[0].question}\nRespuesta: {problems[0].answer}")


if __name__ == "__main__":
    main()
