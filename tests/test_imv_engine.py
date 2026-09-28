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


def test_rule_context_contains_exact_thresholds_requested_for_distillation():
    generator = IMVProblemGenerator()
    context = generator.rule_context

    # Exact IMV patrimony/assets thresholds, including strictness at the boundary.
    assert "1 adulto: patrimonio neto < 26.409,60 €" in context
    assert "activos no societarios <= 52.819,20 €" in context
    assert "2 adultos + 1 menor: patrimonio neto < 47.537,28 €" in context
    assert "activos no societarios <= 95.074,56 €" in context

    # Disability threshold and amount are explicit.
    assert "discapacidad reconocida >= 65%" in context
    assert "+22% de la base de un adulto = 161,39 €/mes" in context

    # Monoparental conditions are operational, not just named.
    assert "guarda y custodia exclusiva" in context
    assert "periodo ininterrumpido >= 12 meses" in context
    assert "dependencia grado >= 3" in context
    assert "incapacidad permanente absoluta o gran invalidez" in context
    assert "víctima de violencia de género" in context

    # Exact CAPI thresholds by composition.
    assert "1 adulto + 1 menor: ingresos anuales < 34.332,48 €" in context
    assert "patrimonio neto < 55.460,16 €" in context
    assert "2 adultos + 2 menores: ingresos anuales < 50.178,24 €" in context
    assert "patrimonio neto < 87.151,68 €" in context


def test_rule_context_is_generated_from_ruleset_not_hardcoded():
    from rlm.generate_problems import build_rule_context

    future_rules = load_ruleset()
    future_rules["amounts"]["disability_min_percent"] = 40
    context = build_rule_context(future_rules)
    assert "discapacidad reconocida >= 40%" in context
    assert "discapacidad reconocida >= 65%" not in context


def test_rule_context_contains_pension_cap_and_rounding_convention():
    context = IMVProblemGenerator().rule_context
    assert "TOPE POR PENSIONES Y SUBSIDIO PARA MAYORES DE 52 AÑOS" in context
    assert "pensiones/subsidios mensuales >= renta garantizada" in context
    assert "IMV final = min(IMV por renta, renta garantizada - pensiones/subsidios mensuales)" in context
    assert "ROUND_HALF_UP" in context
    assert "474,005 -> 474,01" in context
    assert "ingresos computables anuales / 12, SIN redondear" in context
    assert "NO se redondea cada complemento por separado" in context


def test_pension_cap_reduces_imv_but_does_not_by_itself_remove_capi():
    case = reference_case()
    case["persons"][0]["benefits"] = [
        {"type": "pension", "monthly_eur_with_extra_payments": 1000.00}
    ]
    result = evaluate_imv_case(case)
    # RG 1173.76, raw IMV 373.76, pension cap 173.76 -> IMV 173.76.
    assert result["imv_eligible"]
    assert result["imv_monthly_eur"] == 173.76
    assert result["capi_eligible"]
    assert result["total_monthly_eur"] == 231.26


def test_pension_equal_to_guaranteed_income_excludes_imv():
    case = reference_case()
    case["persons"][0]["benefits"] = [
        {"type": "pension", "monthly_eur_with_extra_payments": 1173.76}
    ]
    result = evaluate_imv_case(case)
    assert not result["imv_eligible"]
    assert "PENSION_CAP_EXCLUDES_IMV" in result["failed_requirements"]
    # CAPI is still evaluated independently by the oracle.
    assert result["capi_eligible"]
    assert result["total_monthly_eur"] == 57.50


def test_oracle_rounds_final_imv_half_up_after_unrounded_monthly_income():
    case = reference_case()
    # 8397.06 / 12 = 699.755 exactly. RG 1173.76 - 699.755 = 474.005.
    case["economic"]["countable_income_annual_eur"] = 8397.06
    result = evaluate_imv_case(case)
    assert result["imv_monthly_eur"] == 474.01
    assert result["capi_monthly_eur"] == 57.50
    assert result["total_monthly_eur"] == 531.51


def test_render_uses_exact_application_age_and_explicit_capi_jan1_age():
    import random

    generator = IMVProblemGenerator()
    case = generator._base_case(random.Random(1), adults=1, minors=1, family="ordinary")
    # Applicant birthday is after the application date: exact age is 25, not 26.
    case["persons"][0]["date_of_birth"] = "2000-10-10"
    # Child turns 3 between 1 January and the application date. CAPI must use age 2.
    child = next(p for p in case["persons"] if p["id"] != "p1")
    child["date_of_birth"] = "2023-05-01"
    question, _ = generator.render(case, random.Random(2))

    assert "nacido el 2000-10-10, 25 años a fecha de solicitud" in question
    assert "nacido el 2023-05-01, 3 años a fecha de solicitud" in question
    assert "2 años a 1 de enero de 2026" in question
    # Regression guard: the old year-only calculation would have shown 26 for p1.
    assert "2000-10-10, 26 años" not in question


def test_residence_fail_exposes_each_members_residence_date():
    import random

    generator = IMVProblemGenerator()
    case = generator._base_case(random.Random(3), adults=2, minors=1, family="residence_fail")
    case["persons"][2]["continuous_legal_effective_residence_since"] = "2026-06-01"
    question, _ = generator.render(case, random.Random(4))

    assert "Residencia legal y efectiva continuada por miembro" in question
    assert "p1: sí, desde 2018-01-01" in question
    assert "p2: sí, desde 2018-01-01" in question
    assert "p3: sí, desde 2026-06-01" in question
    assert "La residencia legal y efectiva más reciente" not in question


def test_individual_question_exposes_independence_and_social_security_facts():
    import random

    generator = IMVProblemGenerator()
    case = generator._base_case(random.Random(5), adults=1, minors=0, family="individual")
    case["persons"][0]["date_of_birth"] = "1999-10-10"  # 26 on application date.
    case["applicant_history"]["independent_from_parents_since"] = "2023-09-26"
    case["social_security_registration_periods"] = [
        {"person_id": "p1", "from": "2025-01-15", "to": "2025-08-15"},
        {"person_id": "p1", "from": "2026-01-01", "to": None},
    ]
    question, _ = generator.render(case, random.Random(6))

    assert (
        "Independencia del solicitante: domicilio distinto al de sus "
        "progenitores/tutores desde 2023-09-26"
    ) in question
    assert "Periodos de alta en Seguridad Social:" in question
    assert "2025-01-15 a 2025-08-15" in question
    assert "2026-01-01 a 2026-09-26" in question


def test_rule_context_explains_individual_independence_thresholds():
    context = IMVProblemGenerator().rule_context
    assert "Beneficiario individual: edad mínima general 23 años" in context
    assert "menor de 30 años" in context
    assert "24 meses inmediatamente anteriores" in context
    assert "al menos 12 meses de alta en Seguridad Social" in context
    assert "30 años o más" in context
    assert "al menos 12 meses anteriores" in context


def test_one_adult_with_minors_is_explicitly_non_monoparental_unless_family_says_so():
    import random

    generator = IMVProblemGenerator()
    ordinary = generator._base_case(random.Random(7), adults=1, minors=2, family="ordinary")
    assert all(
        rel.get("custody") == "shared"
        for rel in ordinary["relationships"]
        if rel["type"] == "parent_child"
    )
    ordinary_question, _ = generator.render(ordinary, random.Random(8))
    assert "custodia compartida (no exclusiva)" in ordinary_question
    assert "custodia exclusiva" not in ordinary_question

    monoparental = generator._base_case(
        random.Random(9), adults=1, minors=2, family="monoparental"
    )
    for rel in monoparental["relationships"]:
        if rel["type"] == "parent_child":
            rel["custody"] = "exclusive"
    monoparental_question, _ = generator.render(monoparental, random.Random(10))
    assert "custodia exclusiva" in monoparental_question
