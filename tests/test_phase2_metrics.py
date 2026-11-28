"""Phase-2 metric details: free-text search arguments and clarification requests."""
from tool_use.evaluate import normalise, score_case
from tool_use.metrics import asks_for_clarification, score_workflow


def bdns_case(tema: str) -> dict:
    return {
        "needs_tool": True, "expected_tool": "consultar_ayudas",
        "expected_tools": ["consultar_ayudas"],
        "expected_arguments": {"tema": tema, "max_resultados": 3},
    }


def bdns_call(tema: str) -> dict:
    return {"name": "consultar_ayudas", "arguments": {"tema": tema, "max_resultados": 3},
            "ok": True, "result": {"resultados": []}}


def test_search_topic_ignores_accents_order_prepositions_and_grant_words():
    case = bdns_case("rehabilitación vivienda")
    for tema in ("rehabilitación de vivienda", "Vivienda rehabilitacion",
                 "subvenciones para rehabilitación de vivienda"):
        assert score_case(case, [bdns_call(tema)])["arguments_ok"], tema


def test_search_topic_still_needs_the_same_content_words():
    case = bdns_case("empleo juvenil")
    assert not score_case(case, [bdns_call("empleo")])["arguments_ok"]
    assert not score_case(case, [bdns_call("empleo juvenil rural")])["arguments_ok"]


def test_only_search_fields_get_the_lenient_comparison():
    assert normalise({"nombre": "Ana de la Fuente"}) == {"nombre": "ana de la fuente"}


def test_clarification_accepts_questions_and_request_verbs():
    assert asks_for_clarification("¿Cuál es la edad exacta de los menores?")
    assert asks_for_clarification("Puedes proporcionar el importe de la pensión")
    assert asks_for_clarification("Por favor, proporciona la edad de los menores.")
    assert not asks_for_clarification("No has proporcionado tu patrimonio. El IMV es 420 €.")


def test_clarification_case_requires_the_missing_facts_and_no_call():
    case = {"needs_tool": False, "expected_behavior": "ask_clarification",
            "expected_clarifications": ["edad", "patrimon"]}
    asked = "¿Cuál es la edad exacta de los menores? ¿Cuál es el patrimonio familiar?"
    assert score_workflow(case, asked, [])["end_to_end_success"]
    assert not score_workflow(case, "¿Cuál es la edad de los menores?", [])["end_to_end_success"]
