"""CPU-only tests of the BOE and ruleset tools. No live network or GPU."""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import httpx
import pytest

from rlm.imv_engine import IMVInputError, evaluate_imv_case
from rlm.ruleset_catalog import inspect_rulesets
from tool_use.ruleset_tools import consultar_boe
from tool_use.tools import default_registry

RULES_2026 = Path(__file__).resolve().parents[1] / "rlm/rulesets/imv_2026.json"
BENCH = Path(__file__).resolve().parents[1] / "tool_use/bench/imv_smoke_cases.jsonl"


def case_from_bench():
    first = json.loads(BENCH.read_text(encoding="utf-8").splitlines()[0])
    return first["expected_arguments"]["case"]


def test_domain_registry_has_read_tools_and_confirmation_gate():
    tools = default_registry().tools
    for name in ("fecha_actual", "comprobar_ruleset", "consultar_boe", "crear_ruleset",
                 "calcular_imv"):
        assert name in tools
    assert tools["crear_ruleset"].requires_confirmation
    assert not tools["consultar_boe"].requires_confirmation


def test_current_date_is_madrid_calendar_date():
    result = default_registry().call("fecha_actual", {})
    assert result.ok, result.error
    date.fromisoformat(result.result["fecha_actual"])
    assert result.result["timezone"] == "Europe/Madrid"


def test_date_lookup_does_not_use_current_year_for_historical_application(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_RULESETS_DIR", str(tmp_path))
    (tmp_path / "imv_2026.json").write_bytes(RULES_2026.read_bytes())
    registry = default_registry()
    covered = registry.call("comprobar_ruleset", {"fecha_solicitud": "2026-09-26"})
    assert covered.ok, covered.error
    assert covered.result["status"] == "vigente"
    assert covered.result["ruleset"]["id"] == "imv_es_2026"
    missing = registry.call("comprobar_ruleset", {"fecha_solicitud": "2027-02-01"})
    assert missing.ok and missing.result["status"] == "no_encontrado"
    c = case_from_bench()
    c["application_date"] = "2027-02-01"
    with pytest.raises(IMVInputError, match="outside the loaded ruleset"):
        evaluate_imv_case(c)


def test_overlapping_or_corrupt_catalog_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_RULESETS_DIR", str(tmp_path))
    data = json.loads(RULES_2026.read_text(encoding="utf-8"))
    (tmp_path / "imv_2026.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "imv_2026_second.json").write_text(json.dumps(data), encoding="utf-8")
    assert inspect_rulesets(date(2026, 9, 26))["status"] == "ambiguo"
    (tmp_path / "imv_broken.json").write_text("{oops", encoding="utf-8")
    assert inspect_rulesets(date(2026, 9, 26))["status"] == "requiere_revision"


def test_boe_lookup_uses_actual_documented_endpoint_without_open_redirects(monkeypatch):
    called = {}

    def fake_get(url, *, headers, timeout, follow_redirects):
        called.update(url=url, headers=headers, timeout=timeout,
                      follow_redirects=follow_redirects)
        return httpx.Response(
            200, request=httpx.Request("GET", url),
            json={"status": "OK", "data": {"identificador": "BOE-A-2021-21007"}},
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    record = default_registry().call("consultar_boe", {"boe_id": "BOE-A-2021-21007"})
    assert record.ok, record.error
    assert "/legislacion-consolidada/id/BOE-A-2021-21007/metadatos" in called["url"]
    assert called["headers"]["Accept"] == "application/json"
    assert called["follow_redirects"] is False
    assert record.result["respuesta"]["data"]["identificador"] == "BOE-A-2021-21007"


@pytest.mark.parametrize("bad", ["https://evil.com/", "../BOE-A-2021-21007",
                                  "BOE-B-2026-123", "BOE-A-2026-123/../foo"])
def test_boe_rejects_invalid_document_id(bad):
    with pytest.raises(ValueError, match="BOE id"):
        consultar_boe(boe_id=bad)


def test_boe_http_errors_become_tool_observations(monkeypatch):
    def fail(url, *, headers, timeout, follow_redirects):
        return httpx.Response(503, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fail)
    record = default_registry().call("consultar_boe", {})
    assert not record.ok
    assert "503" in record.error


def test_boe_text_is_bounded_and_marked_truncated(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        return httpx.Response(200, request=httpx.Request("GET", url),
                              text="<texto>" + "x" * 15000 + "</texto>")

    monkeypatch.setattr(httpx, "get", fake_get)
    result = default_registry().call("consultar_boe", {"seccion": "texto"})
    assert result.ok, result.error
    assert result.result["truncado"]
    assert len(result.result["texto"]) == 12000


def test_ruleset_creation_needs_explicit_review_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_RULESETS_DIR", str(tmp_path))
    (tmp_path / "imv_2026.json").write_bytes(RULES_2026.read_bytes())
    candidate = deepcopy(json.loads(RULES_2026.read_text(encoding="utf-8")))
    # Synthetic TEST fixture only: deliberately reuse numeric parameters
    # to test date-selection mechanics, NOT as real 2027 legal rules.
    candidate.update(id="imv_es_2027", effective_from="2027-01-01",
                     effective_to="2027-12-31")
    registry = default_registry()
    refused = registry.call("crear_ruleset", {"ruleset": candidate})
    assert not refused.ok
    assert not (tmp_path / "imv_2027.json").exists()
    accepted = registry.call(
        "crear_ruleset", {"ruleset": candidate, "confirmacion_humana": True}
    )
    assert accepted.ok, accepted.error
    assert (tmp_path / "imv_2027.json").is_file()
    assert accepted.result["created"]
    c = case_from_bench()
    c["application_date"] = "2027-02-01"
    assert evaluate_imv_case(c)["imv_monthly_eur"] == 373.76
    duplicate = registry.call(
        "crear_ruleset", {"ruleset": candidate, "confirmacion_humana": True}
    )
    assert not duplicate.ok
    assert "overlaps" in duplicate.error


def test_ruleset_writer_rejects_incomplete_or_unofficial_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_RULESETS_DIR", str(tmp_path))
    candidate = json.loads(RULES_2026.read_text(encoding="utf-8"))
    candidate["id"] = "imv_es_2027"
    candidate["effective_from"] = "2027-01-01"
    candidate["effective_to"] = "2027-12-31"
    candidate["amounts"].pop("single_adult_monthly_eur")
    missing = default_registry().call(
        "crear_ruleset", {"ruleset": candidate, "confirmacion_humana": True}
    )
    assert not missing.ok
    assert not list(tmp_path.glob("imv_*.json"))
    candidate["amounts"]["single_adult_monthly_eur"] = "733.60"
    candidate["sources"][1]["url"] = "https://seg-social.es.evil.org/source"
    untrusted = default_registry().call(
        "crear_ruleset", {"ruleset": candidate, "confirmacion_humana": True}
    )
    assert not untrusted.ok
    assert not list(tmp_path.glob("imv_*.json"))
