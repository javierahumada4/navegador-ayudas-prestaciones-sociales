"""consultar_boe: block resolution and XML parsing, without network."""

import pytest

from tool_use import tools
from tool_use.registry import ToolRegistry
from tool_use.tools import consultar_boe, parse_boe_block, resolve_boe_block

INDEX = [
    {"id": "pr", "titulo": "[preambulo]"},
    {"id": "a1-2", "titulo": "Artículo\xa010"},
    {"id": "a1-5", "titulo": "Artículo\xa013"},
    {"id": "ai-2", "titulo": "ANEXO II"},
    {"id": "dt-7", "titulo": "Disposición transitoria séptima"},
]

BLOCK_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><status><code>200</code><text>ok</text></status><data>
<bloque id="a1-5" tipo="precepto" titulo="Artículo\xa013">
  <version id_norma="BOE-A-2021-21007" fecha_publicacion="20211221" fecha_vigencia="20220101">
    <p class="articulo">Artículo 13. Texto antiguo.</p>
  </version>
  <version id_norma="BOE-A-2026-7967" fecha_publicacion="20260409" fecha_vigencia="20260410">
    <p class="articulo">Artículo 13. Determinación de la cuantía.</p>
    <p class="parrafo">1. Incremento del <em>30 por ciento</em>
       por miembro adicional.</p>
  </version>
</bloque></data></response>"""


@pytest.mark.parametrize(
    ("query", "block_id"),
    [
        ("13", "a1-5"),
        ("art. 13", "a1-5"),
        ("Artículo 10", "a1-2"),
        ("articulo 10", "a1-2"),
        ("Anexo II", "ai-2"),
        ("disposición transitoria séptima", "dt-7"),
        ("a1-5", "a1-5"),
    ],
)
def test_resolve_boe_block_accepts_natural_names(query, block_id):
    assert resolve_boe_block(INDEX, query)["id"] == block_id


def test_resolve_boe_block_lists_available_blocks_on_miss():
    with pytest.raises(ValueError, match="Artículo 13"):
        resolve_boe_block(INDEX, "art. 99")


def test_parse_boe_block_returns_version_in_force_as_text():
    block = parse_boe_block(BLOCK_XML)
    assert block["titulo"] == "Artículo 13"
    assert block["vigente_desde"] == "20260410"
    assert block["modificado_por"] == "BOE-A-2026-7967"
    assert block["numero_versiones"] == 2
    assert "Texto antiguo" not in block["texto"]
    assert "1. Incremento del 30 por ciento por miembro adicional." in block["texto"]
    assert block["truncado"] is False


def test_consultar_boe_rejects_malformed_law_id_before_any_request(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("must not hit the network")

    monkeypatch.setattr(tools.requests, "get", no_network)
    record = ToolRegistry().register(consultar_boe).call(
        "consultar_boe", {"articulo": "13", "norma": "../../etc"}
    )
    assert not record.ok and "BOE-A-2021-21007" in record.error


def test_consultar_boe_definition_exposes_article_and_default_law():
    params = consultar_boe.definition["function"]["parameters"]
    assert params["required"] == ["articulo"]
    assert params["properties"]["norma"]["default"] == "BOE-A-2021-21007"
