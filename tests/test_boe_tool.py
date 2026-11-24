"""consultar_boe: block resolution and XML parsing, without network."""

import os

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
    block = parse_boe_block(BLOCK_XML, today="20260601")
    assert block["titulo"] == "Artículo 13"
    assert block["vigente_desde"] == "20260410"
    assert block["modificado_por"] == "BOE-A-2026-7967"
    assert block["numero_versiones"] == 2
    assert block["en_vigor"] is True
    assert "Texto antiguo" not in block["texto"]
    assert "1. Incremento del 30 por ciento por miembro adicional." in block["texto"]
    assert block["truncado"] is False


def test_parse_boe_block_ignores_versions_not_yet_in_force():
    # Before 2026-04-10 the amendment is published but not applicable: the old text rules.
    block = parse_boe_block(BLOCK_XML, today="20260301")
    assert block["vigente_desde"] == "20220101"
    assert "Texto antiguo" in block["texto"]
    assert block["en_vigor"] is True
    assert block["numero_versiones"] == 2


def test_parse_boe_block_flags_a_block_with_no_version_in_force_yet():
    block = parse_boe_block(BLOCK_XML, today="20200101")
    assert block["en_vigor"] is False
    assert block["vigente_desde"] == "20220101"


TABLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><data>
<bloque id="ai-2" tipo="encabezado" titulo="ANEXO II">
  <version id_norma="BOE-A-2021-21007" fecha_publicacion="20211221" fecha_vigencia="20220101">
    <p class="anexo_tit">Escala de incrementos</p>
    <table class="tabla">
      <colgroup><col width="35%"/><col width="65%"/></colgroup>
      <thead><tr><th> </th><th>Escala de incrementos</th></tr></thead>
      <tbody>
        <tr><td>Un adulto solo.</td><td>16.614 € (tres veces la renta garantizada)</td></tr>
        <tr><td>Un adulto y un menor.</td><td>1,4</td></tr>
      </tbody>
    </table>
    <p class="nota_pie">Ref. BOE-A-2022-6830</p>
  </version>
</bloque></data></response>"""


def test_parse_boe_block_keeps_table_rows_apart():
    text = parse_boe_block(TABLE_XML, today="20260601")["texto"]
    assert text.splitlines() == [
        "Escala de incrementos",
        "| Escala de incrementos",
        "Un adulto solo. | 16.614 € (tres veces la renta garantizada)",
        "Un adulto y un menor. | 1,4",
        "Ref. BOE-A-2022-6830",
    ]


def test_parse_boe_block_caps_text_by_default_and_keeps_it_whole_on_request():
    long_xml = BLOCK_XML.replace(
        "Determinación de la cuantía.", "Determinación de la cuantía. " + "x" * 9000
    )
    capped = parse_boe_block(long_xml, today="20260601")
    assert capped["truncado"] is True and len(capped["texto"]) == tools.BOE_MAX_CHARS
    whole = parse_boe_block(long_xml, max_chars=None, today="20260601")
    assert whole["truncado"] is False and len(whole["texto"]) > 9000


def test_parse_boe_block_reads_utf8_bytes_without_mojibake():
    block = parse_boe_block(BLOCK_XML.encode("utf-8"), today="20260601")
    assert block["titulo"] == "Artículo 13"
    assert "Determinación de la cuantía" in block["texto"]


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_fetch_boe_block_has_no_text_cap_unless_asked(monkeypatch):
    long_xml = BLOCK_XML.replace("por miembro adicional.", "por miembro adicional. " + "x" * 9000)
    monkeypatch.setattr(tools, "_boe_index", lambda norma: tuple(INDEX))
    monkeypatch.setattr(
        tools.requests, "get", lambda *a, **k: _FakeResponse(long_xml.encode("utf-8"))
    )
    whole = tools.fetch_boe_block("13")
    assert whole["truncado"] is False and len(whole["texto"]) > 9000
    # The model-facing tool keeps the cap.
    capped = tools.consultar_boe(articulo="13")
    assert capped["truncado"] is True and len(capped["texto"]) == tools.BOE_MAX_CHARS


@pytest.mark.skipif(
    not os.environ.get("ARCA_NETWORK_TESTS"),
    reason="hits the real BOE API: set ARCA_NETWORK_TESTS=1 to run it",
)
def test_consultar_boe_returns_real_article_text_not_just_metadata():
    tools._boe_index.cache_clear()
    article = tools.fetch_boe_block("13")
    assert article["titulo"] == "Artículo 13"
    assert len(article["texto"]) > 1000, "got metadata instead of the article body"
    assert "renta garantizada" in article["texto"]
    annex = tools.fetch_boe_block("Anexo II")
    assert "Un adulto y un menor. | 1,4" in annex["texto"]


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
