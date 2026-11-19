"""Corpus building and ingestion: offline checks on parsing, stable ids and the committed corpus."""

import json
from pathlib import Path

import pytest

from rag.fetch_corpus import assemble_norm, html_to_markdown, parse_block, version_in_force
from rag.ingest import CORPUS_DIR, document_id, load_corpus

BLOCK_XML = """<?xml version="1.0" encoding="utf-8"?>
<response><data>
<bloque id="a1-5" tipo="precepto" titulo="Artículo&#160;13">
  <version id_norma="BOE-A-2021-21007" fecha_publicacion="20211221" fecha_vigencia="20220101">
    <p class="articulo">Artículo 13. Texto original.</p>
  </version>
  <version id_norma="BOE-A-2024-10235" fecha_publicacion="20240522" fecha_vigencia="20240601">
    <p class="articulo">Artículo 13. Texto reformado.</p>
    <table><tr><th>Tipo</th><th>Escala</th></tr><tr><td>Un adulto</td><td>1,0</td></tr>
    <tr><td>Dos adultos</td><td>1,4</td></tr></table>
  </version>
  <version id_norma="BOE-A-2030-1" fecha_publicacion="20300101" fecha_vigencia="20300102">
    <p class="articulo">Artículo 13. Texto futuro.</p>
  </version>
</bloque></data></response>""".encode()


def test_parse_block_picks_version_in_force_and_renders_tables_by_row():
    block_type, title, text = parse_block(BLOCK_XML, today="20260101")
    assert (block_type, title) == ("precepto", "Artículo 13")
    assert "Texto reformado" in text and "original" not in text and "futuro" not in text
    assert "Un adulto | 1,0\nDos adultos | 1,4" in text


def test_version_in_force_falls_back_to_earliest_when_all_are_future():
    import xml.etree.ElementTree as ET

    versions = [ET.Element("version", fecha_vigencia=d) for d in ("20300101", "20310101")]
    assert version_in_force(versions, "20260101") is versions[0]


def test_assemble_norm_glues_headings_and_skips_signature():
    blocks = [
        ("encabezado", "CAPÍTULO I", "CAPÍTULO I\nDisposiciones generales"),
        ("precepto", "Artículo 1", "Artículo 1. Objeto.\nTexto."),
        ("firma", "[firma]", "Madrid, 20 de diciembre de 2021."),
    ]
    text = assemble_norm("Ley de prueba", blocks)
    assert (
        text
        == "# Ley de prueba\n\nCAPÍTULO I\nDisposiciones generales\nArtículo 1. Objeto.\nTexto.\n"
    )


def test_html_to_markdown_keeps_main_content_only():
    html = (
        "<body><nav><li>Menú</li></nav><script>var x=1;</script>"
        '<main id="content"><h1>Ingreso</h1><h2>Cuantías</h2><p>En 2026 son 733,6 euros.</p>'
        "<ul><li>Un adulto</li></ul><table><tr><td>Un adulto</td><td>733,60</td></tr></table>"
        "</main><footer>Pie</footer></body>"
    )
    text = html_to_markdown(html, "Título")
    assert text.startswith("# Título\n\n## Cuantías")
    assert "En 2026 son 733,6 euros." in text and "Un adulto | 733,60" in text
    assert "Menú" not in text and "var x" not in text and "Pie" not in text


def test_document_id_is_a_readable_function_of_the_path_only():
    assert document_id(Path("boe/Ley 19-2021.md")) == "boe-ley-19-2021"
    assert document_id(Path("ley-19-2021.md")) == "ley-19-2021"


def test_corpus_ids_are_stable_and_readme_is_not_ingested(tmp_path):
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "README.md").write_text("about the corpus", encoding="utf-8")
    first = [(d.id, d.source) for d in load_corpus(tmp_path)]
    assert first == [("a", "a.txt"), ("b", "b.md")]
    assert first == [(d.id, d.source) for d in load_corpus(tmp_path)]


def test_committed_corpus_matches_its_manifest():
    manifest = json.loads((CORPUS_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    documents = {d.source: d for d in load_corpus()}
    assert {m["file"] for m in manifest} == set(documents)
    assert len(documents) >= 6
    for entry in manifest:
        assert entry["license"] and entry["source_url"].startswith("https://")
        assert documents[entry["file"]].metadata["chars"] == len(
            (CORPUS_DIR / entry["file"]).read_text(encoding="utf-8").strip()
        )


@pytest.mark.parametrize("fragment", ["733,6 euros", "Artículo 13. Determinación de la cuantía"])
def test_corpus_holds_the_texts_the_questions_need(fragment):
    assert any(fragment in d.text for d in load_corpus())
