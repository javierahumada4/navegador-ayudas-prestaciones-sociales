"""Rebuild ``rag/corpus/`` from its public sources.

The corpus is the legal framework of the Ingreso Mínimo Vital (IMV) and the Complemento de
Ayuda para la Infancia (CAPI): six consolidated texts from the BOE open-data API and the
Social Security page with the 2026 amounts. The files are committed (they are small), and this
script regenerates them::

    uv run python -m rag.fetch_corpus            # rewrite rag/corpus/*.md and MANIFEST.json

The BOE publishes *versioned* consolidated texts: each block (article, annex...) carries every
version it has had, some of them not in force yet. We keep the version in force on the day of
the download, render tables one row per line (flattening glues the cells into prose) and glue
chapter headings to the article that follows so a chunk never loses its context. The logic
mirrors the ``consultar_boe`` tool of phase 2, copied here so that ``rag/`` does not depend on
branches that are not merged into ``main``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).parent / "corpus"
BOE_API = "https://www.boe.es/datosabiertos/api/legislacion-consolidada/id"
USER_AGENT = "arca-rag-corpus/1.0 (academic project; ICAI)"
SKIPPED_BLOCK_TYPES = {"firma", "nota_inicial"}


@dataclass(frozen=True)
class BoeNorm:
    slug: str  # file name and chunk-id prefix, e.g. ``ley-19-2021``
    boe_id: str  # BOE-A-YYYY-NNNNN


@dataclass(frozen=True)
class WebPage:
    slug: str
    url: str
    title: str


BOE_NORMS = (
    BoeNorm("ley-19-2021", "BOE-A-2021-21007"),  # Ley 19/2021: the IMV law (and the CAPI)
    BoeNorm("rd-789-2022", "BOE-A-2022-15764"),  # compatibility with income from work
    BoeNorm("rd-635-2022", "BOE-A-2022-12508"),  # Consejo consultivo del IMV
    BoeNorm("rd-64-2022", "BOE-A-2022-1275"),  # Comisión de seguimiento del IMV
    BoeNorm("orden-ism-1375-2021", "BOE-A-2021-20316"),  # Registro de Mediadores Sociales
    BoeNorm("rdl-20-2020", "BOE-A-2020-5493"),  # original IMV decree-law, repealed (distractor)
)
WEB_PAGES = (
    WebPage(
        "ss-imv",
        "https://www.seg-social.es/wps/portal/wss/internet/Trabajadores/"
        "PrestacionesPensionesTrabajadores/65850d68-8d06-4645-bde7-05374ee42ac7",
        "Seguridad Social: Ingreso mínimo vital (requisitos, cuantías 2026 y solicitud)",
    ),
)


# --------------------------------------------------------------------------------- BOE parsing


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _flat_text(element: ET.Element) -> str:
    return _normalize("".join(element.itertext()))


def render_element(element: ET.Element) -> str:
    """Plain text of a version child: a paragraph is one line, a table is one line per row."""
    if element.tag != "table":
        return _flat_text(element)
    rows = []
    for row in element.iter("tr"):
        cells = [_flat_text(cell) for cell in row if cell.tag in ("th", "td")]
        line = " | ".join(cells).strip()
        if line:
            rows.append(line)
    return "\n".join(rows)


def version_in_force(versions: list[ET.Element], today: str) -> ET.Element:
    """Latest version whose ``fecha_vigencia`` (YYYYMMDD) is not after ``today``.

    If every version is still in the future the earliest is returned: the text exists even if
    it has not started to apply.
    """
    in_force = [v for v in versions if (v.get("fecha_vigencia") or "") <= today]
    return in_force[-1] if in_force else versions[0]


def parse_block(xml_bytes: bytes, today: str) -> tuple[str, str, str]:
    """``(block type, title, text)`` of the version in force of one BOE block."""
    block = ET.fromstring(xml_bytes).find("data/bloque")
    if block is None or not block.findall("version"):
        raise ValueError("unexpected BOE response: no block with versions")
    current = version_in_force(block.findall("version"), today)
    lines = [render_element(child) for child in current]
    return (
        block.get("tipo", ""),
        _normalize(block.get("titulo", "")),
        "\n".join(line for line in lines if line),
    )


def assemble_norm(title: str, blocks: list[tuple[str, str, str]], note: str = "") -> str:
    """Markdown of a whole norm: ``# title`` then one paragraph per block.

    Chapter headings (``encabezado`` blocks) are glued to the following block with a single
    newline, so the recursive splitter does not leave a heading alone at the end of a chunk.
    """
    parts = [f"# {title}"]
    if note:
        parts.append(note)
    pending = ""
    for block_type, _, text in blocks:
        if block_type in SKIPPED_BLOCK_TYPES or not text.strip():
            continue
        if block_type == "encabezado":
            pending += text + "\n"
            continue
        parts.append(pending + text)
        pending = ""
    if pending:
        parts.append(pending.rstrip())
    return "\n\n".join(parts) + "\n"


def fetch_norm(client: httpx.Client, norm: BoeNorm, today: str, pause: float = 0.1) -> dict:
    """Download one consolidated norm; returns its Markdown and provenance metadata."""
    meta = client.get(f"{BOE_API}/{norm.boe_id}/metadatos", headers={"Accept": "application/xml"})
    meta.raise_for_status()
    info = ET.fromstring(meta.content).find("data/metadatos")
    title = _normalize(info.findtext("titulo", ""))
    repealed = (
        info.findtext("estatus_derogacion") == "S" or info.findtext("vigencia_agotada") == "S"
    )

    index = client.get(
        f"{BOE_API}/{norm.boe_id}/texto/indice", headers={"Accept": "application/json"}
    )
    index.raise_for_status()
    data = index.json()["data"]
    entries = data["bloque"] if isinstance(data, dict) else data[0]["bloque"]

    blocks = []
    for entry in entries:
        response = client.get(
            f"{BOE_API}/{norm.boe_id}/texto/bloque/{entry['id']}",
            headers={"Accept": "application/xml"},
        )
        response.raise_for_status()
        blocks.append(parse_block(response.content, today))
        time.sleep(pause)
    note = (
        "Norma derogada: se conserva solo como texto histórico y ya no está en vigor."
        if repealed
        else ""
    )
    return {
        "file": f"{norm.slug}.md",
        "text": assemble_norm(title, blocks, note),
        "title": title,
        "source_url": f"https://www.boe.es/buscar/act.php?id={norm.boe_id}",
        "boe_id": norm.boe_id,
        "in_force": not repealed,
        "n_blocks": len(blocks),
        "license": "BOE open data: reuse allowed with attribution (Orden PRE/2065/2014)",
    }


# ------------------------------------------------------------------- Social Security web page


class _MarkdownExtractor(HTMLParser):
    """Turns the main content of the Social Security page into Markdown, dropping the chrome."""

    HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.buffer: list[str] = []
        self.in_main = False
        self.skip_depth = 0
        self.prefix = ""
        self.list_depth = 0

    def _flush(self) -> None:
        text = _normalize("".join(self.buffer)).removeprefix("| ")  # table rows start with a cell
        if text and self.in_main:
            self.lines.append(self.prefix + text)
        self.buffer.clear()
        self.prefix = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "main" and attrs.get("id") == "content":
            self.in_main = True
        if tag in ("script", "style", "noscript", "nav", "form", "button"):
            self.skip_depth += 1
            return
        if tag in self.HEADINGS:
            self._flush()
            self.prefix = self.HEADINGS[tag]
        elif tag == "li":
            self._flush()
            self.list_depth += 1
            self.prefix = "  " * (self.list_depth - 1) + "- "
        elif tag in ("p", "br", "tr", "div"):
            self._flush()
        elif tag in ("td", "th"):
            self.buffer.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "nav", "form", "button"):
            self.skip_depth = max(self.skip_depth - 1, 0)
            return
        if tag in self.HEADINGS or tag in ("p", "li", "tr"):
            self._flush()
        if tag == "li":
            self.list_depth = max(self.list_depth - 1, 0)
        if tag == "main":
            self._flush()
            self.in_main = False

    def handle_data(self, data):
        if not self.skip_depth:
            self.buffer.append(data)


PAGE_NOISE = re.compile(
    r"^(Z6_\w+( Z6_\w+)*|[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}|\{\}|Valore este contenido"
    r"|Encuesta de satisfacción|Gracias por su participación"
    r"|Juntos podemos seguir mejorando este Portal"
    r"|- (Inicio|Trabajadores|Prestaciones / Pensiones de Trabajadores|Solicitudes"
    r"|Histórico de rentas garantizadas IMV))$"
)


def html_to_markdown(html: str, title: str) -> str:
    """Main content of the Social Security page as Markdown, without navigation or widgets."""
    parser = _MarkdownExtractor()
    parser.feed(html)
    parser._flush()
    headings = {line.lstrip("# ") for line in parser.lines if line.startswith("#")}
    kept: list[str] = []
    for line in parser.lines:
        is_menu_entry = line.startswith("- ") and line[2:] in headings  # the page's own index
        is_page_title = line.startswith("# ")  # we write our own title
        if PAGE_NOISE.match(line) or is_menu_entry or is_page_title:
            continue
        if not kept or line != kept[-1]:
            kept.append(line)
    return f"# {title}\n\n" + "\n\n".join(kept) + "\n"


def fetch_page(client: httpx.Client, page: WebPage) -> dict:
    response = client.get(page.url, follow_redirects=True)
    response.raise_for_status()
    return {
        "file": f"{page.slug}.md",
        "text": html_to_markdown(response.text, page.title),
        "title": page.title,
        "source_url": page.url,
        "in_force": True,
        "license": "seg-social.es: public information, reuse with attribution (Ley 37/2007)",
    }


# ------------------------------------------------------------------------------------- driver


def build_corpus(out_dir: Path = CORPUS_DIR, today: str | None = None) -> list[dict]:
    """Download every source, write the Markdown files and ``MANIFEST.json``."""
    today = today or date.today().strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=30, headers=headers) as client:
        records = [fetch_norm(client, norm, today) for norm in BOE_NORMS]
        records += [fetch_page(client, page) for page in WEB_PAGES]
    for record in records:
        text = record.pop("text")
        (out_dir / record["file"]).write_text(text, encoding="utf-8", newline="\n")
        record.update(
            chars=len(text),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            retrieved=f"{today[:4]}-{today[4:6]}-{today[6:]}",
        )
        manifest.append(record)
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    parser.add_argument("--today", help="YYYYMMDD: pick the BOE versions in force that day")
    args = parser.parse_args()
    for record in build_corpus(args.out, args.today):
        print(f"{record['file']:<26} {record['chars']:>8} chars  {record['title'][:70]}")


if __name__ == "__main__":
    main()
