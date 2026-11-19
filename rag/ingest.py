"""Data indexing, step 1: collect the documents of your knowledge base.

Drop your files in ``rag/corpus/`` (PDF, Markdown, plain text, HTML) and this module
turns them into ``Document`` objects with a stable id and a ``source`` you can cite.
Keep the corpus out of git if it is big or not yours to redistribute; leave a script or
a link in ``rag/corpus/README.md`` so the index can be rebuilt.

Run ``uv run python -m rag.ingest`` to see what gets loaded and how long each document is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
NOT_CORPUS = {"readme.md"}  # documentation about the corpus, not part of it


@dataclass
class Document:
    id: str
    source: str
    text: str
    metadata: dict | None = None


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def document_id(relative_path: Path) -> str:
    """Stable, readable id from the path inside the corpus: boe/ley-19.md -> boe-ley-19.

    It depends only on where the file lives, never on its content or on the load order, so the
    chunk ids derived from it (<doc_id>-<index>) survive re-ingestion and can be cited.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", relative_path.with_suffix("").as_posix()).strip("-")
    return slug.lower()


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    if path.suffix.lower() in {".html", ".htm"}:
        return _read_html(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Every supported file under ``corpus_dir``, recursively, as a ``Document``."""
    documents = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.name.lower() in NOT_CORPUS:
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | {
            ".pdf",
            ".html",
            ".htm",
        }:
            continue
        text = read_document(path).strip()
        if not text:
            continue
        relative = path.relative_to(corpus_dir)
        documents.append(
            Document(
                id=document_id(relative),
                source=relative.as_posix(),
                text=text,
                metadata={"chars": len(text), "suffix": path.suffix.lower()},
            )
        )
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    if not docs:
        print(f"No documents found in {CORPUS_DIR}. Add your corpus there (see rag/README.md).")
    for doc in docs:
        print(f"{doc.id}  {doc.metadata['chars']:>8} chars  {doc.source}")
    print(f"{len(docs)} documents, {sum(d.metadata['chars'] for d in docs):,} characters in total")
