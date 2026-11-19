"""The gold set: questions annotated with the *evidence* that answers them.

A question stores a verbatim sentence from the corpus (``evidence``), not a chunk id. Chunk ids
change with the chunking strategy, the evidence does not, so the same fifty questions score the
recursive and the semantic chunker. ``resolve_gold`` maps each evidence string to the chunks that
contain it; a chunk counts as relevant when it holds the evidence, whichever document it is in
(unless the document is repealed: a repealed text is never a valid answer).

``rag/gold/gold.jsonl`` fields::

    id, question, evidence: [str, ...], mode: "any" | "all", category, style, notes?

* ``mode = "any"`` (default): every evidence string is a valid answer on its own (the law and
  the Social Security page often say the same thing); one hit is enough.
* ``mode = "all"``: the answer needs every evidence string (multi-hop questions); recall is the
  fraction of evidence strings found in the top k.

``rag/gold/unanswerable.jsonl`` holds questions the corpus cannot answer (for "no lo sé").

Check the annotations (every evidence must map to at least one chunk)::

    uv run python -m rag.gold audit
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import Chunk
from rag.ingest import CORPUS_DIR

GOLD_DIR = Path(__file__).parent / "gold"


def normalize(text: str) -> str:
    """Collapse whitespace so evidence matches across line breaks."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def in_force_documents(corpus_dir: Path = CORPUS_DIR) -> set[str]:
    """Source file names of the documents that are in force (from ``MANIFEST.json``)."""
    manifest = json.loads((corpus_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    return {entry["file"] for entry in manifest if entry["in_force"]}


@dataclass
class GoldItem:
    id: str
    question: str
    evidence: list[str]
    mode: str = "any"
    category: str = ""
    style: str = "formal"
    notes: str = ""
    # one set of chunk ids per evidence string, filled by ``resolve_gold``
    evidence_chunk_ids: list[set[str]] = field(default_factory=list)

    @property
    def relevant_chunk_ids(self) -> set[str]:
        return set().union(*self.evidence_chunk_ids) if self.evidence_chunk_ids else set()


def load_gold(path: str | Path = GOLD_DIR / "gold.jsonl") -> list[GoldItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            GoldItem(
                id=raw["id"],
                question=raw["question"],
                evidence=raw["evidence"],
                mode=raw.get("mode", "any"),
                category=raw.get("category", ""),
                style=raw.get("style", "formal"),
                notes=raw.get("notes", ""),
            )
        )
    return items


def load_unanswerable(path: str | Path = GOLD_DIR / "unanswerable.jsonl") -> list[str]:
    return [
        json.loads(line)["question"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunks_containing(evidence: str, chunks: list[Chunk], valid_sources: set[str]) -> set[str]:
    """Ids of the chunks (from documents in ``valid_sources``) whose text holds ``evidence``."""
    needle = normalize(evidence)
    return {c.id for c in chunks if c.source in valid_sources and needle in normalize(c.text)}


def resolve_gold(
    items: list[GoldItem], chunks: list[Chunk], valid_sources: set[str] | None = None
) -> list[GoldItem]:
    """Fill ``evidence_chunk_ids`` for a chunking. Raises if an evidence maps to no chunk."""
    valid_sources = valid_sources if valid_sources is not None else in_force_documents()
    for item in items:
        item.evidence_chunk_ids = [
            chunks_containing(e, chunks, valid_sources) for e in item.evidence
        ]
        missing = [
            e for e, ids in zip(item.evidence, item.evidence_chunk_ids, strict=True) if not ids
        ]
        if missing:
            raise ValueError(f"{item.id}: evidence not found in any chunk: {missing[0][:80]!r}")
    return items


def item_recall(ranked_ids: list[str], item: GoldItem, k: int) -> float:
    """Recall@k of one question: 0/1 in ``any`` mode, fraction of evidence found in ``all``."""
    top = set(ranked_ids[:k])
    found = [bool(ids & top) for ids in item.evidence_chunk_ids]
    return float(any(found)) if item.mode == "any" else sum(found) / max(len(found), 1)


def item_reciprocal_rank(ranked_ids: list[str], item: GoldItem) -> float:
    """1 / rank of the first chunk that holds any of the evidence."""
    relevant = item.relevant_chunk_ids
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def audit(chunks: list[Chunk]) -> None:
    """Print, for every question, how many chunks hold each evidence and from which documents."""
    valid = in_force_documents()
    items = load_gold()
    by_id = {c.id: c for c in chunks}
    problems = 0
    for item in items:
        for evidence in item.evidence:
            ids = chunks_containing(evidence, chunks, valid)
            docs = sorted({by_id[i].doc_id for i in ids})
            repealed = sorted(
                {
                    c.doc_id
                    for c in chunks
                    if c.source not in valid and normalize(evidence) in normalize(c.text)
                }
            )
            flag = "" if ids else "  <-- NOT FOUND"
            problems += not ids
            extra = f"  (also in repealed: {', '.join(repealed)})" if repealed else ""
            print(f"{item.id} [{item.mode}] {len(ids)} chunks in {docs}{extra}{flag}")
    print(f"{len(items)} questions, {problems} evidence strings without a chunk")


def main() -> None:
    import argparse

    from rag.chunking import recursive_chunks
    from rag.ingest import load_corpus

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["audit"])
    parser.parse_args()
    audit(recursive_chunks(load_corpus()))


if __name__ == "__main__":
    main()
