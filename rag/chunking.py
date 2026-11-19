"""Data indexing, step 2: chunking.

Two strategies from class. ``recursive_chunks`` is the default of most RAG systems and is
implemented for you with LangChain's splitter. ``semantic_chunks`` is yours to write: the
algorithm is in the docstring and in the slides. Compare both with ``rag/evaluate.py``:
number of chunks, average length, and Recall@k on your gold set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag.ingest import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(id=f"{doc.id}-{i:04d}", doc_id=doc.id, source=doc.source, text=piece, index=i)
            )
    return chunks


MIN_SENTENCE_CHARS = 25


def sentence_spans(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of the sentences of ``text``.

    Splits after ``.``/``!``/``?`` and at every line break (our corpus has one paragraph per
    line). Legal text is full of fragments that are not sentences ("1.", "a)", "art.", a chapter
    title): any piece shorter than ``min_chars`` is glued to the one that follows it.
    """
    pieces, start = [], 0
    for match in re.finditer(r"(?<=[.!?])[ \t]+(?=\S)|\n+", text):
        pieces.append((start, match.start()))
        start = match.end()
    pieces.append((start, len(text)))
    spans: list[tuple[int, int]] = []
    carry: int | None = None
    for begin, end in pieces:
        if not text[begin:end].strip():
            continue
        begin = carry if carry is not None else begin
        carry = None
        if len(text[begin:end].strip()) < min_chars:
            carry = begin
            continue
        spans.append((begin, end))
    if carry is not None:  # a short tail: attach it to the last sentence
        if spans:
            spans[-1] = (spans[-1][0], len(text.rstrip()))
        else:
            spans.append((carry, len(text.rstrip())))
    return spans


def split_sentences(text: str) -> list[str]:
    """Sentences of ``text`` (see ``sentence_spans``), stripped."""
    return [text[a:b].strip() for a, b in sentence_spans(text)]


def consecutive_distances(sentences: list[str], embed_fn) -> np.ndarray:
    """Cosine distance ``1 - e_i · e_{i+1}`` between each sentence and the next one."""
    if len(sentences) < 2:
        return np.zeros(0)
    embeddings = np.asarray(embed_fn(sentences))
    return 1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)


def semantic_chunks(
    documents: list[Document],
    embed_fn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
) -> list[Chunk]:
    """Group consecutive sentences while their embeddings stay close.

    ``embed_fn(list[str]) -> array of shape (n, d)`` with L2-normalised rows (see
    ``rag/embed.py``). The algorithm from the slides:

    1. Split the document into sentences s_1 … s_n.
    2. Embed them: e_1 … e_n.
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars``.

    A chunk keeps the original text between its first and last sentence (line breaks included).
    A single sentence longer than ``max_chunk_chars`` stays whole. The distance profile of a
    document, with the threshold line, is drawn by ``rag/evaluate.py plot-distances``.
    """
    chunks = []
    for doc in documents:
        spans = sentence_spans(doc.text)
        if not spans:
            continue
        distances = consecutive_distances([doc.text[a:b] for a, b in spans], embed_fn)
        first = 0
        groups = []
        for i, distance in enumerate(distances):  # distance between sentence i and i + 1
            too_long = spans[i + 1][1] - spans[first][0] > max_chunk_chars
            if distance > distance_threshold or too_long:
                groups.append((first, i))
                first = i + 1
        groups.append((first, len(spans) - 1))
        for index, (begin, end) in enumerate(groups):
            text = doc.text[spans[begin][0] : spans[end][1]]
            chunks.append(
                Chunk(
                    id=f"{doc.id}-{index:04d}",
                    doc_id=doc.id,
                    source=doc.source,
                    text=text,
                    index=index,
                )
            )
    return chunks


def build_chunks(spec: str, documents: list[Document], embed_fn=None) -> list[Chunk]:
    """Chunk from a spec: ``recursive:<size>:<overlap>`` or ``semantic:<threshold>:<max_chars>``."""
    kind, first, second = spec.split(":")
    if kind == "recursive":
        return recursive_chunks(documents, chunk_size=int(first), chunk_overlap=int(second))
    if kind == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embedding function")
        return semantic_chunks(
            documents, embed_fn, distance_threshold=float(first), max_chunk_chars=int(second)
        )
    raise ValueError(
        f"unknown chunking {spec!r}: use recursive:<size>:<overlap> or semantic:<th>:<max>"
    )


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table."""
    lengths = [len(c.text) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
    }
