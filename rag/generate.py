"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``answer_with_rag`` is what ``POST /rag`` calls. It must:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. build the augmented prompt (``RAG_PROMPT_TEMPLATE`` below is the one from class, with
   chunk ids so the model can cite them);
3. generate the answer with your model;
4. extract the citations and return everything in a ``RagResponse``.

A grader will check that every cited id exists among the returned chunks, and will read
the answers to see whether they actually rest on the retrieved text.
"""

from __future__ import annotations

import re
from pathlib import Path

from api.schemas import RagResponse, RetrievedChunk
from rag.chunking import Chunk, recursive_chunks
from rag.ingest import CORPUS_DIR, load_corpus
from rag.retriever import BM25Retriever, Hit

RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their id in square brackets, like [{example_id}].
If the information is not enough to answer, say so instead of guessing.

Retrieved information:
{chunks}

Question: {question}
"""

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}" for hit in hits
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    return RAG_PROMPT_TEMPLATE.format(
        example_id=example_id, chunks=format_chunks(hits), question=question
    )


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=extract_citations(answer, {c.id for c in chunks}),
        retriever=retriever,
        model=model,
    )


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused."""

    def __init__(
        self, corpus_dir: Path = CORPUS_DIR, chunk_size: int = 800, chunk_overlap: int = 100
    ):
        self.corpus_dir = corpus_dir
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.chunks: list[Chunk] = []
        self.retrievers: dict[str, object] = {}

    def load(self) -> None:
        """Ingest the corpus, chunk it recursively and build the retrievers.

        Chunk ids are ``<doc_id>-<index>`` and depend only on the corpus files, so they are
        stable between runs. The dense and hybrid retrievers are not wired yet.
        """
        documents = load_corpus(self.corpus_dir)
        if not documents:
            raise RuntimeError(
                f"empty corpus in {self.corpus_dir}: run `uv run python -m rag.fetch_corpus`"
            )
        self.chunks = recursive_chunks(documents, self.chunk_size, self.chunk_overlap)
        self.retrievers = {"bm25": BM25Retriever(self.chunks)}

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """Top ``top_k`` chunks for ``question`` with the dense, bm25 or hybrid retriever."""
        if retriever not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retriever {retriever!r}: use dense, bm25 or hybrid")
        if retriever not in self.retrievers:
            raise NotImplementedError(f"the {retriever} retriever is not available yet")
        return self.retrievers[retriever].search(question, top_k=top_k)

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Retrieve, build the prompt, generate, extract citations (generation not wired yet)."""
        raise NotImplementedError("answer generation is not wired yet")


_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        pipeline = RagPipeline()
        pipeline.load()
        _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)
