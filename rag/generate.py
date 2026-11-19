"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``RagPipeline.answer`` is what ``POST /rag`` calls:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. if the best dense similarity is below ``min_dense_score`` the corpus does not cover the
   question: answer "no lo sé" without calling the model;
3. build the augmented prompt (``RAG_PROMPT_TEMPLATE``: class template, in Spanish, with chunk
   ids the model can cite) and generate the answer;
4. check the citations automatically: ids that are not among the retrieved chunks are removed,
   and an answer that is not an abstention but cites nothing valid is replaced by "no lo sé";
5. return everything in a ``RagResponse``.

Phase 4 uses ``build_pipeline().retrieve(query, top_k, "hybrid")`` and never loads the generator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from api.schemas import RagResponse, RetrievedChunk
from rag.chunking import Chunk, build_chunks
from rag.embed import CachedEmbedFn, Embedder
from rag.ingest import CORPUS_DIR, Document, load_corpus
from rag.retriever import (
    DEFAULT_INDEX_DIR,
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
)

log = logging.getLogger("arca.rag")

DEFAULT_CHUNKING = "recursive:800:100"
DEFAULT_WEIGHT_BM25 = 0.5  # λ of the hybrid retriever, see docs/experiments_fase3.md
DEFAULT_MIN_DENSE_SCORE = 0.0  # below this top-1 cosine the question is out of the corpus
DEFAULT_GENERATOR = "Qwen/Qwen3-1.7B"

ABSTENTION = (
    "No lo sé: los documentos del corpus no contienen información suficiente para responder "
    "a esta pregunta."
)

RAG_PROMPT_TEMPLATE = """Responde a la pregunta usando solo la información recuperada de abajo.
Cita los fragmentos en los que te apoyas escribiendo su identificador entre corchetes, por
ejemplo [{example_id}], justo después de cada afirmación.
Si la información no basta para responder, responde exactamente: {abstention}
No uses conocimiento externo ni inventes cifras ni identificadores.

Información recuperada:
{chunks}

Pregunta: {question}
"""

RETRY_REMINDER = (
    "\nRecuerda: cada afirmación debe llevar el identificador de un fragmento entre corchetes, "
    "o bien responde exactamente la frase de «no lo sé»."
)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}" for hit in hits
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    return RAG_PROMPT_TEMPLATE.format(
        example_id=example_id,
        abstention=ABSTENTION,
        chunks=format_chunks(hits),
        question=question,
    )


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def invalid_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids the answer cites that are NOT among the retrieved chunks (hallucinated references)."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match not in valid_ids and match not in seen:
            seen.append(match)
    return seen


def strip_invalid_citations(answer: str, valid_ids: set[str]) -> str:
    """Remove ``[id]`` markers whose id does not exist, so the text never points at nothing."""
    return re.sub(
        r" ?\[([A-Za-z0-9_-]+)\]",
        lambda match: match.group(0) if match.group(1) in valid_ids else "",
        answer,
    )


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().startswith("no lo sé")


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


class Generator:
    """Chat model that writes the answer (bf16 on GPU when there is one, greedy decoding)."""

    def __init__(self, model_id: str = DEFAULT_GENERATOR):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        import torch

        if self.model is None:
            self.load()
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return THINK_PATTERN.sub("", reply).strip()


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused.

    Configuration comes from the constructor or from the environment (``ARCA_RAG_CHUNKING``,
    ``ARCA_RAG_LAMBDA``, ``ARCA_RAG_MIN_SCORE``, ``ARCA_RAG_MODEL``). The generator is loaded the
    first time ``answer`` needs it, so retrieval-only users (phase 4) never pay for it.
    """

    def __init__(
        self,
        corpus_dir: Path = CORPUS_DIR,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunking: str | None = None,
        weight_bm25: float | None = None,
        min_dense_score: float | None = None,
        generator: Generator | None = None,
        embedder: Embedder | None = None,
    ):
        self.corpus_dir, self.index_dir = corpus_dir, Path(index_dir)
        self.chunking = chunking or os.environ.get("ARCA_RAG_CHUNKING", DEFAULT_CHUNKING)
        self.weight_bm25 = (
            weight_bm25
            if weight_bm25 is not None
            else float(os.environ.get("ARCA_RAG_LAMBDA", DEFAULT_WEIGHT_BM25))
        )
        self.min_dense_score = (
            min_dense_score
            if min_dense_score is not None
            else float(os.environ.get("ARCA_RAG_MIN_SCORE", DEFAULT_MIN_DENSE_SCORE))
        )
        self.generator = generator or Generator(os.environ.get("ARCA_RAG_MODEL", DEFAULT_GENERATOR))
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.retrievers: dict[str, object] = {}

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            import torch

            self.embedder = Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
        return self.embedder

    def _load_chunks(self, documents: list[Document]) -> list[Chunk]:
        """Recursive chunks are instant; semantic ones embed every sentence, so they are cached."""
        if self.chunking.startswith("recursive"):
            return build_chunks(self.chunking, documents)
        corpus = json.dumps([self.chunking, [(d.id, d.text) for d in documents]])
        fingerprint = hashlib.sha1(corpus.encode()).hexdigest()[:12]
        name = self.chunking.replace(":", "_")
        path = self.index_dir / f"chunks-{name}-{fingerprint}.json"
        if path.exists():
            return [Chunk(**c) for c in json.loads(path.read_text(encoding="utf-8"))]
        chunks = build_chunks(self.chunking, documents, CachedEmbedFn(self._get_embedder()))
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [chunk.__dict__ for chunk in chunks]
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return chunks

    def load(self) -> None:
        """Ingest the corpus, chunk it and build the three retrievers (dense index persisted).

        Chunk ids are ``<doc_id>-<index>`` and depend only on the corpus files and the chunking,
        so they are stable between runs.
        """
        documents = load_corpus(self.corpus_dir)
        if not documents:
            raise RuntimeError(
                f"empty corpus in {self.corpus_dir}: run `uv run python -m rag.fetch_corpus`"
            )
        self.chunks = self._load_chunks(documents)
        bm25 = BM25Retriever(self.chunks)
        dense = DenseRetriever(self.chunks, self._get_embedder(), self.index_dir)
        hybrid = HybridRetriever(bm25, dense, weight_bm25=self.weight_bm25)
        self.retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """Top ``top_k`` chunks for ``question`` with the dense, bm25 or hybrid retriever."""
        if retriever not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retriever {retriever!r}: use dense, bm25 or hybrid")
        return self.retrievers[retriever].search(question, top_k=top_k)

    def covers(self, question: str) -> bool:
        """Whether the corpus plausibly answers ``question``: top-1 dense cosine above the gate."""
        best = self.retrievers["dense"].search(question, top_k=1)
        return bool(best) and best[0].score >= self.min_dense_score

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Retrieve, build the prompt, generate and verify the citations."""
        hits = self.retrieve(question, top_k, retriever)
        model = self.generator.model_id
        if not self.covers(question):
            return to_response(ABSTENTION, hits, retriever, model)
        valid_ids = {h.chunk.id for h in hits}
        prompt = build_prompt(question, hits)
        answer = self.generator.generate(prompt)
        if not is_abstention(answer) and not extract_citations(answer, valid_ids):
            answer = self.generator.generate(prompt + RETRY_REMINDER)  # one retry, then give up
        bad = invalid_citations(answer, valid_ids)
        if bad:
            log.warning("answer cited ids that were not retrieved: %s", bad)
            answer = strip_invalid_citations(answer, valid_ids)
        if not is_abstention(answer) and not extract_citations(answer, valid_ids):
            answer = ABSTENTION  # an answer that rests on no retrieved chunk is not an answer
        return to_response(answer, hits, retriever, model)


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
