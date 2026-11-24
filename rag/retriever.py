"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

All three retrievers return the same ``Hit`` objects so that ``rag/generate.py`` and
``rag/evaluate.py`` do not care which one is used.

* ``BM25Retriever``: keyword ranking, the baseline.
* ``DenseRetriever``: Qwen3-Embedding vectors in an exact NumPy store persisted on disk.
* ``HybridRetriever``: ``s = λ · s_bm25 + (1 - λ) · s_dense`` over min-max rescaled scores.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.embed import DEFAULT_QUERY_INSTRUCTION

DEFAULT_INDEX_DIR = Path(__file__).parent / "index"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


SPANISH_STOPWORDS = frozenset(
    "a al algo ante como con contra cual cuando de del desde donde e el ella ellos en entre era "
    "es esa ese eso esta este esto fue ha han hay la las le les lo los me mi mas muy ni no nos o "
    "para pero por que se si sin sobre son su sus te tiene un una uno unos y ya".split()
)


def tokenize_es(text: str) -> list[str]:
    """Spanish-aware tokens: accents folded and stop words removed (an ablation of ``tokenize``)."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [t for t in re.findall(r"\w+", folded) if t not in SPANISH_STOPWORDS]


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk], tokenizer=tokenize):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.tokenizer = tokenizer
        self.index = BM25Okapi([tokenizer(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.index.get_scores(self.tokenizer(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def _corpus_key(model_id: str, chunks: list[Chunk]) -> str:
    """Fingerprint of what was embedded: a new model or any changed chunk gives a new index."""
    digest = hashlib.sha1(model_id.encode())
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()[:16]


class DenseRetriever:
    """Embed the chunks once, store the matrix, rank by cosine similarity with the query.

    The store is an exact NumPy matrix of L2-normalised vectors (a dot product is the cosine),
    persisted under ``persist_dir`` (``rag/index/``, outside git) and keyed by a fingerprint of
    the model and the chunk texts: the API does not re-embed the corpus at every start, and a
    changed corpus or chunking never reuses a stale index. With a few thousand chunks an exact
    search is instantaneous, so an approximate index (HNSW...) would add moving parts for no gain.
    The query is embedded with ``query_instruction`` (instruction-aware embeddings).
    """

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        persist_dir: str | Path = DEFAULT_INDEX_DIR,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.query_instruction = query_instruction
        self._query_cache: dict[tuple[str, str], np.ndarray] = {}
        path = Path(persist_dir) / f"dense-{_corpus_key(embedder.model_id, chunks)}.npy"
        if path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = np.asarray(
                embedder.embed_passages([c.text for c in chunks]), dtype=np.float32
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.matrix)
        self.path = path

    def embed_query(self, query: str, instruction: str | None = None) -> np.ndarray:
        instruction = self.query_instruction if instruction is None else instruction
        key = (instruction, query)
        if key not in self._query_cache:
            self._query_cache[key] = self.embedder.embed_queries([query], instruction)[0]
        return self._query_cache[key]

    def search(self, query: str, top_k: int = 5, instruction: str | None = None) -> list[Hit]:
        scores = self.matrix @ self.embed_query(query, instruction)
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def min_max(scores: list[float]) -> list[float]:
    """Rescale to [0, 1]. A flat list carries no ranking information: all 0, or all 1 if > 0."""
    low, high = min(scores, default=0.0), max(scores, default=0.0)
    if high - low <= 1e-12:
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


def fuse(bm25_hits: list[Hit], dense_hits: list[Hit], weight_bm25: float, top_k: int) -> list[Hit]:
    """``s = λ · s_bm25 + (1 - λ) · s_dense`` on min-max rescaled scores, merged by chunk id.

    Each list is rescaled on its own pool; a chunk found by only one retriever gets 0 from the
    other. Ties are broken by chunk id so the ranking is deterministic.
    """
    by_id: dict[str, Chunk] = {}
    fused: dict[str, float] = {}
    for hits, weight in ((bm25_hits, weight_bm25), (dense_hits, 1.0 - weight_bm25)):
        for hit, score in zip(hits, min_max([h.score for h in hits]), strict=True):
            by_id[hit.chunk.id] = hit.chunk
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + weight * score
    ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
    return [Hit(by_id[cid], fused[cid]) for cid in ranked]


class HybridRetriever:
    """``s_hybrid = λ · s_bm25 + (1 - λ) · s_dense`` on scores rescaled to [0, 1].

    Takes the top ``pool`` hits from each retriever, rescales each list with min-max so they are
    comparable, merges by chunk id and returns the top ``top_k`` by the weighted sum. λ is chosen
    on the gold set with ``rag/evaluate.py`` (see ``docs/experiments_fase3.md``).
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        pool = max(self.pool, top_k)
        return fuse(
            self.bm25.search(query, pool), self.dense.search(query, pool), self.weight_bm25, top_k
        )
