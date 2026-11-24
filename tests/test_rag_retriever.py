"""Dense store, score fusion and Spanish tokenisation, with a fake embedder (CPU, no model)."""

import numpy as np
import pytest

from rag.chunking import Chunk
from rag.retriever import (
    DenseRetriever,
    Hit,
    HybridRetriever,
    fuse,
    min_max,
    tokenize_es,
)

VOCAB = ["cuantia", "renta", "menor", "residencia", "sancion"]


class FakeEmbedder:
    """Bag-of-keywords vectors: deterministic, normalised, and it counts how often it is called."""

    model_id = "fake-embedder"

    def __init__(self):
        self.passage_calls = 0

    def _encode(self, text: str) -> np.ndarray:
        vec = np.array([float(word in text.lower()) for word in VOCAB]) + 1e-3
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def embed_passages(self, texts):
        self.passage_calls += 1
        return np.stack([self._encode(t) for t in texts])

    def embed_queries(self, texts, instruction=""):
        return np.stack([self._encode(t) for t in texts])


def make_chunks() -> list[Chunk]:
    texts = [
        "la cuantia de la renta",
        "el menor y su residencia",
        "la sancion aplicable",
        "otra cosa",
    ]
    return [
        Chunk(id=f"d-{i:04d}", doc_id="d", source="d.md", text=t, index=i)
        for i, t in enumerate(texts)
    ]


def hit(chunk_id: str, score: float) -> Hit:
    return Hit(Chunk(id=chunk_id, doc_id="d", source="d.md", text=chunk_id, index=0), score)


def test_dense_ranks_by_cosine_and_returns_scores(tmp_path):
    dense = DenseRetriever(make_chunks(), FakeEmbedder(), persist_dir=tmp_path)
    hits = dense.search("cuantia renta", top_k=2)
    assert hits[0].chunk.id == "d-0000"
    assert hits[0].score > hits[1].score
    assert -1.0 <= hits[1].score <= 1.0


def test_dense_index_is_persisted_and_reused(tmp_path):
    first, second = FakeEmbedder(), FakeEmbedder()
    DenseRetriever(make_chunks(), first, persist_dir=tmp_path)
    reloaded = DenseRetriever(make_chunks(), second, persist_dir=tmp_path)
    assert (first.passage_calls, second.passage_calls) == (1, 0)
    assert reloaded.search("sancion", top_k=1)[0].chunk.id == "d-0002"


def test_dense_index_is_rebuilt_when_a_chunk_changes(tmp_path):
    chunks = make_chunks()
    DenseRetriever(chunks, FakeEmbedder(), persist_dir=tmp_path)
    chunks[3] = Chunk(id="d-0003", doc_id="d", source="d.md", text="la sancion", index=3)
    embedder = FakeEmbedder()
    DenseRetriever(chunks, embedder, persist_dir=tmp_path)
    assert embedder.passage_calls == 1
    assert len(list(tmp_path.glob("dense-*.npy"))) == 2


def test_min_max_rescales_and_handles_flat_lists():
    assert min_max([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert min_max([0.0, 0.0]) == [0.0, 0.0]
    assert min_max([3.0, 3.0]) == [1.0, 1.0]
    assert min_max([]) == []


def test_fuse_weights_the_two_rescaled_lists():
    bm25 = [hit("a", 10.0), hit("b", 0.0)]  # a is the keyword favourite
    dense = [hit("b", 0.9), hit("a", 0.1)]  # b is the semantic favourite
    assert [h.chunk.id for h in fuse(bm25, dense, 1.0, 2)] == ["a", "b"]
    assert [h.chunk.id for h in fuse(bm25, dense, 0.0, 2)] == ["b", "a"]
    tie = fuse(bm25, dense, 0.5, 2)
    assert [h.score for h in tie] == [0.5, 0.5]
    assert [h.chunk.id for h in tie] == ["a", "b"]  # deterministic tie-break by id


def test_fuse_gives_zero_to_chunks_one_retriever_missed():
    fused = fuse([hit("a", 5.0), hit("b", 1.0)], [hit("c", 0.8), hit("a", 0.2)], 0.5, 3)
    scores = {h.chunk.id: h.score for h in fused}
    assert scores["a"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)
    assert scores["c"] == pytest.approx(0.5)
    assert scores["b"] == 0.0


def test_hybrid_combines_bm25_and_dense(tmp_path):
    pytest.importorskip("rank_bm25")
    from rag.retriever import BM25Retriever

    chunks = make_chunks()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, FakeEmbedder(), persist_dir=tmp_path)
    hybrid = HybridRetriever(bm25, dense, weight_bm25=0.5, pool=4)
    hits = hybrid.search("cuantia renta", top_k=3)
    assert hits[0].chunk.id == "d-0000" and len(hits) == 3
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_tokenize_es_folds_accents_and_drops_stopwords():
    assert tokenize_es("¿Cuál es la CUANTÍA de la prestación?") == ["cuantia", "prestacion"]
