"""RagPipeline on the committed corpus: BM25 retrieval and stable chunk ids (CPU, no network)."""

import numpy as np
import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("langchain_text_splitters")

from rag.generate import RagPipeline  # noqa: E402
from rag.retriever import Hit  # noqa: E402

# The dense index needs an embedder; a fake one keeps these BM25/id tests offline (the real
# Qwen3-Embedding is a 1.2 GB download) and a temporary index dir keeps rag/index/ untouched.


class FakeEmbedder:
    model_id = "fake-embedder"

    def embed_passages(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32) / 2

    def embed_queries(self, texts, instruction=""):
        return np.ones((len(texts), 4), dtype=np.float32) / 2


def make_pipeline(index_dir) -> RagPipeline:
    pipe = RagPipeline(chunking="recursive:800:100", index_dir=index_dir, embedder=FakeEmbedder())
    pipe.load()
    return pipe


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> RagPipeline:
    return make_pipeline(tmp_path_factory.mktemp("rag-index"))


def test_bm25_returns_hits_with_the_phase4_contract(pipeline):
    hits = pipeline.retrieve("cuantía de la renta garantizada para un adulto solo", 5, "bm25")
    assert len(hits) == 5 and all(isinstance(h, Hit) for h in hits)
    chunk = hits[0].chunk
    assert chunk.id == f"{chunk.doc_id}-{chunk.index:04d}"
    assert chunk.source.endswith(".md") and chunk.text
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_bm25_finds_the_guaranteed_income_in_the_law(pipeline):
    hits = pipeline.retrieve("renta garantizada adulto solo pensiones no contributivas", 5, "bm25")
    assert any(
        h.chunk.doc_id == "ley-19-2021" and "renta garantizada" in h.chunk.text for h in hits
    )


def test_chunk_ids_are_unique_and_stable_across_loads(pipeline, tmp_path):
    ids = [c.id for c in pipeline.chunks]
    assert len(ids) == len(set(ids)) > 300
    other = make_pipeline(tmp_path)
    assert ids == [c.id for c in other.chunks]


def test_unknown_retriever_is_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.retrieve("x", 3, "fuzzy")
