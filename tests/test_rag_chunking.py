"""Sentence splitting and semantic chunking with a fake embedder (CPU, no model)."""

import numpy as np

from rag.chunking import consecutive_distances, semantic_chunks, sentence_spans, split_sentences
from rag.ingest import Document


def topic_embedder(sentences: list[str]) -> np.ndarray:
    """One axis per topic word, so same-topic sentences are identical and others orthogonal."""
    topics = ["renta", "menor", "sancion"]
    vectors = np.array(
        [[float(t in s.lower()) for t in topics] + [1e-6] for s in sentences], dtype=float
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


TEXT = (
    "La renta se calcula cada año. La renta incluye los ingresos del hogar.\n"
    "El menor debe residir en España. El menor cuenta como miembro adicional.\n"
    "La sancion se impone por infraccion grave."
)


def test_sentence_spans_split_on_periods_and_lines():
    sentences = split_sentences(TEXT)
    assert sentences[0] == "La renta se calcula cada año."
    assert len(sentences) == 5
    assert all(
        TEXT[a:b].strip() == s for (a, b), s in zip(sentence_spans(TEXT), sentences, strict=True)
    )


def test_short_fragments_are_glued_to_the_next_sentence():
    lines = [
        "CAPÍTULO I",
        "Disposiciones generales de la norma.",
        "1. La cuantía es mensual.",
        "art. 4 del texto refundido aplicable.",
    ]
    text = "\n".join(lines)
    sentences = split_sentences(text)
    assert sentences[0].startswith("CAPÍTULO I\nDisposiciones")
    assert not any(s in {"1.", "art."} for s in sentences)


def test_distances_are_zero_inside_a_topic_and_one_across():
    sentences = split_sentences(TEXT)
    distances = consecutive_distances(sentences, topic_embedder)
    assert np.allclose(distances, [0, 1, 0, 1], atol=1e-3)


def test_semantic_chunks_cut_at_topic_changes_and_keep_the_original_text():
    doc = Document(id="d", source="d.md", text=TEXT)
    chunks = semantic_chunks([doc], topic_embedder, distance_threshold=0.25)
    assert [c.id for c in chunks] == ["d-0000", "d-0001", "d-0002"]
    assert (
        chunks[0].text == "La renta se calcula cada año. La renta incluye los ingresos del hogar."
    )
    assert "\n" not in chunks[0].text and chunks[2].text.startswith("La sancion")
    assert all(c.text in TEXT for c in chunks)


def test_semantic_chunks_respect_max_chars_and_a_high_threshold_merges_everything():
    doc = Document(id="d", source="d.md", text=TEXT)
    one = semantic_chunks([doc], topic_embedder, distance_threshold=2.0, max_chunk_chars=10_000)
    assert len(one) == 1 and one[0].text == TEXT
    capped = semantic_chunks([doc], topic_embedder, distance_threshold=2.0, max_chunk_chars=80)
    assert len(capped) > 1 and all(len(c.text) <= 80 for c in capped)


def test_semantic_chunks_of_an_empty_document_are_empty():
    assert semantic_chunks([Document(id="d", source="d.md", text="   ")], topic_embedder) == []
