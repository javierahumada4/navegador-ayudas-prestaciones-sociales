"""Prompt, citation checking and the answer flow with a fake generator (CPU, no model)."""

import pytest

from rag.chunking import Chunk
from rag.generate import (
    ABSTENTION,
    RETRY_REMINDER,
    RagPipeline,
    build_prompt,
    extract_citations,
    invalid_citations,
    is_abstention,
    strip_invalid_citations,
    to_response,
)
from rag.retriever import Hit


def make_hits() -> list[Hit]:
    chunks = [
        Chunk("ley-0001", "ley", "ley.md", "La cuantía mínima es de 10 euros mensuales.", 1),
        Chunk("ley-0002", "ley", "ley.md", "El pago se hace por transferencia bancaria.", 2),
    ]
    return [Hit(c, 1.0 - i / 10) for i, c in enumerate(chunks)]


class ScriptedGenerator:
    """Returns canned replies in order and records the prompts it received."""

    model_id = "scripted"

    def __init__(self, *replies: str):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


class FixedRetriever:
    def __init__(self, hits: list[Hit]):
        self.hits = hits

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return self.hits[:top_k]


def make_pipeline(generator, dense_score: float = 0.9, min_dense_score: float = 0.5) -> RagPipeline:
    pipeline = RagPipeline(generator=generator, min_dense_score=min_dense_score)
    hits = make_hits()
    top = [Hit(hits[0].chunk, dense_score)]
    pipeline.retrievers = {
        "bm25": FixedRetriever(hits),
        "hybrid": FixedRetriever(hits),
        "dense": FixedRetriever(top),
    }
    return pipeline


def test_prompt_lists_every_chunk_with_its_id_and_the_question():
    prompt = build_prompt("¿Cuánto es el mínimo?", make_hits())
    assert "[ley-0001] (from ley.md)" in prompt and "[ley-0002]" in prompt
    assert "Pregunta: ¿Cuánto es el mínimo?" in prompt
    assert "ejemplo [ley-0001]" in prompt and ABSTENTION in prompt


def test_citation_check_separates_real_from_invented_ids():
    answer = "Son 10 euros [ley-0001] y se paga por banco [ley-0099] [ley-0002]."
    valid = {"ley-0001", "ley-0002"}
    assert extract_citations(answer, valid) == ["ley-0001", "ley-0002"]
    assert invalid_citations(answer, valid) == ["ley-0099"]
    assert (
        strip_invalid_citations(answer, valid)
        == "Son 10 euros [ley-0001] y se paga por banco [ley-0002]."
    )


def test_is_abstention_recognises_the_canonical_phrase():
    assert is_abstention(ABSTENTION) and is_abstention("  No lo sé, lo siento")
    assert not is_abstention("El importe es de 10 euros [ley-0001].")


def test_to_response_only_reports_citations_that_exist():
    response = to_response("Ver [ley-0001] y [fantasma-1].", make_hits(), "bm25", "m")
    assert response.citations == ["ley-0001"]
    assert [c.id for c in response.chunks] == ["ley-0001", "ley-0002"]


def test_answer_returns_the_generated_text_with_verified_citations():
    generator = ScriptedGenerator("Son 10 euros mensuales [ley-0001].")
    response = make_pipeline(generator).answer("¿Mínimo?", top_k=2, retriever="hybrid")
    assert response.answer.startswith("Son 10 euros") and response.citations == ["ley-0001"]
    assert response.retriever == "hybrid" and response.model == "scripted"


def test_answer_abstains_without_calling_the_model_when_the_corpus_does_not_cover_it():
    generator = ScriptedGenerator()
    response = make_pipeline(generator, dense_score=0.2).answer("¿Cuánto cobra un ministro?")
    assert response.answer == ABSTENTION and response.citations == []
    assert generator.prompts == [] and len(response.chunks) == 2


def test_answer_retries_once_when_the_first_reply_cites_nothing():
    generator = ScriptedGenerator("Son 10 euros.", "Son 10 euros [ley-0001].")
    response = make_pipeline(generator).answer("¿Mínimo?")
    assert response.citations == ["ley-0001"]
    assert generator.prompts[1].endswith(RETRY_REMINDER)


def test_answer_with_only_invented_citations_becomes_an_abstention():
    generator = ScriptedGenerator("Son 10 euros [nada-0001].", "Son 10 euros [nada-0002].")
    response = make_pipeline(generator).answer("¿Mínimo?")
    assert response.answer == ABSTENTION and response.citations == []


def test_the_model_can_abstain_on_its_own():
    generator = ScriptedGenerator(ABSTENTION)
    response = make_pipeline(generator).answer("¿Mínimo?")
    assert response.answer == ABSTENTION and len(generator.prompts) == 1


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError):
        make_pipeline(ScriptedGenerator()).retrieve("x", 3, "fuzzy")
