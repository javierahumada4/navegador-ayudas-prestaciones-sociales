"""Fixes found by the pipeline scorecard (CPU, scripted models)."""

import json

import numpy as np
import pytest

from agent.react_agent import ReActAgent, cited_ids, final_answer, ungrounded_numbers
from pipeline_eval.tune_rag import best_gate, best_lambda
from tool_use.registry import ToolRegistry, tool


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Fake knowledge base.

    Args:
        query: Question.
        top_k: Passages.
    """
    return "[ss-imv-0007] (ss-imv.md)\nLos titulares deberán tener una edad mínima de 23 años."


def call(name, **arguments):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def agent(*outputs):
    seen = []

    def generator(messages, spec, tools):
        if messages[0]["role"] != "system":
            return "- resumen", 1, 1
        seen.append(messages)
        return outputs[min(len(seen), len(outputs)) - 1], 10, 5

    registry = ToolRegistry().register(search_knowledge_base, final_answer)
    return ReActAgent(registry=registry, generator=generator)


def test_invented_figures_get_one_verification_step():
    a = agent(call("search_knowledge_base", query="edad"),
              call("final_answer", answer="23 años; hay 15 días para comunicar cambios."),
              call("search_knowledge_base", query="plazo para comunicar cambios"),
              call("final_answer", answer="23 años; el plazo no aparece en los fragmentos."))
    r = a.run("edad mínima y días para comunicar un cambio", max_steps=6, brain="base")
    assert r.steps[1].observation.startswith("Comprobación") and "15" in r.steps[1].observation
    assert r.succeeded and "no aparece" in r.final_answer


def test_grounded_figures_pass_and_the_kb_is_cited_automatically():
    a = agent(call("search_knowledge_base", query="edad"), call("final_answer", answer="Son 23 años."))
    r = a.run("edad mínima", max_steps=4, brain="base")
    assert r.n_steps == 2 and r.final_answer.endswith("[ss-imv-0007]")


def test_numbers_helpers():
    assert ungrounded_numbers("Son 733,60 € y 15 días [ley-19-2021-0137]", ["total 733.6"]) == ["15"]
    assert ungrounded_numbers("Paso 1 y 2", []) == []  # small integers are not checked
    assert cited_ids(["[a-0001] (x)\nt", "[b-0002] (y)\nt", "[a-0001] (x)\nt"]) == ["a-0001", "b-0002"]


def test_time_budget_stops_before_the_ngrok_limit(monkeypatch):
    monkeypatch.setenv("ARCA_AGENT_TIME_BUDGET", "0")
    r = agent(call("search_knowledge_base", query="x")).run("t", max_steps=5, brain="base")
    assert not r.succeeded and r.final_answer.startswith("No he podido terminar a tiempo")


def test_plain_text_answers_lose_the_thought_label():
    r = agent("Pensamiento: No puedo presentar solicitudes.").run("t", max_steps=2, brain="base")
    assert r.final_answer == "No puedo presentar solicitudes."


def test_repealed_documents_are_not_served(tmp_path, monkeypatch):
    pytest.importorskip("rank_bm25")
    from rag.generate import RagPipeline

    class Fake:
        model_id = "fake"

        def embed_passages(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

        def embed_queries(self, texts, instruction=""):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    pipe = RagPipeline(chunking="recursive:800:100", index_dir=tmp_path, embedder=Fake())
    pipe.load()
    question = "real decreto-ley ingreso mínimo vital prestación no contributiva"
    assert "rdl-20-2020.md" in pipe.repealed_sources
    assert all(h.chunk.source != "rdl-20-2020.md" for h in pipe.retrieve(question, 10, "bm25"))
    monkeypatch.setenv("ARCA_RAG_INCLUDE_REPEALED", "1")
    assert any(h.chunk.source == "rdl-20-2020.md" for h in pipe.retrieve(question, 10, "bm25"))


def test_rag_tuning_picks_from_the_reports(tmp_path):
    for name, mrr in (("recursive_800_100", 0.46), ("recursive_1200_150", 0.49)):
        report = {str(w): {"mrr": mrr - abs(w - 0.3), "recall@5": 0.7} for w in (0.0, 0.3, 0.5)}
        report["best"] = {"lambda": 0.3, "cv_mrr": mrr - 0.02}
        (tmp_path / f"phase3_lambda_{name}.json").write_text(json.dumps(report))
    (tmp_path / "phase3_calibrate.json").write_text(json.dumps(
        {"answerable": [0.6] * 48 + [0.45, 0.5], "unanswerable": [0.4, 0.42, 0.55, 0.62]}))
    assert best_lambda(tmp_path) == {"chunking": "recursive:1200:150", "lambda": 0.3, "mrr": 0.49,
                                     "recall@5": 0.7, "cv_mrr": pytest.approx(0.47)}
    gate = best_gate(tmp_path, keep=0.96)
    # 0.56 keeps 48/50 answerable (>= 0.96) and rejects 3 of the 4 negatives
    assert gate["min_score"] == 0.56 and gate["rejected"] == 0.75


def test_tool_loop_resamples_once_before_giving_up():
    from tool_use.executor import ToolLoop

    broken = '<tool_call>{"name": "calcular_imv", "arguments": {"case": oops}}</tool_call>'
    loop = ToolLoop(registry=None)
    sampled = []

    def generate(messages):
        sampled.append(getattr(loop, "_sample_next", False))
        return "Lo siento, no tengo datos suficientes." if sampled[-1] else broken

    loop.generate = generate
    result = loop.run("Calcula el IMV", max_turns=4)
    assert sampled.count(True) == 1 and "no tengo datos" in result.answer
