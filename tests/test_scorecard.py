"""Pipeline scorecard against a simulated API (CPU, no models, no network)."""

import json

import httpx
import pytest

pytest.importorskip("langchain_text_splitters")

from pipeline_eval import scorecard as sc  # noqa: E402


def fake_api(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    path = request.url.path
    if path == "/health":
        return httpx.Response(200, json={"status": "ok", "project": "x", "phases": {
            p: {"status": "ready"} for p in ("reasoning", "tools", "rag", "agent")}})
    if path == "/reasoning":
        ok = True
        return httpx.Response(200, json={"thinking": "", "answer": body["expected_answer"], "raw": "",
                                         "has_valid_format": True, "tokens_generated": 300,
                                         "verifier": {"is_correct": ok, "predicted": "1", "expected": "1"}})
    if path == "/tools":
        return httpx.Response(200, json={"answer": "Hola", "tool_calls": [], "turns": 1})
    if path == "/rag":
        return httpx.Response(200, json={"answer": "No lo sé: no consta.", "chunks": [
            {"id": "ley-19-2021-0001", "source": "ley-19-2021.md", "score": 0.5, "text": "t"}],
            "citations": [], "retriever": body["retriever"]})
    if path == "/agent":
        if body.get("session_id") == "new":
            return httpx.Response(200, json={"final_answer": "733,60 €/mes", "steps": [], "n_steps": 1,
                                             "succeeded": True, "session_id": "abc" * 11})
        answer = "Al año son 8.803,20 €." if body.get("session_id") else "No puedo hacerlo."
        return httpx.Response(200, json={"final_answer": answer, "steps": [], "n_steps": 1,
                                         "succeeded": True, "tokens_used": 10})
    return httpx.Response(404, text="<html>not found</html>")


@pytest.fixture
def client():
    return sc.Client("http://api", 30, transport=httpx.MockTransport(fake_api))


def test_phase1_uses_the_verifier_verdict(client):
    items = sc.phase1_items(3)
    assert len(items) == 3 and items[0]["expected"]
    rows = [sc.run_phase1(client, it) for it in items]
    s = sc.summarize_phase1(rows)
    assert s["pass@1"] == 1.0 and s["http_errors"] == 0 and s["over_300s"] == 0


def test_phase2_scores_with_the_phase2_metrics(client):
    cases = sc.jsonl(sc.PHASE2_BENCH)
    rows = [sc.run_phase2(client, c) for c in cases[:5]]
    s = sc.summarize_phase2(rows)
    assert s["selection"] is not None and s["n"] == 5


def test_phase3_measures_recall_citations_and_abstention(client):
    gold, unanswerable = sc.rag_items()
    rows = [sc.run_phase3(client, gold[0], "hybrid"), sc.run_phase3(client, unanswerable[0], "hybrid")]
    s = sc.summarize_phase3(rows)
    assert s["unanswerable_abstained"] == 1.0 and s["false_abstentions"] == 1.0
    assert s["citations_exist"] == 1.0 and s["recall@5"] in (0.0, 1.0)


def test_phase4_and_multi_turn_memory(client):
    tasks = {t["id"]: t for t in sc.jsonl(sc.AGENT_BENCH)}
    row = sc.run_phase4(client, tasks["I03"], "base", 4)
    assert row["correct"]  # "No puedo hacerlo." is the right answer to an impossible task
    conv = sc.run_conversation(client, sc.CONVERSATIONS[0], "base")
    assert conv["correct"]


def test_errors_and_markdown(client):
    res = client.call("POST", "/nope", {})
    assert res["status"] == 404 and "raw" in res["body"]
    card = {"label": "t", "api": "x", "finished": "now", "summary": {
        "health": {"phases": 4, "ready": 4, "status": {}},
        "phase1": sc.summarize_phase1([{**res, "correct": True, "format": True, "truncated": False,
                                        "tokens": 1}])}}
    assert "## phase1" in sc.markdown(card)


def test_compare_marks_direction_per_metric():
    from pipeline_eval.compare import table, verdict

    assert verdict("pass@1", 0.4, 0.5) == "mejor" and verdict("p50_s", 10, 20) == "peor"
    before = {"label": "baseline", "summary": {"phase1": {"pass@1": 0.4, "n": 40},
                                               "phase4": {"base": {"success": 0.3}}}}
    after = {"label": "fixes", "summary": {"phase1": {"pass@1": 0.5, "n": 40},
                                           "phase4": {"base": {"success": 0.5}}}}
    text = table(before, after)
    assert "| phase1 | pass@1 | 0.400 | 0.500 | 0.100 | mejor |" in text
    assert "| phase4 base | success | 0.300 | 0.500 | 0.200 | mejor |" in text
