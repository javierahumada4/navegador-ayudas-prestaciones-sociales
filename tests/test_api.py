"""HTTP contract checks without requiring a GPU or downloading model weights."""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_lists_all_phases():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body["phases"]) == {"reasoning", "tools", "rag", "agent"}
    for phase in body["phases"].values():
        assert phase["status"] in {"ready", "pending", "error"}


def test_tools_endpoint_contract_without_real_model(monkeypatch):
    from api.app import backends
    from api.schemas import ToolsResponse

    class StubToolLoop:
        def run(self, query, max_turns=5):
            return ToolsResponse(answer="Respuesta simulada", tool_calls=[], turns=1, model="stub")

    monkeypatch.setattr(backends, "tools", lambda: StubToolLoop())
    response = client.post("/tools", json={"query": "Consulta de prueba"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Respuesta simulada"
    assert response.json()["tool_calls"] == []


def test_request_validation():
    assert client.post("/reasoning", json={}).status_code == 422
    assert client.post("/rag", json={"question": "x", "retriever": "magic"}).status_code == 422
    assert client.post("/agent", json={"task": "x", "max_steps": 0}).status_code == 422


def test_docs_available():
    assert client.get("/docs").status_code == 200


def test_agent_session_id_is_validated():
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    bad = client.post("/agent", json={"task": "x", "session_id": "../../etc"})
    assert bad.status_code == 422
