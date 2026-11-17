"""The HTTP request, not the language model, authorizes saving a report."""

from fastapi.testclient import TestClient

from api.app import app, backends
from api.schemas import ToolsResponse


def test_tools_api_defaults_to_no_file_writes_and_accepts_explicit_confirmation(monkeypatch):
    calls = []

    class FakeLoop:
        def run(self, query, max_turns, *, confirmar_guardado=False):
            calls.append((query, max_turns, confirmar_guardado))
            return ToolsResponse(answer="Respuesta de prueba", tool_calls=[], turns=1, model="fake")

    monkeypatch.setattr(backends, "tools", lambda: FakeLoop())
    client = TestClient(app)

    standard = client.post("/tools", json={"query": "Calcular IMV"})
    assert standard.status_code == 200
    assert calls[-1] == ("Calcular IMV", 5, False)

    authorized = client.post("/tools", json={
        "query": "Calcular IMV y guardar informe",
        "max_turns": 3,
        "confirmar_guardado": True,
    })
    assert authorized.status_code == 200
    assert calls[-1] == ("Calcular IMV y guardar informe", 3, True)

    for non_boolean in ("yes", 1, "true", None):
        invalid = client.post("/tools", json={
            "query": "Calcular IMV", "confirmar_guardado": non_boolean,
        })
        assert invalid.status_code == 422
    assert len(calls) == 2
