"""Phase 4: the ReAct loop with a scripted model (CPU, no weights, no network)."""

import json
import time
from types import SimpleNamespace

import pytest

from agent.memory import ConversationMemory
from agent.models import brain_spec
from agent.react_agent import ReActAgent, final_answer, render_hits
from tool_use.domain_tools import calcular_imv, guardar_informe
from tool_use.registry import ToolRegistry, tool

CASE = {
    "application_date": "2026-06-01",
    "applicant_id": "adult",
    "persons": [{
        "id": "adult", "date_of_birth": "1980-01-01", "legal_residence_in_spain": True,
        "effective_residence_in_spain": True, "continuous_legal_effective_residence_since": "2020-01-01",
    }],
    "domicile": {"co_resident_ids": ["adult"], "same_domicile_since": "2020-01-01"},
    "relationships": [],
    "applicant_history": {"independent_from_parents_since": "2010-01-01"},
    "economic": {"countable_income_annual_eur": 0, "net_worth_eur": 0, "non_corporate_assets_eur": 0},
}


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Fake knowledge base.

    Args:
        query: Question.
        top_k: Passages.
    """
    return "[ley-19-2021-0137] (ley-19-2021.md)\nLa renta garantizada para un adulto solo..."


@tool
def slow_tool(seconds: float) -> str:
    """Sleeps.

    Args:
        seconds: How long.
    """
    time.sleep(seconds)
    return "done"


def call(name: str, **arguments) -> str:
    return f"Pensamiento: uso {name}.\n<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def scripted(*outputs):
    """A generator that replays ``outputs`` and records what it was shown."""
    seen = []

    def generator(messages, spec, tools):
        if messages[0]["role"] != "system":  # a memory summary call, not a loop step
            return "- resumen de los pasos anteriores", 10, 5
        seen.append(messages)
        if len(seen) > len(outputs):
            return outputs[-1], 10, 5
        return outputs[len(seen) - 1], 10, 5

    generator.seen = seen
    return generator


def make_agent(*outputs, **kwargs) -> ReActAgent:
    registry = ToolRegistry().register(
        calcular_imv, guardar_informe, search_knowledge_base, slow_tool, final_answer
    )
    return ReActAgent(registry=registry, generator=scripted(*outputs), **kwargs)


def test_chains_knowledge_base_and_engine_then_answers():
    agent = make_agent(
        call("search_knowledge_base", query="renta garantizada adulto"),
        call("calcular_imv", case=CASE),
        call("final_answer", answer="Le corresponden 733,60 €/mes [ley-19-2021-0137]."),
    )
    response = agent.run("¿Cuánto IMV cobra este caso?", max_steps=5, brain="base")
    assert response.succeeded and "733,60" in response.final_answer
    assert [s.action.name for s in response.steps] == [
        "search_knowledge_base", "calcular_imv", "final_answer"
    ]
    assert '"total_monthly_eur": 733.6' in response.steps[1].observation
    assert response.n_steps == 3 and response.tokens_used == 45
    assert response.model.startswith("base:")
    # the observation reached the model as a tool message
    last_prompt = agent.generator.seen[-1]
    assert any(m["role"] == "tool" and "733.6" in m["content"] for m in last_prompt)


def test_tool_timeout_becomes_an_observation():
    agent = make_agent(
        call("slow_tool", seconds=1.0),
        call("final_answer", answer="La herramienta no respondió."),
        timeouts={"slow_tool": 0.1},
    )
    started = time.perf_counter()
    response = agent.run("tarea", max_steps=3, brain="base")
    assert time.perf_counter() - started < 0.9
    assert response.steps[0].observation.startswith("TimeoutError")
    assert response.succeeded


def test_malformed_json_is_reported_and_the_agent_recovers():
    broken = '<tool_call>{"name": "calcular_imv", "arguments": {"case": oops}}</tool_call>'
    agent = make_agent(broken, call("calcular_imv", case=CASE), call("final_answer", answer="733,60"))
    response = agent.run("tarea", max_steps=5, brain="base")
    assert response.steps[0].action is None
    assert "not valid JSON" in response.steps[0].observation
    assert response.succeeded and response.n_steps == 3


def test_invalid_arguments_and_unknown_tools_do_not_kill_the_loop():
    agent = make_agent(
        call("calcular_imv", case={"applicant_id": "x"}),
        call("no_existe", x=1),
        call("final_answer", answer="Falta información."),
    )
    response = agent.run("tarea", max_steps=5, brain="base")
    assert response.steps[0].observation.startswith("Error: invalid arguments")
    assert "unknown tool" in response.steps[1].observation
    assert response.succeeded


def test_repeated_identical_calls_are_not_executed_and_stop_the_run():
    agent = make_agent(call("search_knowledge_base", query="x"))  # forever the same call
    response = agent.run("tarea", max_steps=10, brain="base")
    assert not response.succeeded
    assert response.n_steps == 4  # 1 real call + 3 repeats
    assert all("repetida" in s.observation for s in response.steps[1:])


def test_side_effect_tool_needs_an_explicit_request(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    outputs = (call("guardar_informe", contenido="Informe de simulación IMV: " + "importe 733,60 €/mes. " * 5), call("final_answer", answer="ok"))
    response = make_agent(*outputs).run("¿Cuánto IMV me toca?", max_steps=3, brain="base")
    assert response.steps[0].observation.startswith("No ejecutado")
    assert not list(tmp_path.iterdir())

    response = make_agent(*outputs).run("Guarda un informe TXT con el cálculo", max_steps=3, brain="base")
    assert not response.steps[0].observation.startswith("No ejecutado")
    assert list(tmp_path.iterdir())


def test_max_steps_is_respected():
    outputs = [call("search_knowledge_base", query=f"pregunta {i}") for i in range(20)]
    response = make_agent(*outputs).run("tarea", max_steps=4, brain="base")
    assert not response.succeeded and response.n_steps == 4 and len(response.steps) == 4


def test_plain_text_answer_is_accepted_and_flagged():
    response = make_agent("<think>fácil</think>No puedo calcularlo: falta la fecha de nacimiento.").run(
        "tarea", max_steps=3, brain="thinking"
    )
    assert response.succeeded and response.final_answer.startswith("No puedo")
    assert "sin final_answer" in response.steps[0].thought and "fácil" in response.steps[0].thought


def test_only_the_first_of_several_calls_is_executed():
    two = call("search_knowledge_base", query="a") + "\n" + call("calcular_imv", case=CASE)
    response = make_agent(two, call("final_answer", answer="ok")).run("tarea", max_steps=3, brain="base")
    assert response.steps[0].action.name == "search_knowledge_base"
    assert "las demás se ignoraron" in response.steps[0].observation


def test_memory_is_summarised_when_over_budget():
    outputs = [call("search_knowledge_base", query=f"pregunta {i}") for i in range(6)]
    outputs.append(call("final_answer", answer="ok"))
    agent = make_agent(*outputs, token_budget=200, token_counter=lambda spec, text: len(text) // 4)
    response = agent.run("TAREA ORIGINAL con fecha 2026-06-01", max_steps=10, brain="base")
    assert response.succeeded
    # Every prompt keeps the task verbatim as the first user message...
    assert all(p[1] == {"role": "user", "content": "TAREA ORIGINAL con fecha 2026-06-01"}
               for p in agent.generator.seen)
    # ...and after compression the summary sits in the system prompt.
    assert any("Summary of the conversation so far" in p[0]["content"] for p in agent.generator.seen)


def test_memory_policy_pins_the_task_trims_observations_and_survives_a_failed_summary():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=50, keep_last=2)
    memory.add("user", "la tarea")
    memory.add("assistant", "a1")
    memory.add("tool", "x" * 5000, name="consultar_boe")
    memory.add("assistant", "a2")
    memory.add("tool", "obs2", name="calcular_imv")
    shown = []

    def summarize(messages):
        shown.extend(messages)
        raise RuntimeError("model down")

    assert memory.compress(summarize)
    assert memory.messages[0]["content"] == "la tarea" and len(memory.messages) == 3
    assert len(shown[1]["content"]) < 1600 and "recortada" in shown[1]["content"]
    assert "consultar_boe" in memory.summary  # mechanical digest after the failure
    assert memory.compressions == 1


def test_brains():
    assert brain_spec("thinking").enable_thinking and not brain_spec("base").enable_thinking
    with pytest.raises(ValueError):
        brain_spec("gpt")


def test_rlm_brain_needs_the_adapter(monkeypatch):
    monkeypatch.delenv("ARCA_RLM_ADAPTER", raising=False)
    with pytest.raises(NotImplementedError):
        brain_spec("rlm")
    monkeypatch.setenv("ARCA_RLM_ADAPTER", "JES0406/imv-sft-lora/grpo_final")
    assert brain_spec("rlm").adapter == "JES0406/imv-sft-lora/grpo_final"


def test_knowledge_base_hits_are_rendered_with_citable_ids():
    chunk = SimpleNamespace(id="ley-19-2021-0137", source="ley-19-2021.md", text="texto " * 400)
    text = render_hits([SimpleNamespace(chunk=chunk, score=0.9)])
    assert text.startswith("[ley-19-2021-0137] (ley-19-2021.md)\n") and text.endswith("[...]")
    assert render_hits([]).startswith("Sin resultados")


def test_hung_tools_do_not_block_later_calls():
    agent = make_agent(timeouts={"slow_tool": 0.05, "calcular_imv": 5})
    for _ in range(6):  # more hung calls than a small worker pool would hold
        assert agent.execute("slow_tool", {"seconds": 0.5}).startswith("TimeoutError")
    assert '"total_monthly_eur": 733.6' in agent.execute("calcular_imv", {"case": CASE})


def test_summary_is_capped():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10, keep_last=1,
                                max_summary_chars=100)
    for i in range(5):
        memory.add("user" if i == 0 else "assistant", f"m{i}")
        memory.add("tool", "y" * 50, name="t")
        memory.compress(lambda msgs: "z" * 80)
    assert len(memory.summary) <= 106


def test_rlm_brain_borrows_the_phase1_model_by_resolved_adapter_path(monkeypatch):
    import sys

    from agent.models import BrainSpec, ModelPool

    fake_inference = SimpleNamespace(_resolve_adapter=lambda a: "/cache/snap/grpo_final")
    monkeypatch.setitem(sys.modules, "rlm.inference", fake_inference)
    reasoning = SimpleNamespace(model="M", tokenizer="T", base_model="Qwen/Qwen3-1.7B",
                                adapter_path="/cache/snap/grpo_final")
    fake_app = SimpleNamespace(backends=SimpleNamespace(reasoning=lambda: reasoning))
    monkeypatch.setitem(sys.modules, "api.app", fake_app)
    spec = BrainSpec("rlm", "Qwen/Qwen3-1.7B", "JES0406/imv-sft-lora/grpo_final", False, 1024)
    assert ModelPool._borrow(spec) == ("T", "M")


def test_sessions_carry_previous_turns_into_the_next_run():
    agent = make_agent(
        call("final_answer", answer="Le corresponden 733,60 €/mes."),
        call("final_answer", answer="Sí, la misma cifra."),
    )
    first = agent.run("¿Cuánto IMV para el caso A?", max_steps=3, brain="base", session_id="new")
    assert first.session_id and len(first.session_id) >= 32  # issued by the server
    second = agent.run("¿Y si lo pide mañana?", max_steps=3, brain="base", session_id=first.session_id)
    assert second.session_id == first.session_id
    system = agent.generator.seen[-1][0]["content"]
    assert "Usuario: ¿Cuánto IMV para el caso A?" in system and "733,60" in system
    # another session starts clean, and a run without session keeps nothing
    agent.run("otra cosa", max_steps=3, brain="base", session_id="new")
    assert "Conversación anterior" not in agent.generator.seen[-1][0]["content"]
    assert agent.run("sin sesión", max_steps=3, brain="base").session_id is None


def test_client_chosen_or_expired_session_ids_are_rejected():
    from agent.react_agent import UnknownSession

    agent = make_agent(call("final_answer", answer="ok"))
    with pytest.raises(UnknownSession):
        agent.run("t", max_steps=2, brain="base", session_id="s1")


def test_old_turns_are_summarised_and_sessions_are_bounded(monkeypatch):
    from agent import react_agent
    from agent.memory import SessionMemory

    session = SessionMemory(count_tokens=len, token_budget=120, keep_turns=1)
    for i in range(4):
        session.add_turn(f"pregunta {i} " + "x" * 30, f"respuesta {i}", lambda msgs: "- resumido")
    assert session.compressions >= 1 and len(session.turns) == 1
    assert "resumido" in session.context() and "pregunta 3" in session.context()

    monkeypatch.setattr(react_agent, "MAX_SESSIONS", 2)
    agent = make_agent(call("final_answer", answer="ok"))
    ids = [agent.run("t", max_steps=2, brain="base", session_id="new").session_id for _ in range(3)]
    assert list(agent.sessions) == ids[1:]
