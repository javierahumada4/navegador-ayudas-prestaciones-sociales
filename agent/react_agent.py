"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool we built.

Phases 1 to 3 meet here. The brain is a model (``agent/models.py``: ``base``, our phase-1 RLM
``rlm``, or Qwen3 with native reasoning ``thinking``); the hands are the phase-2 tools
(``calcular_imv``, ``consultar_ayudas``, ``guardar_informe``, ``consultar_boe``) plus
``search_knowledge_base`` over the phase-3 retriever; the loop is below.

Decisions (and why):

- **One action per step.** If the model writes several ``<tool_call>`` blocks only the first is
  executed and the observation says so: the trace stays a clean Thought/Action/Observation
  sequence and every step can be checked.
- **Timeouts per tool** (``TOOL_TIMEOUTS``, overridable with ``ARCA_AGENT_TOOL_TIMEOUT``): each
  call runs in its own daemon thread and the agent stops waiting after the limit. Python cannot
  kill the thread; a slow HTTP call finishes in the background (``requests`` has its own 15 s
  limit) and its result is discarded. The model sees ``TimeoutError`` and decides. At most
  ``MAX_STUCK_CALLS`` timed-out calls may still be running; beyond that, tools are refused
  immediately instead of piling up threads.
- **Error recovery.** Malformed JSON, unknown tools, invalid arguments, exceptions and timeouts
  all become observations (``ToolRegistry.call`` never raises); nothing kills the loop.
- **Loop detection.** An identical call (same name and arguments) is not executed again: the
  observation tells the model to reuse the previous result. After ``MAX_REPEATS`` repeats the
  run stops as failed instead of burning the remaining steps.
- **Tools with side effects** (``requires_confirmation``, i.e. ``guardar_informe``) only run if
  the task itself asks for a report or file (same detector as phase 2); otherwise the
  observation says it was not executed. The API has no second turn to ask the user.
- **Answer without ``final_answer``**: accepted as the final answer (small models often answer
  in plain text) and flagged in the step's thought.
- **Memory** (``agent/memory.py``), two levels. Within a run: the task stays verbatim, old steps
  are summarised by the same brain when the context exceeds ``ARCA_AGENT_TOKEN_BUDGET`` (default
  6000 tokens). Across runs (``session_id``): each turn's task and final answer are kept and the
  oldest turns are summarised past ``ARCA_AGENT_SESSION_BUDGET`` (default 1500 tokens). The server
  issues session ids (send ``session_id="new"``, read it back from the response); client-chosen
  ids are rejected so nobody can read someone else's conversation by guessing an id. Sessions
  live in process memory only (at most ``MAX_SESSIONS``, least recently used dropped first): they
  may hold personal data and are never written to disk.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
import threading
from dataclasses import dataclass, field, replace

from agent.memory import ConversationMemory, SessionMemory
from agent.models import BrainSpec, brain_spec
from agent.prompts import REACT_SYSTEM_PROMPT, SUMMARY_PROMPT
from api.schemas import AgentAction, AgentResponse, AgentStep
from tool_use.parser import format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import consultar_boe, default_registry

MAX_REPEATS = 3
MAX_SESSIONS = 100
KB_PASSAGE_CHARS = 1600  # whole chunks: the tuned chunking is 1200 characters (was 900)
TOOL_TIMEOUTS = {  # seconds; the first search loads the embedder and may index the corpus
    "calcular_imv": 10,
    "guardar_informe": 10,
    "consultar_ayudas": 30,
    "consultar_boe": 30,
    "search_knowledge_base": 180,
}
# One daemon thread per call (a hung tool never blocks the next call), and a cap on threads
# still running after their timeout so a stuck dependency cannot pile up threads forever.
MAX_STUCK_CALLS = 16
_STUCK = threading.BoundedSemaphore(MAX_STUCK_CALLS)


@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task, in Spanish.
    """
    return answer


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Busca en la base de conocimiento legal del IMV/CAPI (Ley 19/2021, reales decretos, página de
    la Seguridad Social con las cuantías de 2026) y devuelve los fragmentos más relevantes con su
    id entre corchetes, para citarlos.

    Args:
        query: Lo que se busca, escrito como una pregunta.
        top_k: Cuántos fragmentos devolver (1 a 10).
    """
    from rag.generate import build_pipeline

    retriever = os.environ.get("ARCA_AGENT_RETRIEVER", "hybrid")
    hits = build_pipeline().retrieve(query, max(1, min(int(top_k), 10)), retriever)
    return render_hits(hits)


def render_hits(hits) -> str:
    if not hits:
        return "Sin resultados en la base de conocimiento."
    blocks = []
    for hit in hits:
        text = " ".join(hit.chunk.text.split())
        if len(text) > KB_PASSAGE_CHARS:
            text = text[:KB_PASSAGE_CHARS] + " [...]"
        blocks.append(f"[{hit.chunk.id}] ({hit.chunk.source})\n{text}")
    return "\n\n".join(blocks)


def report_requested(task: str) -> bool:
    from tool_use.executor import _report_requested

    return _report_requested(task)


def agent_registry() -> ToolRegistry:
    """Every tool of phases 2 and 3, plus final_answer. BOE is always on for the agent."""
    registry = default_registry()
    registry.register(consultar_boe, search_knowledge_base, final_answer)
    return registry


def call_with_timeout(registry: ToolRegistry, name: str, arguments: dict, limit: float):
    """Run ``registry.call`` in its own daemon thread; None if it did not finish in ``limit`` s.

    Shared by the JSON agent and the code agent so both have the same timeouts.
    """
    if not _STUCK.acquire(blocking=False):
        from api.schemas import ToolCallRecord

        return ToolCallRecord(
            name=name, arguments=arguments, ok=False, latency_ms=0.0,
            error="demasiadas herramientas bloqueadas en este momento",
        )
    box: dict = {}

    def target() -> None:
        try:
            box["record"] = registry.call(name, arguments)
        finally:
            _STUCK.release()

    thread = threading.Thread(target=target, name=f"agent-tool-{name}", daemon=True)
    thread.start()
    thread.join(timeout=limit)
    return box.get("record")


CITATION = re.compile(r"\[([a-z0-9]+(?:-[a-z0-9]+)*-\d{4})\]")
THOUGHT_LABEL = re.compile(r"^\s*(?:Pensamiento|Thought)\s*:\s*", re.IGNORECASE)


def ungrounded_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers of the answer (>= 10, so not list markers) found in no observation or task."""
    from agent.evaluate import NUMBER, parse_number

    known = set()
    for text in sources:
        known |= {round(n, 2) for n in (parse_number(t) for t in NUMBER.findall(text)) if n is not None}
    missing = []
    for token in NUMBER.findall(CITATION.sub("", answer)):
        value = parse_number(token)
        if value is not None and abs(value) >= 10 and round(value, 2) not in known:
            missing.append(token)
    return missing


def cited_ids(observations: list[str]) -> list[str]:
    """Ids of the first passage of every knowledge-base observation, in order, without repeats."""
    ids: list[str] = []
    for text in observations:
        match = re.match(r"\[([^\]]+)\]", text)
        if match and match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


class UnknownSession(LookupError):
    """A session_id the server did not issue or has already dropped."""


Generator = Callable[[list[dict], BrainSpec, list[dict]], tuple[str, int, int]]


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=agent_registry)
    token_budget: int = field(
        default_factory=lambda: int(os.environ.get("ARCA_AGENT_TOKEN_BUDGET", "6000"))
    )
    generator: Generator | None = None  # injected in tests; default: agent.models.generate
    token_counter: Callable[[BrainSpec, str], int] | None = None
    timeouts: dict[str, float] = field(default_factory=lambda: dict(TOOL_TIMEOUTS))
    sessions: OrderedDict = field(default_factory=OrderedDict)
    _sessions_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls()
        agent.load()
        return agent

    def load(self) -> None:
        """Check the configuration; weights load lazily on the first run of each brain, so
        ``/health`` stays cheap and only the brains actually used take GPU memory."""
        from importlib.util import find_spec

        if self.generator is None and (find_spec("torch") is None or find_spec("transformers") is None):
            raise NotImplementedError("Phase 4 needs torch/transformers: uv sync --extra train")
        brain_spec("base")
        if "ARCA_AGENT_TOOL_TIMEOUT" in os.environ:
            limit = float(os.environ["ARCA_AGENT_TOOL_TIMEOUT"])
            self.timeouts = {name: limit for name in self.timeouts}

    def system_prompt(self, max_steps: int) -> str:
        lines = []
        for t in self.registry.tools.values():
            first = t.description.strip().splitlines()[0] if t.description.strip() else ""
            lines.append(f"- {t.name}: {first}")
        return REACT_SYSTEM_PROMPT.format(max_steps=max_steps, tools_description="\n".join(lines))

    # -- model ---------------------------------------------------------------

    def generate(self, messages: list[dict], spec: BrainSpec) -> tuple[str, int, int]:
        if self.generator is not None:
            return self.generator(messages, spec, self.registry.definitions())
        from agent.models import generate

        return generate(spec, messages, self.registry.definitions())

    def count_tokens(self, spec: BrainSpec, text: str) -> int:
        if self.token_counter is not None:
            return self.token_counter(spec, text)
        if self.generator is not None:
            return len(text) // 4  # scripted runs: rough estimate, no tokenizer
        from agent.models import count_tokens

        return count_tokens(spec, text)

    def summarize(self, messages: list[dict], spec: BrainSpec, usage: Counter) -> str:
        history = "\n".join(f"{m.get('name') or m['role']}: {m['content']}" for m in messages)
        prompt = [{"role": "user", "content": SUMMARY_PROMPT.format(history=history)}]
        short = replace(spec, enable_thinking=False, max_new_tokens=384)
        text, prompt_tokens, completion_tokens = self.generate(prompt, short)
        usage["tokens"] += prompt_tokens + completion_tokens
        return parse_tool_calls(text).text  # drop a stray <think> or tool call

    # -- tools ---------------------------------------------------------------

    def execute(self, name: str, arguments: dict) -> str:
        limit = self.timeouts.get(name, 30)
        record = call_with_timeout(self.registry, name, arguments, limit)
        if record is None:
            return (
                f"TimeoutError: la herramienta {name} no respondió en {limit:g} s. "
                "Prueba otra herramienta o responde con lo que tengas, indicando el fallo."
            )
        if record.ok:
            return format_tool_result(name, record.result)
        return f"Error: {record.error}"

    # -- loop ----------------------------------------------------------------

    def session(self, session_id: str, spec: BrainSpec) -> tuple[str, SessionMemory]:
        """``"new"`` opens a session with a server-issued id; any other id must have been issued.

        Ids are 32 random URL-safe characters (``secrets.token_urlsafe(24)``), so a caller cannot
        guess another person's session; an id the server did not issue (or already dropped)
        raises ``UnknownSession`` and the API answers 404.
        """
        with self._sessions_lock:
            if session_id == "new":
                session_id = secrets.token_urlsafe(24)
                self.sessions[session_id] = SessionMemory(
                    count_tokens=lambda text: self.count_tokens(spec, text),
                    token_budget=int(os.environ.get("ARCA_AGENT_SESSION_BUDGET", "1500")),
                )
                while len(self.sessions) > MAX_SESSIONS:
                    self.sessions.popitem(last=False)
            elif session_id in self.sessions:
                self.sessions.move_to_end(session_id)
            else:
                raise UnknownSession("unknown or expired session_id: start one with session_id='new'")
            return session_id, self.sessions[session_id]

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm",
            session_id: str | None = None) -> AgentResponse:
        spec = brain_spec(brain)
        if not session_id:
            return self._run(task, max_steps, spec, "")
        session_id, session = self.session(session_id, spec)
        with session.lock:  # one turn at a time per conversation
            response = self._run(task, max_steps, spec, session.context())
            usage = Counter()
            session.add_turn(task, response.final_answer,
                             lambda msgs: self.summarize(msgs, spec, usage))
        response.tokens_used += usage["tokens"]
        response.session_id = session_id
        return response

    def _run(self, task: str, max_steps: int, spec: BrainSpec, history: str) -> AgentResponse:
        usage: Counter = Counter()
        steps: list[AgentStep] = []
        seen: Counter = Counter()
        repeats = 0
        system = self.system_prompt(max_steps)
        if history:
            system += f"\nConversación anterior con este usuario:\n{history}\n"
        memory = ConversationMemory(
            system,
            count_tokens=lambda text: self.count_tokens(spec, text),
            token_budget=self.token_budget,
        )
        memory.add("user", task)
        started = time.monotonic()
        budget_s = float(os.environ.get("ARCA_AGENT_TIME_BUDGET", "240"))
        observations: list[str] = []  # every tool result, for the grounding check
        kb_observations: list[str] = []
        verified = False

        def finish(answer: str, n: int, ok: bool) -> AgentResponse:
            if ok and kb_observations and not CITATION.search(answer):
                ids = cited_ids(kb_observations)
                if ids:  # the model used the knowledge base but did not cite it
                    answer = f"{answer}\n\nFragmentos consultados: " + ", ".join(f"[{i}]" for i in ids)
            return AgentResponse(
                final_answer=answer, steps=steps, n_steps=n, succeeded=ok,
                model=f"{spec.name}:{spec.label}", tokens_used=usage["tokens"],
            )

        for step in range(max_steps):
            if time.monotonic() - started > budget_s:  # ngrok cuts requests at 300 s
                last = observations[-1][:400] if observations else ""
                return finish(
                    "No he podido terminar a tiempo." + (f" Último resultado obtenido: {last}" if last else ""),
                    step, False,
                )
            raw, prompt_tokens, completion_tokens = self.generate(memory.render(), spec)
            usage["tokens"] += prompt_tokens + completion_tokens
            parsed = parse_tool_calls(raw)
            thought = "\n".join(p for p in (parsed.thinking, parsed.text) if p).strip()
            valid = [c for c in parsed.tool_calls if c.error is None]
            broken = [c for c in parsed.tool_calls if c.error is not None]

            if not valid:
                if broken:  # malformed call: tell the model, let it fix it
                    observation = "Error: " + "; ".join(c.error for c in broken) + (
                        ". Escribe la llamada como JSON válido dentro de <tool_call>...</tool_call>."
                    )
                    steps.append(AgentStep(thought=thought or "(llamada mal formada)", observation=observation))
                    memory.add("assistant", raw)
                    memory.add("tool", observation, name="error")
                    memory.compress(lambda msgs: self.summarize(msgs, spec, usage))
                    continue
                if parsed.text:  # plain-text answer: accept it, flag it
                    steps.append(AgentStep(thought=f"{thought}\n(respuesta sin final_answer)".strip()))
                    return finish(THOUGHT_LABEL.sub("", parsed.text), step + 1, True)
                observation = "Respuesta vacía: llama a una herramienta o a final_answer."
                steps.append(AgentStep(thought=thought or "(vacío)", observation=observation))
                memory.add("assistant", raw)
                memory.add("user", observation)
                continue

            call = valid[0]
            action = AgentAction(name=call.name, arguments=call.arguments)
            if call.name == "final_answer":
                answer = str(call.arguments.get("answer") or parsed.text or "").strip()
                missing = ungrounded_numbers(answer, [task, history, *observations])
                if missing and not verified and step + 1 < max_steps:
                    # One chance to check figures that no tool returned ("15 días" invented).
                    verified = True
                    observation = (
                        "Comprobación: tu respuesta incluye cifras que no aparecen en ninguna "
                        f"observación ni en la tarea: {', '.join(missing[:5])}. Si es un dato, "
                        "búscalo con una herramienta; si lo has calculado a partir de datos "
                        "observados o no lo encuentras, dilo y vuelve a llamar a final_answer."
                    )
                    steps.append(AgentStep(thought=thought, action=action, observation=observation))
                    memory.add("assistant", raw)
                    memory.add("tool", observation, name="final_answer")
                    continue
                steps.append(AgentStep(thought=thought, action=action))
                return finish(answer, step + 1, bool(answer))

            key = json.dumps([call.name, call.arguments], sort_keys=True, ensure_ascii=False)
            seen[key] += 1
            spec_tool = self.registry.tools.get(call.name)
            if seen[key] > 1:
                repeats += 1
                observation = (
                    "Llamada repetida idéntica: ya se ejecutó y su resultado está más arriba. "
                    "Úsalo, cambia los argumentos o llama a final_answer."
                )
            elif spec_tool is not None and spec_tool.requires_confirmation and not report_requested(task):
                observation = (
                    f"No ejecutado: {call.name} tiene efectos y el usuario no ha pedido guardar "
                    "ningún informe. Continúa sin esta herramienta."
                )
            else:
                observation = self.execute(call.name, call.arguments)
                observations.append(observation)
                if call.name == "search_knowledge_base":
                    kb_observations.append(observation)
            if len(valid) > 1:
                observation += "\n(Solo se ejecuta una herramienta por paso; las demás se ignoraron.)"
            steps.append(AgentStep(thought=thought, action=action, observation=observation))
            if repeats >= MAX_REPEATS:
                return finish(
                    "No he podido completar la tarea: el agente repetía la misma llamada.",
                    step + 1, False,
                )
            memory.add("assistant", raw)
            memory.add("tool", observation, name=call.name)
            memory.compress(lambda msgs: self.summarize(msgs, spec, usage))

        return finish(
            f"No he podido completar la tarea en {max_steps} pasos.", max_steps, False
        )
