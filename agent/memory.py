"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 6  # most recent messages that are never summarised
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

    def tokens(self) -> int:
        text = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return self.count_tokens(text)

    def over_budget(self) -> bool:
        return self.tokens() > self.token_budget

    def render(self) -> list[dict]:
        """Messages to send to the model: system (+ summary) followed by the live messages."""
        system = self.system_prompt
        if self.summary:
            system += f"\n\nSummary of the conversation so far:\n{self.summary}"
        return [{"role": "system", "content": system}, *self.messages]

    pinned: int = 1  # first messages that stay verbatim: the user's task
    max_observation_chars: int = 1500  # tool outputs are cut to this before summarising
    max_summary_chars: int = 4000  # the summary itself must not grow without bound
    compressions: int = 0

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Our policy: what is summarised and when.

        - When: only when the rendered context exceeds ``token_budget``.
        - Always kept verbatim: the first ``pinned`` messages (the user's task: losing its exact
          wording, numbers or dates is the worst failure of a summary) and the last
          ``keep_last`` messages (the step in progress and its observation).
        - Summarised: everything in between, with every tool observation cut to
          ``max_observation_chars`` first. Observations are the bulk of the context (a BOE
          article, a list of grants) and the model only needs the facts it already used.
        - If the summary call fails, the middle is replaced by a mechanical digest (tool names,
          arguments and the start of each observation): the loop must never die on memory.
        Returns True if a compression happened.
        """
        if not self.over_budget() or len(self.messages) <= self.pinned + self.keep_last:
            return False
        head = self.messages[: self.pinned]
        middle = self.messages[self.pinned : -self.keep_last]
        recent = self.messages[-self.keep_last :]
        trimmed = [self._trim(m) for m in middle]
        try:
            new_summary = summarize(trimmed).strip()
        except Exception:  # noqa: BLE001 - fall back to a digest that needs no model
            new_summary = ""
        if not new_summary:
            new_summary = digest(trimmed, max_chars=200)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        if len(self.summary) > self.max_summary_chars:  # keep the most recent facts
            self.summary = "[...]\n" + self.summary[-self.max_summary_chars :]
        self.messages = head + recent
        self.compressions += 1
        return True

    def _trim(self, message: dict) -> dict:
        if message.get("role") != "tool" or len(message["content"]) <= self.max_observation_chars:
            return message
        cut = message["content"][: self.max_observation_chars]
        return {**message, "content": f"{cut} [... observación recortada]"}


def digest(messages: list[dict], max_chars: int = 200) -> str:
    """A model-free summary: one line per message, observations shortened."""
    lines = []
    for m in messages:
        text = " ".join(m["content"].split())[:max_chars]
        who = m.get("name") or m["role"]
        lines.append(f"- {who}: {text}")
    return "\n".join(lines)


@dataclass
class SessionMemory:
    """Multi-turn memory across ``POST /agent`` calls that share a ``session_id``.

    Each finished turn is stored as (task, final answer): the tool traffic of a turn stays in that
    turn, only its conclusion carries over. Before a new turn the session renders as context for
    the system prompt (summary + the last turns verbatim). When the stored turns exceed
    ``token_budget`` the oldest ones, all but ``keep_turns``, are summarised by the model (or by a
    mechanical digest if the summary call fails), so a long conversation keeps a bounded size.
    """

    count_tokens: Callable[[str], int]
    token_budget: int = 1500
    keep_turns: int = 2
    max_summary_chars: int = 3000
    turns: list[tuple[str, str]] = field(default_factory=list)
    summary: str = ""
    compressions: int = 0

    def context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Resumen de turnos anteriores:\n{self.summary}")
        for task, answer in self.turns:
            parts.append(f"Usuario: {task}\nAgente: {answer}")
        return "\n\n".join(parts)

    def add_turn(self, task: str, answer: str, summarize: Callable[[list[dict]], str]) -> None:
        self.turns.append((task, answer))
        if self.count_tokens(self.context()) <= self.token_budget or len(self.turns) <= self.keep_turns:
            return
        old, self.turns = self.turns[: -self.keep_turns], self.turns[-self.keep_turns :]
        messages = [m for t, a in old for m in ({"role": "user", "content": t},
                                                 {"role": "assistant", "content": a})]
        try:
            new_summary = summarize(messages).strip()
        except Exception:  # noqa: BLE001
            new_summary = ""
        new_summary = new_summary or digest(messages, max_chars=300)
        self.summary = f"{self.summary}\n{new_summary}".strip()[-self.max_summary_chars :]
        self.compressions += 1
