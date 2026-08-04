"""Driving the graph from one interrupt to the next.

The graph is UI-agnostic by design: it pauses at an ``interrupt()`` and does not
care whether the answer comes from a terminal, a persona script, or a web form
added later. This module is the small amount of glue that turns "the graph
paused" into "somebody answered", and it is shared by the CLI and the
evaluation harness so both drive exactly the same code path.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from langgraph.types import Command

from dtv_rea.settings import GRAPH_RECURSION_LIMIT
from dtv_rea.state import SessionState, Turn, new_session


class Answerer(Protocol):
    """Anything that can answer an interrupt payload."""

    def __call__(self, payload: dict[str, Any]) -> str: ...


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the pending interrupt's payload out of an invoke result."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"kind": "answer", "question": str(value)}


def run_session(
    graph: Any,
    session_id: str,
    answerer: Answerer,
    *,
    resume: bool = False,
    on_agent_turn: Callable[[Turn], None] | None = None,
    max_interrupts: int = 200,
) -> SessionState:
    """Run an interview to completion and return the final session state.

    ``resume`` picks a paused session back up from the checkpointer instead of
    starting a new one - the stakeholder who left mid-interview and came back.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    reported = 0

    def report(state: dict[str, Any]) -> None:
        nonlocal reported
        session = state.get("session")
        if session is None or on_agent_turn is None:
            return
        for turn in session.turns[reported:]:
            if turn.role == "agent":
                on_agent_turn(turn)
        reported = len(session.turns)

    if resume:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            # Nothing pending - the session already finished.
            return snapshot.values["session"]
        result = graph.invoke(None, config)
    else:
        result = graph.invoke({"session": new_session(session_id)}, config)

    report(result)

    for _ in range(max_interrupts):
        payload = _interrupt_payload(result)
        if payload is None:
            break
        answer = answerer(payload)
        result = graph.invoke(Command(resume=answer), config)
        report(result)
    else:  # pragma: no cover - only a malformed script gets here
        raise RuntimeError(
            f"Interview exceeded {max_interrupts} interrupts without finishing."
        )

    session = result.get("session")
    if session is None:  # pragma: no cover - defensive
        session = graph.get_state(config).values["session"]
    return session


def persona_answerer(script: Any) -> Answerer:
    """Adapt a :class:`~dtv_rea.persona.PersonaScript` to the answerer shape."""

    def answer(payload: dict[str, Any]) -> str:
        return script.respond(payload)

    return answer


__all__ = ["Answerer", "persona_answerer", "run_session"]
