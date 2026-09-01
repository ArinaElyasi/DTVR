"""The model port, and the two implementations of it (spec sections 1.5, 4).

:class:`ModelPort` is the whole surface the graph is allowed to touch. It has
exactly the methods needed to *phrase questions* and *structure answers* -
there is deliberately no method that would let a model decide what happens
next.

Two implementations:

:class:`StubModel`
    Scripted and deterministic. Needs no network and no API key. Every test and
    every ``--stub`` evaluation run uses it, which is what makes the claim
    "the deterministic core works offline" checkable rather than aspirational.

:class:`GroqModel`
    A Groq-hosted model (see :mod:`dtv_rea.groq_model`, imported lazily so
    that nothing here requires ``langchain-groq`` to be importable offline).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from dtv_rea.persona import PersonaScript
from dtv_rea.settings import DIMENSION_TOPICS, MATURITY_LEVELS
from dtv_rea.state import ExtractionResult, Flag, SessionState, Turn


class Focus(BaseModel):
    """What ``decide_gap`` decided to ask about next.

    Built in code and handed to the model. The model never chooses one.
    """

    kind: str  # "dimension" | "goal" | "fabrication_recovery" | "retry_dimension"
    description: str
    dimension: str | None = None
    requirement_id: str | None = None
    flag_id: str | None = None
    attempt: int = 1

    def as_dict(self) -> dict[str, str]:
        """Serialisable form stored in ``SessionState.pending_focus``."""
        return {
            "kind": self.kind,
            "description": self.description,
            "dimension": self.dimension or "",
            "requirement_id": self.requirement_id or "",
            "flag_id": self.flag_id or "",
            "attempt": str(self.attempt),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "Focus":
        return cls(
            kind=raw.get("kind", "dimension"),
            description=raw.get("description", ""),
            dimension=raw.get("dimension") or None,
            requirement_id=raw.get("requirement_id") or None,
            flag_id=raw.get("flag_id") or None,
            attempt=int(raw.get("attempt") or 1),
        )


class MaturityProposal(BaseModel):
    """P3's output. A level with no description is not a valid proposal."""

    level: str
    reasoning: str
    description: str
    rationale: str = ""
    user_roles: list[str] = Field(default_factory=list)
    context_of_use: str = ""


@runtime_checkable
class ModelPort(Protocol):
    """Everything the graph may ask a language model to do. Nothing more."""

    def opener(self, state: SessionState) -> str:
        """P2 - phrase the one open question that starts the interview."""

    def propose_maturity(self, state: SessionState) -> MaturityProposal:
        """P3 - map the stated purpose onto a 4R level, with a description."""

    def ask(self, state: SessionState, focus: Focus) -> str:
        """P4 - phrase one question about the topic code has already chosen."""

    def extract(
        self, state: SessionState, answer: Turn, focus: Focus
    ) -> ExtractionResult:
        """P5 - structure one stakeholder answer into candidate requirements."""

    def phrase_flag(self, state: SessionState, flag: Flag) -> str:
        """P6 - put a validator finding to the human so they can settle it."""


# --------------------------------------------------------------------------
# Call logging - shared by both implementations
# --------------------------------------------------------------------------


class CallLogger:
    """Appends one JSON object per model call to ``llm_calls.jsonl``.

    Records a *hash* of the prompt rather than the prompt itself: the log is an
    operational record, not a second copy of the transcript.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[dict[str, object]] = []

    def log(self, **fields: object) -> None:
        self.records.append(fields)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(fields, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# StubModel
# --------------------------------------------------------------------------


class StubModel:
    """A deterministic stand-in for the hosted model.

    Questions are templated, extractions come from the persona script. Given
    the same persona it produces byte-identical runs, which is what lets the
    graph be tested before any API call exists.
    """

    name = "stub"

    def __init__(
        self,
        script: PersonaScript | None = None,
        logger: CallLogger | None = None,
    ) -> None:
        self.script = script
        self.logger = logger or CallLogger()

    # -- P2 ------------------------------------------------------------
    def opener(self, state: SessionState) -> str:
        self.logger.log(call="opener", model=self.name, prompt="P2")
        return (
            "To start: what would you like a digital twin of your system to do "
            "for you, and why does that matter right now? It helps if you say "
            "who would be using it and when."
        )

    # -- P3 ------------------------------------------------------------
    def propose_maturity(self, state: SessionState) -> MaturityProposal:
        self.logger.log(call="propose_maturity", model=self.name, prompt="P3")
        if self.script is not None:
            level = self.script.expected_level
            description = self.script.maturity_description
            reasoning = self.script.maturity_reasoning
        else:
            level = "Replication"
            description = (
                "The twin will show you what the machine is doing and work out "
                "when something has gone wrong. It will not predict problems "
                "in advance, and it will not change settings by itself."
            )
            reasoning = "Default stub proposal."
        if level not in MATURITY_LEVELS:  # pragma: no cover - guards typos
            level = "Replication"
        # Deliberately empty unless the persona says otherwise. The stub does
        # not paraphrase, so echoing the purpose statement back into
        # "rationale" would only duplicate it in the output document.
        record = (self.script.data.get("purpose_record", {}) if self.script else {})
        return MaturityProposal(
            level=level,
            reasoning=reasoning,
            description=description,
            rationale=str(record.get("rationale", "")),
            user_roles=list(record.get("user_roles", [])),
            context_of_use=str(record.get("context_of_use", "")),
        )

    # -- P4 ------------------------------------------------------------
    def ask(self, state: SessionState, focus: Focus) -> str:
        self.logger.log(
            call="ask", model=self.name, prompt="P4", focus=focus.kind
        )
        if focus.kind == "fabrication_recovery":
            return (
                f"I do not have a figure for this one: \"{focus.description}\". "
                "I would rather ask than guess - what should that value be?"
            )
        if focus.kind == "goal":
            requirement = state.requirement(focus.requirement_id or "")
            text = requirement.text if requirement else focus.description
            return (
                f"You said: \"{text}\". How well does that have to work - what "
                "is the number it has to hit?"
            )
        if focus.kind == "retry_dimension":
            topic = DIMENSION_TOPICS.get(focus.dimension or "", focus.description)
            return (
                f"Let me try that differently. Thinking about {topic} - is "
                "there anything there you need, or does that simply not apply "
                "to your situation?"
            )
        topic = DIMENSION_TOPICS.get(focus.dimension or "", focus.description)
        return f"Tell me about {topic}."

    # -- P5 ------------------------------------------------------------
    def extract(
        self, state: SessionState, answer: Turn, focus: Focus
    ) -> ExtractionResult:
        self.logger.log(
            call="extract",
            model=self.name,
            prompt="P5",
            turn=answer.index,
            parse_retries=0,
        )
        if self.script is None:
            return ExtractionResult()
        return self.script.extraction_for(answer.index, session=state)

    # -- P6 ------------------------------------------------------------
    def phrase_flag(self, state: SessionState, flag: Flag) -> str:
        self.logger.log(
            call="phrase_flag", model=self.name, prompt="P6", code=flag.code
        )
        tail = {
            "duplicate_obligation": (
                " Are these one thing or two? Keeping both is a perfectly good "
                "answer."
            ),
            "unverifiable_predicate": (
                " What number should that be measured against?"
            ),
            "fabricated_goal": (
                " I did not want to guess it. What should that value be?"
            ),
            "maturity_inconsistency": " Which of those should stand?",
            "orphan_goal": " Which requirement was that number meant for?",
        }.get(flag.code, " How would you like to settle this?")
        return flag.message + tail


__all__ = [
    "CallLogger",
    "Focus",
    "MaturityProposal",
    "ModelPort",
    "StubModel",
]
