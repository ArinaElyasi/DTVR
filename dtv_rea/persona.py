"""Scripted stakeholders, for offline runs and for evaluation (spec section 2).

A persona is a JSON file holding two things at once:

* what the **stakeholder** says at each interrupt, and
* what the **stub model** should extract from that answer.

Keeping both in one file is what makes an offline run reproducible end to end.
The stub model is not guessing at the answer it was handed - it is reading the
extraction the persona author wrote, including the deliberately fabricated
goals used to prove that V1 catches them.

No code changes are needed to add a case: drop a new ``<name>.json`` into
``data/personas/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dtv_rea.settings import personas_dir
from dtv_rea.state import (
    ExtractionResult,
    NotApplicableClaim,
    RequirementCandidate,
)


class PersonaError(RuntimeError):
    """Raised when a persona file is missing or malformed."""


def _resolve_target(session: Any | None, text: str) -> str:
    """Map a requirement's text to the id the agent gave it, if it has one."""
    if session is None:
        return text
    for requirement in getattr(session, "requirements", []):
        if requirement.text == text:
            return requirement.id
    return text


class PersonaScript:
    """A scripted stakeholder, consumed in order.

    ``respond`` answers whatever the graph is currently asking for and, for
    interview answers, records which script entry produced which transcript
    turn. The stub model then looks the extraction up by turn index, so the two
    halves can never drift out of step.
    """

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self.data = data
        self.source = source
        self.name: str = str(data.get("id") or (source.stem if source else "persona"))
        self.description: str = str(data.get("description", ""))
        self._turns: list[dict[str, Any]] = list(data.get("turns", []))
        self._cursor = 0
        self._flag_cursor = 0
        #: transcript turn index -> the script entry that produced it
        self._extractions: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "PersonaScript":
        if not path.exists():
            raise PersonaError(f"No persona file at {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise PersonaError(f"{path} is not valid JSON: {error}") from error
        return cls(data, source=path)

    @classmethod
    def by_name(cls, name: str) -> "PersonaScript":
        """Load ``data/personas/<name>.json``.

        ``name`` may also be a path to a file anywhere on disk.
        """
        candidate = Path(name).expanduser()
        if candidate.suffix.lower() == ".json" and candidate.exists():
            return cls.load(candidate)
        return cls.load(personas_dir() / f"{name}.json")

    @classmethod
    def available(cls) -> list[str]:
        directory = personas_dir()
        if not directory.exists():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def respond(self, payload: dict[str, Any]) -> str:
        """Answer one interrupt. ``payload`` is what the graph passed to it."""
        kind = payload.get("kind")
        if kind == "purpose":
            return str(self.data.get("purpose_answer", "")).strip()
        if kind == "maturity":
            return str(self.data.get("maturity", {}).get("answer", "Yes, that is right."))
        if kind == "flag":
            return self._next_flag_answer()
        return self._next_interview_answer(int(payload.get("turn_index", -1)))

    def _next_interview_answer(self, turn_index: int) -> str:
        if self._cursor >= len(self._turns):
            fallback = self.data.get("fallback_answer")
            if fallback is None:
                # Nothing scripted left. Stopping is the honest thing to do -
                # inventing more stakeholder speech would corrupt the run.
                return "stop"
            return str(fallback)

        entry = self._turns[self._cursor]
        self._cursor += 1
        if turn_index >= 0:
            self._extractions[turn_index] = entry
        return str(entry.get("answer", ""))

    def _next_flag_answer(self) -> str:
        answers = self.data.get("flag_answers", [])
        if not answers:
            return "Keep both - they are different things."
        index = min(self._flag_cursor, len(answers) - 1)
        self._flag_cursor += 1
        return str(answers[index])

    # ------------------------------------------------------------------
    # What the stub model reads back
    # ------------------------------------------------------------------

    def extraction_for(
        self, turn_index: int, session: Any | None = None
    ) -> ExtractionResult:
        """The extraction scripted for the answer at ``turn_index``."""
        entry = self._extractions.get(turn_index)
        if entry is None:
            return ExtractionResult()
        return self._build_extraction(entry.get("extract", {}), turn_index, session)

    @staticmethod
    def _build_extraction(
        spec: dict[str, Any], turn_index: int, session: Any | None = None
    ) -> ExtractionResult:
        candidates: list[RequirementCandidate] = []
        for raw in spec.get("requirements", []):
            payload = dict(raw)
            # "ref" is a small convenience: "this" means the turn currently
            # being extracted. A literal integer lets a persona author point a
            # citation somewhere wrong on purpose, which is how the
            # ref-to-an-agent-turn path gets exercised offline.
            reference = payload.pop("ref", "this")
            if reference == "this":
                resolved: int | None = turn_index
            elif reference is None:
                resolved = None
            else:
                resolved = int(reference)
            payload["stakeholder_utterance_ref"] = resolved

            # Personas name the requirement a goal belongs to by its *text*,
            # never by id. Ids are allocated by the agent in commit order and a
            # persona that hardcoded one would break the moment the interview
            # took a different path. Unresolvable text is passed through as-is
            # so that V4's orphan-goal check is the thing that reports it.
            target_text = payload.pop("goal_target_text", None)
            if target_text is not None:
                payload["goal_target_id"] = _resolve_target(session, str(target_text))

            if payload.get("designGoal") is not None:
                payload.setdefault("designGoal_provenance", "stated")
            candidates.append(RequirementCandidate(**payload))

        claims: list[NotApplicableClaim] = []
        for raw in spec.get("not_applicable", []):
            payload = dict(raw)
            payload.setdefault("stakeholder_utterance_ref", turn_index)
            claims.append(NotApplicableClaim(**payload))

        return ExtractionResult(
            requirements=candidates,
            not_applicable=claims,
            note=str(spec.get("note", "")),
        )

    # ------------------------------------------------------------------
    # What the stub model says
    # ------------------------------------------------------------------

    @property
    def purpose_answer(self) -> str:
        """What this stakeholder says when asked what the twin is for."""
        return str(self.data.get("purpose_answer", "")).strip()

    @property
    def expected_level(self) -> str:
        return str(self.data.get("maturity", {}).get("expected_level", "Replication"))

    @property
    def maturity_description(self) -> str:
        return str(
            self.data.get("maturity", {}).get(
                "description",
                "The twin will show you what the machine is doing and work out "
                "when something has gone wrong, using the data it collects. It "
                "will not predict problems before they happen, and it will not "
                "change any machine settings on its own.",
            )
        )

    @property
    def maturity_reasoning(self) -> str:
        return str(
            self.data.get("maturity", {}).get(
                "reasoning",
                "The stated purpose needs the twin to interpret machine "
                "behaviour and act on what it finds, which is more than "
                "representation and less than prediction.",
            )
        )


__all__ = ["PersonaError", "PersonaScript"]
