"""The typed session state - the agent's only memory (spec section 1.4).

One running :class:`SessionState` object *is* the agent's memory. Requirements
are never re-derived from the transcript; they are committed once and then
carried forward.

The load-bearing schema decision lives in :class:`Requirement`::

    designGoal_provenance: Literal["stated"] | None

There is deliberately no ``"agent_derived"`` option. A design goal either
traces to something the stakeholder actually said, or it does not exist. The
research claim is enforced by the type system before the validator ever runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from dtv_rea.settings import ATTEMPT_CAP, DIMENSIONS, MATURITY_LEVELS

Dimension = Literal[
    "data_collection_integration",
    "virtual_environment",
    "intelligence_layer",
    "automation_feedback",
]

CoverageStatus = Literal["covered", "uncovered", "not_applicable"]

MaturityLevel = Literal["Representation", "Replication", "Reality", "Relational"]

FlagCode = Literal[
    "fabricated_goal",
    "unverifiable_predicate",
    "duplicate_obligation",
    "orphan_goal",
    "maturity_inconsistency",
]

#: A dimension parked by the attempt cap records this instead of a real
#: confirming stakeholder turn index (spec section 1.4).
PARKED_CONFIRMATION: int = -1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Leaf models
# --------------------------------------------------------------------------


class Turn(BaseModel):
    """One conversation turn. ``index`` is the citation target used by V1."""

    index: int
    role: Literal["agent", "stakeholder"]
    text: str
    node: str | None = None
    timestamp: str = Field(default_factory=_now)


class Purpose(BaseModel):
    """What the twin is for - captured once, at the top of the interview."""

    statement: str
    rationale: str = ""
    user_roles: list[str] = Field(default_factory=list)
    context_of_use: str = ""


class MaturityRecord(BaseModel):
    """HITL-1's audit record (spec section 1.2, ``confirm_maturity``).

    ``description_shown_to_human`` exists so evaluation can detect
    rubber-stamping: a bare "OK" to an unexplained label is not a confirmation.
    """

    value: MaturityLevel
    provenance: Literal["agent_proposed_human_confirmed"] = (
        "agent_proposed_human_confirmed"
    )
    agent_reasoning: str
    description_shown_to_human: str
    human_response: Literal["confirmed", "overridden"]
    overridden_to: MaturityLevel | None = None
    confirming_turn: int | None = None


class Requirement(BaseModel):
    """A committed requirement. ``UR1.<n>``, sequential, never renumbered."""

    id: str
    text: str
    type: Literal["functional", "performance"]
    dimension: Dimension
    verifyMethod: Literal["Inspection", "Test", "Analysis", "Demonstration"]
    verifyMethod_provenance: Literal["agent_proposed_human_confirmed"] = (
        "agent_proposed_human_confirmed"
    )
    verifyMethod_confirmed: bool = False
    designGoal: str | None = None
    # No "agent_derived" member exists. This is the research claim, encoded in
    # the schema rather than asked for in a prompt.
    designGoal_provenance: Literal["stated"] | None = None
    stakeholder_utterance_ref: int | None = None
    priority: Literal["must", "should", "could"] = "must"
    rationale: str = ""
    status: Literal["complete", "needs_clarification"] = "complete"
    source_turn: int | None = None

    @model_validator(mode="after")
    def _goal_and_provenance_agree(self) -> "Requirement":
        if self.designGoal is None and self.designGoal_provenance is not None:
            raise ValueError(
                "designGoal_provenance must be None when there is no designGoal"
            )
        if self.designGoal is not None and self.designGoal_provenance != "stated":
            raise ValueError(
                "a designGoal must carry designGoal_provenance='stated'; there "
                "is no other legitimate provenance for a number"
            )
        return self


class RequirementCandidate(BaseModel):
    """What ``extract`` produces, before the validator has had a say.

    Two shapes are legal:

    * a **new requirement** - ``goal_target_id`` is ``None``, and the candidate
      is committed with a freshly allocated ``UR1.<n>`` id;
    * a **goal update** - ``goal_target_id`` names an already-committed
      requirement, and the candidate supplies the number the stakeholder has
      just given for it (the fabrication-recovery and "I don't know" paths).
    """

    text: str = ""
    type: Literal["functional", "performance"] = "functional"
    dimension: Dimension
    verifyMethod: Literal["Inspection", "Test", "Analysis", "Demonstration"] | None = None
    designGoal: str | None = None
    designGoal_provenance: Literal["stated"] | None = None
    stakeholder_utterance_ref: int | None = None
    priority: Literal["must", "should", "could"] = "must"
    rationale: str = ""
    status: Literal["complete", "needs_clarification"] = "complete"
    goal_target_id: str | None = None

    def default_verify_method(self) -> str:
        """Spec section 1.4: functional to Inspection, performance to Test."""
        if self.verifyMethod is not None:
            return self.verifyMethod
        return "Test" if self.type == "performance" else "Inspection"


class NotApplicableClaim(BaseModel):
    """A stakeholder saying a whole dimension does not apply.

    Only the stakeholder can produce one of these. The maturity level never
    sets ``not_applicable``: the FDM case is Replication yet still pauses the
    machine, so levels do not partition dimensions.
    """

    dimension: Dimension
    stakeholder_utterance_ref: int
    quote: str = ""


class Flag(BaseModel):
    """A validator finding.

    ``severity`` is contractual. ``hard`` blocked the commit; ``flag`` allowed
    it but blocks *termination* until a human resolves the finding.
    """

    id: str
    code: FlagCode
    severity: Literal["hard", "flag"]
    message: str
    dimension: Dimension | None = None
    requirement_id: str | None = None
    related_requirement_id: str | None = None
    turn_index: int | None = None
    candidate_text: str | None = None
    candidate_goal: str | None = None
    status: Literal["open", "resolved"] = "open"
    resolution: str | None = None
    resolved_by: Literal["human", "condition_no_longer_holds"] | None = None


class Snapshot(BaseModel):
    """One row of the evaluation dataset. Written after every turn."""

    turn: int
    n_requirements: int
    n_open_flags: int
    coverage: dict[str, str]


class ExtractionResult(BaseModel):
    """The full payload ``extract`` returns for one stakeholder answer."""

    requirements: list[RequirementCandidate] = Field(default_factory=list)
    not_applicable: list[NotApplicableClaim] = Field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------


def _default_coverage() -> dict[str, str]:
    return {dimension: "uncovered" for dimension in DIMENSIONS}


class SessionState(BaseModel):
    """Everything the agent knows. Serialised whole into ``session.json``."""

    session_id: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    status: Literal["in_progress", "complete", "partial"] = "in_progress"

    purpose: Purpose | None = None
    maturity: MaturityRecord | None = None

    turns: list[Turn] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    snapshots: list[Snapshot] = Field(default_factory=list)

    coverage_status: dict[str, str] = Field(default_factory=_default_coverage)
    not_applicable_confirmations: dict[str, int] = Field(default_factory=dict)
    parked_reasons: dict[str, str] = Field(default_factory=dict)

    #: Times each dimension has been asked about (Q3 attempt cap).
    attempts: dict[str, int] = Field(default_factory=dict)
    #: Times a missing design goal has been asked about, keyed by requirement id.
    goal_attempts: dict[str, int] = Field(default_factory=dict)

    #: Set when the stakeholder types a stop keyword or closes the input (Q2).
    stop_requested: bool = False

    # Scratch fields the graph nodes hand to each other. Persisted so a
    # resumed session picks up mid-question.
    pending_question: str | None = None
    pending_focus: dict[str, str] = Field(default_factory=dict)
    pending_flag_id: str | None = None
    pending_flag_message: str | None = None
    pending_not_applicable: list[NotApplicableClaim] = Field(default_factory=list)
    extraction_note: str = ""
    route: str | None = None
    next_requirement_number: int = 1
    next_flag_number: int = 1

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def add_turn(self, role: str, text: str, node: str | None = None) -> Turn:
        """Append a turn and return it. The index is its citation handle."""
        turn = Turn(index=len(self.turns), role=role, text=text, node=node)
        self.turns.append(turn)
        self.updated_at = _now()
        return turn

    def turn(self, index: int | None) -> Turn | None:
        """Return the turn at ``index``, or ``None`` if there is no such turn."""
        if index is None or index < 0 or index >= len(self.turns):
            return None
        return self.turns[index]

    def stakeholder_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "stakeholder"]

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def allocate_requirement_id(self) -> str:
        new_id = f"UR1.{self.next_requirement_number}"
        self.next_requirement_number += 1
        return new_id

    def commit_requirement(self, requirement: Requirement) -> Requirement:
        """Commit a requirement and mark its dimension covered."""
        self.requirements.append(requirement)
        if self.coverage_status.get(requirement.dimension) == "uncovered":
            self.coverage_status[requirement.dimension] = "covered"
        self.updated_at = _now()
        return requirement

    def requirement(self, requirement_id: str) -> Requirement | None:
        for requirement in self.requirements:
            if requirement.id == requirement_id:
                return requirement
        return None

    def requirements_in(self, dimension: str) -> list[Requirement]:
        return [r for r in self.requirements if r.dimension == dimension]

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def mark_not_applicable(self, dimension: str, confirming_turn: int) -> None:
        """Record a stakeholder-confirmed ``not_applicable`` dimension.

        ``confirming_turn`` must be the index of a real stakeholder turn. Use
        :meth:`mark_parked` for the attempt-cap case instead of passing -1.
        """
        self.coverage_status[dimension] = "not_applicable"
        self.not_applicable_confirmations[dimension] = confirming_turn
        self.updated_at = _now()

    def mark_parked(self, dimension: str, reason: str) -> None:
        """Q3: park a dimension the stakeholder would not engage with.

        Parked dimensions record confirmation index ``-1`` so that a parked
        dimension can never be mistaken for one the stakeholder ruled out.
        """
        self.coverage_status[dimension] = "not_applicable"
        self.not_applicable_confirmations[dimension] = PARKED_CONFIRMATION
        self.parked_reasons[dimension] = reason
        self.updated_at = _now()

    def is_parked(self, dimension: str) -> bool:
        return (
            self.coverage_status.get(dimension) == "not_applicable"
            and self.not_applicable_confirmations.get(dimension)
            == PARKED_CONFIRMATION
        )

    def next_gap(self) -> str | None:
        """First dimension still ``uncovered``, in the fixed checklist order."""
        for dimension in DIMENSIONS:
            if self.coverage_status.get(dimension, "uncovered") == "uncovered":
                return dimension
        return None

    def record_attempt(self, dimension: str) -> int:
        self.attempts[dimension] = self.attempts.get(dimension, 0) + 1
        return self.attempts[dimension]

    def attempts_exhausted(self, dimension: str) -> bool:
        """Q3: true once the dimension has already been asked about N times.

        Checked *before* asking, so the counter never exceeds the cap and the
        recorded number is exactly how many questions the stakeholder saw.
        """
        return self.attempts.get(dimension, 0) >= ATTEMPT_CAP

    # ------------------------------------------------------------------
    # Design-goal gaps
    # ------------------------------------------------------------------

    def requirement_missing_goal(self) -> Requirement | None:
        """First committed performance requirement that *silently* lacks a goal.

        A requirement parked as ``needs_clarification`` is acceptable and is
        skipped here - that is the Q1 "I don't know" outcome, and it does not
        block termination.
        """
        for requirement in self.requirements:
            if (
                requirement.type == "performance"
                and requirement.designGoal is None
                and requirement.status == "complete"
            ):
                return requirement
        return None

    def record_goal_attempt(self, requirement_id: str) -> int:
        self.goal_attempts[requirement_id] = (
            self.goal_attempts.get(requirement_id, 0) + 1
        )
        return self.goal_attempts[requirement_id]

    def goal_attempts_exhausted(self, requirement_id: str) -> bool:
        return self.goal_attempts.get(requirement_id, 0) >= ATTEMPT_CAP

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------

    def allocate_flag_id(self) -> str:
        new_id = f"F{self.next_flag_number}"
        self.next_flag_number += 1
        return new_id

    def add_flag(self, flag: Flag) -> Flag:
        self.flags.append(flag)
        self.updated_at = _now()
        return flag

    def open_flags(self) -> list[Flag]:
        return [flag for flag in self.flags if flag.status == "open"]

    def flag(self, flag_id: str) -> Flag | None:
        for flag in self.flags:
            if flag.id == flag_id:
                return flag
        return None

    def resolve_flag(
        self,
        flag_id: str,
        resolution: str,
        resolved_by: str = "human",
    ) -> Flag | None:
        flag = self.flag(flag_id)
        if flag is None:
            return None
        flag.status = "resolved"
        flag.resolution = resolution
        flag.resolved_by = resolved_by  # type: ignore[assignment]
        self.updated_at = _now()
        return flag

    # ------------------------------------------------------------------
    # Termination and audit
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """Spec section 1.4 termination rule - all three clauses must hold.

        Every dimension is covered, ruled out by the stakeholder, or parked;
        **and** no committed performance requirement silently lacks a goal;
        **and** no flags are open.
        """
        if any(
            self.coverage_status.get(dimension, "uncovered") == "uncovered"
            for dimension in DIMENSIONS
        ):
            return False
        if self.requirement_missing_goal() is not None:
            return False
        if self.open_flags():
            return False
        return True

    def snapshot(self) -> Snapshot:
        """Append a snapshot row. This log *is* the evaluation dataset."""
        row = Snapshot(
            turn=len(self.turns),
            n_requirements=len(self.requirements),
            n_open_flags=len(self.open_flags()),
            coverage=dict(self.coverage_status),
        )
        self.snapshots.append(row)
        self.updated_at = _now()
        return row


def new_session(session_id: str) -> SessionState:
    """Create an empty session with every dimension uncovered."""
    return SessionState(session_id=session_id)


__all__ = [
    "ATTEMPT_CAP",
    "DIMENSIONS",
    "MATURITY_LEVELS",
    "PARKED_CONFIRMATION",
    "CoverageStatus",
    "Dimension",
    "ExtractionResult",
    "Flag",
    "FlagCode",
    "MaturityLevel",
    "MaturityRecord",
    "NotApplicableClaim",
    "Purpose",
    "Requirement",
    "RequirementCandidate",
    "SessionState",
    "Snapshot",
    "Turn",
    "new_session",
]
