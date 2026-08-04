"""The LangGraph state machine (spec section 1.2).

Topology, with model-touching nodes marked [LLM] and human pauses marked HITL:

    START
      -> capture_purpose        [LLM: P2]
      -> await_purpose          [interrupt]
      -> propose_maturity       [LLM: P3]
      -> confirm_maturity       [interrupt]                        <- HITL-1
      -> decide_gap             [code: checklist lookup]
            |- gap found      -> generate_question   [LLM: P4]
            |                    -> await_answer     [interrupt]   <- HITL-2
            |                    -> extract          [LLM: P5]
            |                    -> validate         [code: V1-V5]
            |                    -> commit_or_flag   [code]
            |                    -> decide_gap            (the cycle)
            |- no gap, flags  -> resolve_flags       [LLM: P6]
            |                    -> await_flag_resolution [interrupt] <- HITL-3
            |                    -> decide_gap
            |- complete       -> finalize            [code: write outputs]
    END

Two implementation notes about the shape.

**Why the interrupts sit in their own nodes.** The spec draws
``capture_purpose`` and ``resolve_flags`` as single nodes that both call a model
and pause for a human. LangGraph re-executes an interrupted node from the top
when it resumes, so a node that calls the model *before* interrupting would
call it twice and bill for it twice. Splitting each into an LLM node followed by
a pure interrupt node preserves the contract exactly and makes each model call
happen once. Everything before an ``interrupt()`` in this file is pure.

**Why every node deep-copies the session.** Nodes return a new object rather
than mutating a shared one, so each checkpoint is a true snapshot of the state
at that step rather than a view onto the latest mutation. The turn-by-turn
audit trail is the evaluation dataset; it has to be trustworthy.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from dtv_rea.llm import Focus, ModelPort
from dtv_rea.settings import (
    ATTEMPT_CAP,
    DIMENSION_LABELS,
    DUPLICATE_THRESHOLD,
    MATURITY_LEVELS,
    STOP_WORDS_INPUT,
    checkpoint_db,
)
from dtv_rea.state import (
    Flag,
    MaturityRecord,
    Purpose,
    Requirement,
    RequirementCandidate,
    SessionState,
)
from dtv_rea.validator import (
    Verdict,
    content_words,
    numeric_tokens,
    overlap_ratio,
    validate_batch,
    validate_candidate,
)


class GraphState(TypedDict, total=False):
    """What flows between nodes.

    ``session`` is the durable state. ``candidates`` and ``verdicts`` are
    transient hand-offs along the extract -> validate -> commit chain.
    """

    session: SessionState
    candidates: list[RequirementCandidate]
    verdicts: list[Verdict]


def is_stop(text: str | None) -> bool:
    """Q2 - has the stakeholder asked to stop?

    A typed keyword is the primary signal because the end-of-input key differs
    between platforms (Ctrl-D on macOS, Ctrl-Z then Enter on Windows). The CLI
    turns an end-of-input into the word "stop" before it gets here, so both
    routes converge on one check.
    """
    if text is None:
        return True
    return text.strip().lower() in STOP_WORDS_INPUT


def _copy(graph_state: GraphState) -> SessionState:
    return graph_state["session"].model_copy(deep=True)


# --------------------------------------------------------------------------
# Flag helpers
# --------------------------------------------------------------------------


def _add_finding_flag(
    session: SessionState,
    finding: Any,
    requirement_id: str | None,
    candidate: RequirementCandidate,
    turn_index: int | None,
) -> Flag:
    return session.add_flag(
        Flag(
            id=session.allocate_flag_id(),
            code=finding.code,
            severity=finding.severity,
            message=finding.message,
            dimension=candidate.dimension,
            requirement_id=requirement_id,
            related_requirement_id=finding.related_requirement_id,
            turn_index=turn_index,
            candidate_text=candidate.text or None,
            candidate_goal=candidate.designGoal,
        )
    )


def attach_goal(requirement: Requirement, goal: str, reference: int | None) -> None:
    """Record a stakeholder-sourced design goal on an existing requirement.

    Promoting the type is part of the job, not a nicety. DTV section 3.2 ties
    design goals to performance requirements, so a requirement that has just
    acquired a number *is* a performance requirement and must be verified by
    Test rather than Inspection. Observed in the live FDM run: Llama 3.3
    supplies goals for requirements it had labelled "functional".

    Kept in one function because the goal can arrive by two different routes -
    an extraction that targets an existing requirement, and a human answering
    a flag at HITL-3 - and the rule has to be identical on both.
    """
    requirement.designGoal = goal
    requirement.designGoal_provenance = "stated"
    requirement.stakeholder_utterance_ref = reference
    requirement.status = "complete"
    if requirement.type != "performance":
        requirement.type = "performance"
        if not requirement.verifyMethod_confirmed:
            requirement.verifyMethod = "Test"


def _link_fabrication_flags(session: SessionState) -> None:
    """Attach each open fabrication flag to the requirement it now concerns.

    A rejected candidate has no id - it never became a requirement. Once the
    human has been asked and the requirement is committed, the flag needs to
    point at it, so that the flag can be closed when a goal arrives and so that
    Q1 can stop the recovery question repeating when it never will.
    """
    for flag in session.open_flags():
        if flag.code != "fabricated_goal" or flag.requirement_id:
            continue
        if not flag.candidate_text:
            continue
        wanted = content_words(flag.candidate_text)
        for requirement in session.requirements:
            if requirement.text == flag.candidate_text or (
                overlap_ratio(wanted, content_words(requirement.text))
                >= DUPLICATE_THRESHOLD
            ):
                flag.requirement_id = requirement.id
                break


def _auto_resolve_flags(session: SessionState) -> None:
    """Close flags whose condition provably no longer holds.

    This is *re-running the check*, not the agent deciding something. Only the
    two number-shaped codes are eligible:

    * ``unverifiable_predicate`` - the requirement now carries a design goal,
      so the predicate is pinned down and V2 would no longer fire.
    * ``fabricated_goal`` - a requirement matching the rejected candidate has
      since been committed with a goal that traces to a stakeholder turn, which
      is precisely the outcome the flag was demanding.

    ``duplicate_obligation`` and ``maturity_inconsistency`` are never closed
    this way. Those are judgements, and only the human makes them.
    """
    for flag in session.open_flags():
        if flag.code == "unverifiable_predicate":
            requirement = session.requirement(flag.requirement_id or "")
            if requirement is not None and requirement.designGoal is not None:
                session.resolve_flag(
                    flag.id,
                    f"A design goal was supplied for {requirement.id}: "
                    f"\"{requirement.designGoal}\".",
                    resolved_by="condition_no_longer_holds",
                )
        elif flag.code == "fabricated_goal":
            replacement = _replacement_for(session, flag)
            if replacement is not None:
                session.resolve_flag(
                    flag.id,
                    f"The stakeholder supplied the value; {replacement.id} now "
                    f"records \"{replacement.designGoal}\" from turn "
                    f"{replacement.stakeholder_utterance_ref}.",
                    resolved_by="condition_no_longer_holds",
                )


def _replacement_for(session: SessionState, flag: Flag) -> Requirement | None:
    """Find a committed, properly sourced goal that answers a fabrication flag."""
    if flag.requirement_id:
        requirement = session.requirement(flag.requirement_id)
        if requirement is not None and requirement.designGoal is not None:
            return requirement
    if not flag.candidate_text:
        return None
    wanted = content_words(flag.candidate_text)
    for requirement in session.requirements:
        if requirement.designGoal is None:
            continue
        if requirement.text == flag.candidate_text:
            return requirement
        if overlap_ratio(wanted, content_words(requirement.text)) >= DUPLICATE_THRESHOLD:
            return requirement
    return None


def apply_flag_resolution(session: SessionState, flag: Flag, answer: str) -> None:
    """Turn a human's HITL-3 answer into a state change (spec section 1.2).

    The parse is deliberately small and deterministic. The agent never decides
    what the human meant beyond three keywords, and it never merges anything
    the human did not ask it to merge.
    """
    lowered = answer.strip().lower()
    resolution = answer.strip()

    if flag.code == "duplicate_obligation":
        merging = "merge" in lowered or "same" in lowered or "one thing" in lowered
        if merging and flag.requirement_id and flag.related_requirement_id:
            dropped = session.requirement(flag.requirement_id)
            survivor = session.requirement(flag.related_requirement_id)
            if dropped is not None and survivor is not None:
                session.requirements.remove(dropped)
                survivor.rationale = (
                    f"{survivor.rationale} Merged with {dropped.id} at the "
                    f"stakeholder's direction: \"{dropped.text}\"."
                ).strip()
                resolution = f"Merged {dropped.id} into {survivor.id}. {resolution}"
        else:
            resolution = f"Kept both as distinct obligations. {resolution}"

    elif flag.code in {"fabricated_goal", "unverifiable_predicate"}:
        # If the human's own words contain a number, it is now a legitimate
        # source - it came from a stakeholder turn. V1 still has to agree.
        target = flag.requirement_id
        if target and numeric_tokens(answer):
            candidate = RequirementCandidate(
                dimension=flag.dimension or "data_collection_integration",  # type: ignore[arg-type]
                goal_target_id=target,
                designGoal=resolution,
                designGoal_provenance="stated",
                stakeholder_utterance_ref=len(session.turns) - 1,
            )
            verdict = validate_candidate(candidate, session)
            requirement = session.requirement(target)
            if not verdict.rejected and requirement is not None:
                attach_goal(
                    requirement, resolution, candidate.stakeholder_utterance_ref
                )

    session.resolve_flag(flag.id, resolution, resolved_by="human")


# --------------------------------------------------------------------------
# Node factories
# --------------------------------------------------------------------------


def build_nodes(model: ModelPort) -> dict[str, Callable[[GraphState], GraphState]]:
    """Build every node as a closure over the model. Pure functions get none."""

    # -- capture_purpose / await_purpose --------------------------------

    def capture_purpose(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        question = model.opener(session)
        session.add_turn("agent", question, node="capture_purpose")
        session.pending_question = question
        return {"session": session}

    def await_purpose(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        answer = interrupt(
            {
                "kind": "purpose",
                "question": session.pending_question or "",
                "turn_index": len(session.turns),
            }
        )
        if is_stop(answer):
            session.stop_requested = True
            return {"session": session}
        session.add_turn("stakeholder", str(answer), node="await_purpose")
        session.purpose = Purpose(statement=str(answer).strip())
        session.pending_question = None
        session.snapshot()
        return {"session": session}

    # -- propose_maturity / confirm_maturity (HITL-1) --------------------

    def propose_maturity(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        proposal = model.propose_maturity(session)
        if session.purpose is not None:
            session.purpose.rationale = proposal.rationale
            session.purpose.user_roles = list(proposal.user_roles)
            session.purpose.context_of_use = proposal.context_of_use
        session.pending_focus = {
            "kind": "maturity",
            "level": proposal.level,
            "reasoning": proposal.reasoning,
            "description": proposal.description,
        }
        session.add_turn("agent", proposal.description, node="propose_maturity")
        return {"session": session}

    def confirm_maturity(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        proposed = session.pending_focus.get("level", "Replication")
        description = session.pending_focus.get("description", "")
        answer = interrupt(
            {
                "kind": "maturity",
                "question": description,
                "level": proposed,
                "turn_index": len(session.turns),
            }
        )
        if is_stop(answer):
            session.stop_requested = True
            return {"session": session}

        turn = session.add_turn("stakeholder", str(answer), node="confirm_maturity")
        override = _named_level(str(answer), proposed)
        session.maturity = MaturityRecord(
            value=(override or proposed),  # type: ignore[arg-type]
            agent_reasoning=session.pending_focus.get("reasoning", ""),
            description_shown_to_human=description,
            human_response="overridden" if override else "confirmed",
            overridden_to=override,  # type: ignore[arg-type]
            confirming_turn=turn.index,
        )
        session.pending_focus = {}
        session.snapshot()
        return {"session": session}

    # -- decide_gap (pure code) -----------------------------------------

    def decide_gap(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        session.pending_focus = {}

        if session.stop_requested:
            session.route = "finalize"
            return {"session": session}

        # The loop exists so that parking a dimension or a goal immediately
        # re-runs the checklist instead of costing the stakeholder a turn.
        for _ in range(len(session.coverage_status) + len(session.requirements) + 8):
            focus = _next_focus(session)
            if focus is None:
                break
            if focus.kind == "__parked__":
                continue
            session.pending_focus = focus.as_dict()
            session.route = "ask"
            return {"session": session}

        if session.open_flags():
            session.route = "resolve_flags"
        else:
            session.route = "finalize"
        return {"session": session}

    # -- generate_question / await_answer (HITL-2) -----------------------

    def generate_question(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        focus = Focus.from_dict(session.pending_focus)
        question = model.ask(session, focus)
        session.add_turn("agent", question, node="generate_question")
        session.pending_question = question
        return {"session": session}

    def await_answer(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        answer = interrupt(
            {
                "kind": "answer",
                "question": session.pending_question or "",
                "turn_index": len(session.turns),
                "focus": session.pending_focus.get("kind", "dimension"),
            }
        )
        if is_stop(answer):
            session.stop_requested = True
            return {"session": session}
        session.add_turn("stakeholder", str(answer), node="await_answer")
        session.pending_question = None
        return {"session": session}

    # -- extract / validate / commit_or_flag -----------------------------

    def extract(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        focus = Focus.from_dict(session.pending_focus)
        answer_turn = session.turns[-1]
        result = model.extract(session, answer_turn, focus)

        # Any goal the model attached without citing anything is pinned to the
        # turn being extracted. That is not a favour to the model - it makes V1
        # check the number against the answer it actually came from, instead of
        # rejecting it on a technicality and losing the real finding.
        candidates: list[RequirementCandidate] = []
        for candidate in result.requirements:
            if (
                candidate.designGoal is not None
                and candidate.stakeholder_utterance_ref is None
            ):
                candidate = candidate.model_copy(
                    update={"stakeholder_utterance_ref": answer_turn.index}
                )
            candidates.append(candidate)

        session.pending_not_applicable = list(result.not_applicable)
        session.extraction_note = result.note
        return {"session": session, "candidates": candidates}

    def validate(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        candidates = graph_state.get("candidates", [])
        verdicts = validate_batch(candidates, session)
        return {"session": session, "verdicts": verdicts}

    def commit_or_flag(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        verdicts = graph_state.get("verdicts", [])
        turn_index = session.turns[-1].index if session.turns else None

        for verdict in verdicts:
            candidate = verdict.candidate

            if verdict.rejected:
                # Reject and ask the human. Never regenerate - a regenerated
                # number is still an invented number.
                for finding in verdict.hard_findings:
                    _add_finding_flag(
                        session, finding, candidate.goal_target_id, candidate,
                        turn_index,
                    )
                continue

            if candidate.goal_target_id is not None:
                requirement = session.requirement(candidate.goal_target_id)
                if requirement is None:  # pragma: no cover - V4 caught this
                    continue
                if candidate.designGoal is not None:
                    attach_goal(
                        requirement,
                        candidate.designGoal,
                        candidate.stakeholder_utterance_ref,
                    )
                elif candidate.status == "needs_clarification":
                    # Q1: "I don't know" parks the goal and moves on.
                    requirement.status = "needs_clarification"
            else:
                requirement = Requirement(
                    id=session.allocate_requirement_id(),
                    text=candidate.text,
                    type=candidate.type,
                    dimension=candidate.dimension,
                    verifyMethod=candidate.default_verify_method(),  # type: ignore[arg-type]
                    designGoal=candidate.designGoal,
                    designGoal_provenance=(
                        "stated" if candidate.designGoal is not None else None
                    ),
                    stakeholder_utterance_ref=candidate.stakeholder_utterance_ref,
                    priority=candidate.priority,
                    rationale=candidate.rationale,
                    status=candidate.status,
                    source_turn=turn_index,
                )
                session.commit_requirement(requirement)

            for finding in verdict.soft_findings:
                _add_finding_flag(
                    session, finding, requirement.id, candidate, turn_index
                )

        for claim in session.pending_not_applicable:
            session.mark_not_applicable(
                claim.dimension, claim.stakeholder_utterance_ref
            )
        session.pending_not_applicable = []

        _link_fabrication_flags(session)
        _auto_resolve_flags(session)
        session.snapshot()
        return {"session": session, "candidates": [], "verdicts": []}

    # -- resolve_flags / await_flag_resolution (HITL-3) ------------------

    def resolve_flags(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        open_flags = session.open_flags()
        if not open_flags:  # pragma: no cover - routing guarantees one exists
            session.pending_flag_id = None
            return {"session": session}
        flag = open_flags[0]
        message = model.phrase_flag(session, flag)
        session.add_turn("agent", message, node="resolve_flags")
        session.pending_flag_id = flag.id
        session.pending_flag_message = message
        return {"session": session}

    def await_flag_resolution(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        flag_id = session.pending_flag_id
        answer = interrupt(
            {
                "kind": "flag",
                "question": session.pending_flag_message or "",
                "flag_id": flag_id or "",
                "turn_index": len(session.turns),
            }
        )
        if is_stop(answer):
            session.stop_requested = True
            return {"session": session}
        session.add_turn(
            "stakeholder", str(answer), node="await_flag_resolution"
        )
        flag = session.flag(flag_id or "")
        if flag is not None:
            apply_flag_resolution(session, flag, str(answer))
        session.pending_flag_id = None
        session.pending_flag_message = None
        session.snapshot()
        return {"session": session}

    # -- finalize --------------------------------------------------------

    def finalize(graph_state: GraphState) -> GraphState:
        session = _copy(graph_state)
        session.status = "partial" if session.stop_requested else "complete"
        for requirement in session.requirements:
            # The verifyMethod pattern is agent-proposed; confirmation is
            # batched here, at the end (spec section 4, HITL-3).
            requirement.verifyMethod_confirmed = True
        session.pending_question = None
        session.pending_focus = {}
        session.route = "done"
        session.snapshot()
        return {"session": session}

    return {
        "capture_purpose": capture_purpose,
        "await_purpose": await_purpose,
        "propose_maturity": propose_maturity,
        "confirm_maturity": confirm_maturity,
        "decide_gap": decide_gap,
        "generate_question": generate_question,
        "await_answer": await_answer,
        "extract": extract,
        "validate": validate,
        "commit_or_flag": commit_or_flag,
        "resolve_flags": resolve_flags,
        "await_flag_resolution": await_flag_resolution,
        "finalize": finalize,
    }


# --------------------------------------------------------------------------
# decide_gap's priority order - the heart of "decisions in code"
# --------------------------------------------------------------------------


def _next_focus(session: SessionState) -> Focus | None:
    """What to ask about next, or ``None`` when nothing is left to ask.

    Priority, highest first:

    1. **A rejected number.** V1 threw a goal out; the human is asked for it
       directly (P4 rule 3). This outranks everything, because the alternative
       is a document with a hole in it.
    2. **A performance requirement that silently lacks a goal.** The spec's
       rule that the number-question takes priority over a new topic.
    3. **The first uncovered dimension**, in fixed checklist order.

    Returning a focus whose ``kind`` is ``"__parked__"`` means "I just parked
    something, ask me again" - the caller loops rather than spending a turn.
    """
    # 1. fabrication recovery
    for flag in session.open_flags():
        if flag.code != "fabricated_goal":
            continue
        subject_requirement = session.requirement(flag.requirement_id or "")
        if (
            subject_requirement is not None
            and subject_requirement.status == "needs_clarification"
        ):
            # Q1. The stakeholder has already said they do not have this
            # number. Asking a second and third time is badgering, and the
            # answer will not change. The flag stays open, so HITL-3 still
            # forces a human to close it before the session can finish.
            continue
        if session.goal_attempts_exhausted(flag.id):
            continue  # hand it to HITL-3 rather than ask a fourth time
        attempt = session.record_goal_attempt(flag.id)
        subject = flag.candidate_text or flag.requirement_id or "that value"
        return Focus(
            kind="fabrication_recovery",
            description=subject,
            dimension=flag.dimension,
            requirement_id=flag.requirement_id,
            flag_id=flag.id,
            attempt=attempt,
        )

    # 2. a committed performance requirement with no goal
    requirement = session.requirement_missing_goal()
    if requirement is not None:
        if session.goal_attempts_exhausted(requirement.id):
            # Q1/Q3: park it as needing clarification rather than looping.
            requirement.status = "needs_clarification"
            return Focus(kind="__parked__", description="")
        attempt = session.record_goal_attempt(requirement.id)
        return Focus(
            kind="goal",
            description=requirement.text,
            dimension=requirement.dimension,
            requirement_id=requirement.id,
            attempt=attempt,
        )

    # 3. the first uncovered dimension
    dimension = session.next_gap()
    if dimension is not None:
        if session.attempts_exhausted(dimension):
            # Q3: park and continue.
            session.mark_parked(
                dimension,
                f"No usable answer after {ATTEMPT_CAP} attempts.",
            )
            return Focus(kind="__parked__", description="")
        attempt = session.record_attempt(dimension)
        return Focus(
            kind="dimension" if attempt == 1 else "retry_dimension",
            description=DIMENSION_LABELS.get(dimension, dimension),
            dimension=dimension,
            attempt=attempt,
        )

    return None


def _named_level(answer: str, proposed: str) -> str | None:
    """Return the level the human overrode to, or ``None`` for a confirmation."""
    lowered = answer.lower()
    for level in MATURITY_LEVELS:
        if level.lower() in lowered and level != proposed:
            return level
    return None


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def route_after_decide_gap(graph_state: GraphState) -> str:
    return graph_state["session"].route or "finalize"


def _stop_router(next_node: str) -> Callable[[GraphState], str]:
    def router(graph_state: GraphState) -> str:
        return "finalize" if graph_state["session"].stop_requested else next_node

    return router


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_graph(model: ModelPort, checkpointer: Any | None = None):
    """Compile the interview graph. ``checkpointer`` is required for interrupts."""
    nodes = build_nodes(model)
    builder: StateGraph = StateGraph(GraphState)

    for name, function in nodes.items():
        builder.add_node(name, function)

    builder.add_edge(START, "capture_purpose")
    builder.add_edge("capture_purpose", "await_purpose")
    builder.add_conditional_edges(
        "await_purpose",
        _stop_router("propose_maturity"),
        {"propose_maturity": "propose_maturity", "finalize": "finalize"},
    )
    builder.add_edge("propose_maturity", "confirm_maturity")
    builder.add_conditional_edges(
        "confirm_maturity",
        _stop_router("decide_gap"),
        {"decide_gap": "decide_gap", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "decide_gap",
        route_after_decide_gap,
        {
            "ask": "generate_question",
            "resolve_flags": "resolve_flags",
            "finalize": "finalize",
        },
    )
    builder.add_edge("generate_question", "await_answer")
    builder.add_conditional_edges(
        "await_answer",
        _stop_router("extract"),
        {"extract": "extract", "finalize": "finalize"},
    )
    builder.add_edge("extract", "validate")
    builder.add_edge("validate", "commit_or_flag")
    builder.add_edge("commit_or_flag", "decide_gap")
    builder.add_edge("resolve_flags", "await_flag_resolution")
    builder.add_conditional_edges(
        "await_flag_resolution",
        _stop_router("decide_gap"),
        {"decide_gap": "decide_gap", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


#: Types this project stores in checkpoints. Declaring them explicitly keeps
#: LangGraph from falling back on its "unregistered type" path, which currently
#: warns and is documented to start refusing in a future release.
_CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("dtv_rea.state", "SessionState"),
    ("dtv_rea.state", "Requirement"),
    ("dtv_rea.state", "RequirementCandidate"),
    ("dtv_rea.state", "NotApplicableClaim"),
    ("dtv_rea.state", "Flag"),
    ("dtv_rea.state", "Turn"),
    ("dtv_rea.state", "Purpose"),
    ("dtv_rea.state", "MaturityRecord"),
    ("dtv_rea.state", "Snapshot"),
    ("dtv_rea.validator", "Finding"),
    ("dtv_rea.validator", "Verdict"),
)


def open_checkpointer(path: Path | None = None) -> tuple[SqliteSaver, sqlite3.Connection]:
    """Open the durable checkpointer at ``./runs/checkpoints.db``.

    Returns the saver and its connection so the caller can close it. The
    connection allows cross-thread use because LangGraph may run nodes on a
    worker thread; the interview is single-threaded regardless.
    """
    database = path or checkpoint_db()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
    return SqliteSaver(connection, serde=serde), connection


__all__ = [
    "GraphState",
    "apply_flag_resolution",
    "build_graph",
    "build_nodes",
    "is_stop",
    "open_checkpointer",
]
