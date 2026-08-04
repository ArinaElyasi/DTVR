"""Phase 4 - the graph, end to end, on the stub model.

Every test in this file runs the real LangGraph state machine, the real
validator and the real checkpointer, with no network and no API key. This is
the proof that the deterministic core works before any API call exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from dtv_rea.graph import build_graph, is_stop, open_checkpointer
from dtv_rea.llm import StubModel
from dtv_rea.persona import PersonaScript
from dtv_rea.runner import persona_answerer, run_session
from dtv_rea.settings import DIMENSIONS
from dtv_rea.state import SessionState
from dtv_rea.validator import audit_committed_goals

from tests.conftest import make_requirement


@pytest.fixture
def checkpointer(tmp_path: Path) -> Iterator[object]:
    saver, connection = open_checkpointer(tmp_path / "checkpoints.db")
    try:
        yield saver
    finally:
        connection.close()


def run_persona(name: str, checkpointer: object, session_id: str = "t") -> SessionState:
    script = PersonaScript.by_name(name)
    graph = build_graph(StubModel(script), checkpointer=checkpointer)
    return run_session(graph, session_id, persona_answerer(script))


@pytest.fixture
def fdm(checkpointer: object) -> SessionState:
    return run_persona("fdm_stakeholder", checkpointer, "fdm")


# --------------------------------------------------------------------------
# The stop signal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["stop", "STOP", "  Stop  ", "quit", "exit", ":q", "/stop", None]
)
def test_stop_keywords_are_recognised_on_every_platform(text: str | None) -> None:
    assert is_stop(text)


@pytest.mark.parametrize("text", ["stop the print when it clogs", "no", ""])
def test_ordinary_answers_are_not_mistaken_for_stopping(text: str) -> None:
    assert not is_stop(text)


# --------------------------------------------------------------------------
# The FDM run - the headline case
# --------------------------------------------------------------------------


def test_the_interview_completes(fdm: SessionState) -> None:
    assert fdm.status == "complete"
    assert fdm.is_complete()


def test_all_twelve_requirements_are_committed(fdm: SessionState) -> None:
    assert len(fdm.requirements) == 12


def test_every_dimension_is_covered(fdm: SessionState) -> None:
    for dimension in DIMENSIONS:
        assert fdm.coverage_status[dimension] == "covered", dimension


def test_the_fabricated_goal_was_rejected_and_never_committed(
    fdm: SessionState,
) -> None:
    """The seeded 95% target must not appear anywhere in the output."""
    assert not any(
        requirement.designGoal and "95" in requirement.designGoal
        for requirement in fdm.requirements
    )
    fabrication_flags = [f for f in fdm.flags if f.code == "fabricated_goal"]
    assert len(fabrication_flags) == 1
    assert fabrication_flags[0].severity == "hard"


def test_fabricated_goal_rate_is_zero_on_an_independent_audit(
    fdm: SessionState,
) -> None:
    """The headline metric, re-derived from the transcript alone.

    This deliberately does not consult the validator's own verdicts. A metric
    that trusts the component it is measuring measures nothing.
    """
    assert audit_committed_goals(fdm) == []


def test_the_rejected_number_was_replaced_by_one_the_human_gave(
    fdm: SessionState,
) -> None:
    clog = next(
        requirement
        for requirement in fdm.requirements
        if "detect clogs" in requirement.text
    )
    assert clog.designGoal == "Accuracy >=90%"
    assert clog.designGoal_provenance == "stated"
    source = fdm.turn(clog.stakeholder_utterance_ref)
    assert source is not None and source.role == "stakeholder"
    assert "90" in source.text


def test_both_published_paper_defects_are_flagged(fdm: SessionState) -> None:
    codes = {flag.code for flag in fdm.flags}
    assert "unverifiable_predicate" in codes   # UR1.11 "minimize"
    assert "duplicate_obligation" in codes     # UR1.2 / UR1.7


def test_the_minimize_requirement_is_the_one_flagged_as_unverifiable(
    fdm: SessionState,
) -> None:
    flag = next(f for f in fdm.flags if f.code == "unverifiable_predicate")
    requirement = fdm.requirement(flag.requirement_id or "")
    assert requirement is not None
    assert "minimize data loss" in requirement.text


def test_the_duplicate_flag_names_the_continuous_temperature_pair(
    fdm: SessionState,
) -> None:
    duplicates = [f for f in fdm.flags if f.code == "duplicate_obligation"]
    pairs = []
    for flag in duplicates:
        left = fdm.requirement(flag.requirement_id or "")
        right = fdm.requirement(flag.related_requirement_id or "")
        if left and right:
            pairs.append((left.text, right.text))
    assert any(
        "bed and nozzle temperature" in left
        and "temperature, position, and vibration" in right
        for left, right in pairs
    )


def test_every_flag_is_resolved_before_the_session_can_finish(
    fdm: SessionState,
) -> None:
    assert fdm.open_flags() == []
    assert all(flag.status == "resolved" for flag in fdm.flags)


def test_duplicates_are_only_ever_resolved_by_a_human(fdm: SessionState) -> None:
    """The agent must never auto-merge. Spec section 1.2, ``resolve_flags``."""
    for flag in fdm.flags:
        if flag.code == "duplicate_obligation":
            assert flag.resolved_by == "human"


def test_the_vague_predicate_flag_closed_because_a_number_arrived(
    fdm: SessionState,
) -> None:
    flag = next(f for f in fdm.flags if f.code == "unverifiable_predicate")
    assert flag.resolved_by == "condition_no_longer_holds"
    requirement = fdm.requirement(flag.requirement_id or "")
    assert requirement is not None and requirement.designGoal == "data loss <=10%"


def test_every_committed_goal_quotes_a_real_stakeholder_turn(
    fdm: SessionState,
) -> None:
    for requirement in fdm.requirements:
        if requirement.designGoal is None:
            continue
        assert requirement.designGoal_provenance == "stated"
        turn = fdm.turn(requirement.stakeholder_utterance_ref)
        assert turn is not None, requirement.id
        assert turn.role == "stakeholder", requirement.id


def test_all_six_published_design_goals_are_recovered(fdm: SessionState) -> None:
    goals = {
        requirement.designGoal
        for requirement in fdm.requirements
        if requirement.designGoal
    }
    assert goals == {
        "Accuracy of +/- 2.5 C from target",
        "Accuracy of +/- 10% from target",
        "Accuracy >=90%",
        "data loss <=10%",
        "100% of data sent is received by the DT",
    }


def test_hitl_1_recorded_what_the_human_was_actually_shown(
    fdm: SessionState,
) -> None:
    """A bare "OK" to an unexplained label would be rubber-stamping."""
    maturity = fdm.maturity
    assert maturity is not None
    assert maturity.value == "Replication"
    assert maturity.human_response == "confirmed"
    assert maturity.provenance == "agent_proposed_human_confirmed"
    assert len(maturity.description_shown_to_human) > 80
    assert "4R" not in maturity.description_shown_to_human


def test_snapshots_accumulate_one_row_per_turn(fdm: SessionState) -> None:
    assert len(fdm.snapshots) >= len(fdm.stakeholder_turns())
    counts = [row.n_requirements for row in fdm.snapshots]
    assert counts == sorted(counts)     # requirements only ever accumulate
    assert counts[-1] == 12


def test_elicitation_stays_within_the_target_turn_budget(
    fdm: SessionState,
) -> None:
    """Spec section 5 target: <= 12 stakeholder turns on the FDM persona."""
    assert len(fdm.stakeholder_turns()) <= 12


def test_verify_methods_follow_the_pattern_and_are_confirmed(
    fdm: SessionState,
) -> None:
    for requirement in fdm.requirements:
        expected = "Test" if requirement.type == "performance" else "Inspection"
        assert requirement.verifyMethod == expected
        assert requirement.verifyMethod_confirmed
        assert requirement.verifyMethod_provenance == "agent_proposed_human_confirmed"


def test_a_goal_arriving_later_promotes_the_requirement_to_performance() -> None:
    """DTV 3.2 ties goals to performance requirements, so the type follows.

    Observed live: Llama 3.3 supplies a goal for a requirement it had labelled
    functional. Left alone that requirement keeps verifyMethod "Inspection",
    which is the wrong way to check a number.
    """
    from dtv_rea.graph import attach_goal

    requirement = make_requirement(
        "UR1.9",
        "The DT shall detect clogs using vibration data",
        dimension="intelligence_layer",
        requirement_type="functional",
    )
    assert requirement.verifyMethod == "Inspection"

    attach_goal(requirement, "Accuracy >=90%", reference=11)

    assert requirement.type == "performance"
    assert requirement.verifyMethod == "Test"
    assert requirement.designGoal_provenance == "stated"
    assert requirement.stakeholder_utterance_ref == 11
    assert requirement.status == "complete"


def test_a_confirmed_verify_method_is_not_overwritten() -> None:
    """The human's confirmation outranks the agent's pattern."""
    from dtv_rea.graph import attach_goal

    requirement = make_requirement(
        "UR1.9", "The DT shall detect clogs", requirement_type="functional"
    )
    requirement.verifyMethod = "Demonstration"
    requirement.verifyMethod_confirmed = True

    attach_goal(requirement, "Accuracy >=90%", reference=3)

    assert requirement.type == "performance"
    assert requirement.verifyMethod == "Demonstration"


def test_every_requirement_with_a_goal_is_a_performance_requirement(
    fdm: SessionState,
) -> None:
    for requirement in fdm.requirements:
        if requirement.designGoal is not None:
            assert requirement.type == "performance", requirement.id
            assert requirement.verifyMethod == "Test", requirement.id


def test_the_run_is_reproducible(checkpointer: object) -> None:
    first = run_persona("fdm_stakeholder", checkpointer, "repeat-a")
    second = run_persona("fdm_stakeholder", checkpointer, "repeat-b")
    assert [r.text for r in first.requirements] == [
        r.text for r in second.requirements
    ]
    assert [f.code for f in first.flags] == [f.code for f in second.flags]


# --------------------------------------------------------------------------
# Durability
# --------------------------------------------------------------------------


def test_a_paused_interview_survives_a_process_restart(tmp_path: Path) -> None:
    """A stakeholder can leave mid-interview and come back."""
    database = tmp_path / "checkpoints.db"
    script = PersonaScript.by_name("fdm_stakeholder")

    # First "process": answer the opener, then walk away.
    saver, connection = open_checkpointer(database)
    graph = build_graph(StubModel(script), checkpointer=saver)
    config = {"configurable": {"thread_id": "resumable"}}
    from dtv_rea.state import new_session
    from langgraph.types import Command

    result = graph.invoke({"session": new_session("resumable")}, config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume=script.data["purpose_answer"]), config)
    connection.close()

    # Second "process": nothing in memory, everything from the database.
    saver, connection = open_checkpointer(database)
    try:
        graph = build_graph(StubModel(script), checkpointer=saver)
        snapshot = graph.get_state(config)
        assert snapshot.next  # the interview is genuinely mid-flight
        assert snapshot.values["session"].purpose is not None
        assert (
            "bank of FDM printers"
            in snapshot.values["session"].purpose.statement
        )
    finally:
        connection.close()
