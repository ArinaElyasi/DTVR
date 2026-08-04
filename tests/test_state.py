"""Phase 1 - the state model. Pure Python, no network, no API key."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dtv_rea.settings import ATTEMPT_CAP, DIMENSIONS
from dtv_rea.state import (
    PARKED_CONFIRMATION,
    Flag,
    Requirement,
    SessionState,
    new_session,
)

from tests.conftest import make_requirement


# --------------------------------------------------------------------------
# The schema-level research claim
# --------------------------------------------------------------------------


def test_design_goal_provenance_has_no_agent_derived_member() -> None:
    """A number can only be "stated". The schema admits no other origin."""
    with pytest.raises(ValidationError):
        Requirement(
            id="UR1.1",
            text="The DT shall hold the nozzle temperature.",
            type="performance",
            dimension="data_collection_integration",
            verifyMethod="Test",
            designGoal="Accuracy of +/- 2.5 C from target",
            designGoal_provenance="agent_derived",  # type: ignore[arg-type]
            stakeholder_utterance_ref=3,
        )


def test_goal_without_provenance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Requirement(
            id="UR1.1",
            text="The DT shall hold the nozzle temperature.",
            type="performance",
            dimension="data_collection_integration",
            verifyMethod="Test",
            designGoal="Accuracy of +/- 2.5 C from target",
            designGoal_provenance=None,
            stakeholder_utterance_ref=3,
        )


def test_provenance_without_goal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Requirement(
            id="UR1.1",
            text="The DT shall collect the electric current.",
            type="functional",
            dimension="data_collection_integration",
            verifyMethod="Inspection",
            designGoal=None,
            designGoal_provenance="stated",
        )


# --------------------------------------------------------------------------
# Checklist ordering
# --------------------------------------------------------------------------


def test_next_gap_follows_the_fixed_checklist_order(session: SessionState) -> None:
    assert session.next_gap() == "data_collection_integration"

    session.coverage_status["data_collection_integration"] = "covered"
    assert session.next_gap() == "virtual_environment"

    session.coverage_status["virtual_environment"] = "not_applicable"
    assert session.next_gap() == "intelligence_layer"

    session.coverage_status["intelligence_layer"] = "covered"
    assert session.next_gap() == "automation_feedback"

    session.coverage_status["automation_feedback"] = "covered"
    assert session.next_gap() is None


def test_next_gap_ignores_the_order_requirements_were_committed_in(
    session: SessionState,
) -> None:
    """Committing out of order does not reorder the checklist."""
    session.commit_requirement(
        make_requirement("UR1.1", "The DT shall pause on clogs.", "automation_feedback")
    )
    assert session.next_gap() == "data_collection_integration"


def test_a_fresh_session_has_every_dimension_uncovered(session: SessionState) -> None:
    assert set(session.coverage_status) == set(DIMENSIONS)
    assert set(session.coverage_status.values()) == {"uncovered"}


# --------------------------------------------------------------------------
# The three coverage states, plus parking
# --------------------------------------------------------------------------


def test_committing_a_requirement_covers_its_dimension(session: SessionState) -> None:
    session.commit_requirement(
        make_requirement("UR1.1", "The DT shall collect the electric current.")
    )
    assert session.coverage_status["data_collection_integration"] == "covered"


def test_not_applicable_records_a_confirming_stakeholder_turn(
    session: SessionState,
) -> None:
    session.add_turn("agent", "Does the twin need to decide anything itself?")
    turn = session.add_turn("stakeholder", "No, we do not want any of that.")

    session.mark_not_applicable("intelligence_layer", turn.index)

    assert session.coverage_status["intelligence_layer"] == "not_applicable"
    assert session.not_applicable_confirmations["intelligence_layer"] == turn.index
    assert not session.is_parked("intelligence_layer")


def test_parking_is_distinguishable_from_a_stakeholder_ruling_it_out(
    session: SessionState,
) -> None:
    """Q3. A parked dimension records -1, never a real turn index."""
    session.mark_parked("intelligence_layer", "no usable answer after 3 attempts")

    assert session.coverage_status["intelligence_layer"] == "not_applicable"
    assert (
        session.not_applicable_confirmations["intelligence_layer"]
        == PARKED_CONFIRMATION
    )
    assert session.is_parked("intelligence_layer")
    assert session.parked_reasons["intelligence_layer"]


def test_the_attempt_cap_allows_exactly_n_questions(session: SessionState) -> None:
    """Q3, N=3. Checked before asking, so the stakeholder sees exactly N."""
    for _ in range(ATTEMPT_CAP - 1):
        session.record_attempt("virtual_environment")
    assert not session.attempts_exhausted("virtual_environment")

    session.record_attempt("virtual_environment")
    assert session.attempts_exhausted("virtual_environment")
    assert session.attempts["virtual_environment"] == ATTEMPT_CAP


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------


def _cover_everything(session: SessionState) -> None:
    for dimension in DIMENSIONS:
        session.coverage_status[dimension] = "covered"


def test_uncovered_dimension_blocks_completion(session: SessionState) -> None:
    _cover_everything(session)
    session.coverage_status["automation_feedback"] = "uncovered"
    assert not session.is_complete()


def test_all_dimensions_resolved_completes(session: SessionState) -> None:
    _cover_everything(session)
    assert session.is_complete()


def test_parked_and_not_applicable_dimensions_still_allow_completion(
    session: SessionState,
) -> None:
    _cover_everything(session)
    session.mark_parked("virtual_environment", "capped")
    session.mark_not_applicable("intelligence_layer", 4)
    assert session.is_complete()


def test_an_open_flag_blocks_completion(session: SessionState) -> None:
    _cover_everything(session)
    session.add_flag(
        Flag(
            id="F1",
            code="duplicate_obligation",
            severity="flag",
            message="UR1.2 and UR1.7 look like the same obligation.",
        )
    )
    assert not session.is_complete()


def test_resolving_the_flag_unblocks_completion(session: SessionState) -> None:
    _cover_everything(session)
    session.add_flag(
        Flag(
            id="F1",
            code="duplicate_obligation",
            severity="flag",
            message="UR1.2 and UR1.7 look like the same obligation.",
        )
    )
    session.resolve_flag("F1", "Distinct - keep both.")
    assert session.is_complete()


def test_a_performance_requirement_silently_missing_a_goal_blocks_completion(
    session: SessionState,
) -> None:
    _cover_everything(session)
    session.commit_requirement(
        make_requirement(
            "UR1.10",
            "The DT shall detect operation status using electric current data.",
            dimension="intelligence_layer",
            requirement_type="performance",
        )
    )
    assert session.requirement_missing_goal() is not None
    assert not session.is_complete()


def test_a_goal_parked_as_needs_clarification_does_not_block_completion(
    session: SessionState,
) -> None:
    """Q1. "I don't know" parks the goal; it must not deadlock the interview."""
    _cover_everything(session)
    session.commit_requirement(
        make_requirement(
            "UR1.10",
            "The DT shall detect operation status using electric current data.",
            dimension="intelligence_layer",
            requirement_type="performance",
            status="needs_clarification",
        )
    )
    assert session.requirement_missing_goal() is None
    assert session.is_complete()


def test_a_functional_requirement_without_a_goal_never_blocks_completion(
    session: SessionState,
) -> None:
    _cover_everything(session)
    session.commit_requirement(
        make_requirement("UR1.3", "All data elements shall be retained.")
    )
    assert session.is_complete()


# --------------------------------------------------------------------------
# Transcript, ids and snapshots
# --------------------------------------------------------------------------


def test_turn_indices_are_the_citation_handles(session: SessionState) -> None:
    first = session.add_turn("agent", "What is the twin for?")
    second = session.add_turn("stakeholder", "Catching clogs during a print.")

    assert (first.index, second.index) == (0, 1)
    assert session.turn(1) is second
    assert session.turn(99) is None
    assert session.turn(None) is None
    assert [t.text for t in session.stakeholder_turns()] == [
        "Catching clogs during a print."
    ]


def test_requirement_ids_are_sequential_and_never_reused(
    session: SessionState,
) -> None:
    assert [session.allocate_requirement_id() for _ in range(3)] == [
        "UR1.1",
        "UR1.2",
        "UR1.3",
    ]


def test_snapshot_records_the_row_the_evaluation_reads(session: SessionState) -> None:
    session.add_turn("agent", "Question?")
    session.add_turn("stakeholder", "Answer.")
    session.commit_requirement(
        make_requirement("UR1.1", "The DT shall collect the electric current.")
    )
    session.add_flag(
        Flag(id="F1", code="unverifiable_predicate", severity="flag", message="vague")
    )

    row = session.snapshot()

    assert row.turn == 2
    assert row.n_requirements == 1
    assert row.n_open_flags == 1
    assert row.coverage["data_collection_integration"] == "covered"
    assert session.snapshots == [row]


def test_state_round_trips_through_json(session: SessionState) -> None:
    """The checkpointer and session.json both depend on this."""
    session.add_turn("stakeholder", "Temperature to within 2.5 degrees.")
    session.commit_requirement(
        make_requirement(
            "UR1.7",
            "The DT shall collect bed and nozzle temperature continuously.",
            requirement_type="performance",
            design_goal="Accuracy of +/- 2.5 C from target",
            ref=0,
        )
    )

    restored = SessionState.model_validate_json(session.model_dump_json())

    assert restored.requirements[0].designGoal_provenance == "stated"
    assert restored.requirements[0].stakeholder_utterance_ref == 0
    assert restored.coverage_status == session.coverage_status


def test_new_session_helper(session: SessionState) -> None:
    assert new_session("abc").session_id == "abc"
