"""Phase 2 - V1 to V5. Pure Python, zero network, no API key.

The two mandatory cases from the published FDM paper are here verbatim:
UR1.11's unverifiable "minimize", and the UR1.2 / UR1.7 duplicate obligation.
"""

from __future__ import annotations

import pytest

from dtv_rea.settings import DUPLICATE_MIN_SHARED_WORDS, DUPLICATE_THRESHOLD
from dtv_rea.state import RequirementCandidate, SessionState
from dtv_rea.validator import (
    content_words,
    light_stem,
    numeric_tokens,
    overlap_ratio,
    validate_batch,
    validate_candidate,
    vague_predicate,
)

from tests.conftest import make_requirement


def candidate(**overrides: object) -> RequirementCandidate:
    payload: dict[str, object] = {
        "text": "The DT shall collect the bed temperature continuously.",
        "type": "performance",
        "dimension": "data_collection_integration",
    }
    payload.update(overrides)
    return RequirementCandidate(**payload)  # type: ignore[arg-type]


def codes(verdict) -> set[str]:
    return {finding.code for finding in verdict.findings}


# --------------------------------------------------------------------------
# Normalisation primitives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Accuracy of +/- 2.5 C from target", {"2.5"}),
        ("Accuracy >=90%", {"90"}),
        ("90", {"90"}),
        ("90.0 %", {"90"}),
        ("data loss <=10%", {"10"}),
        ("100% of data sent is received by the DT", {"100"}),
        ("plus or minus 2.50 degrees", {"2.5"}),
        ("up to 1,500 samples", {"1500"}),
        ("no numbers here", set()),
        (None, set()),
    ],
)
def test_numeric_token_normalisation(text: str | None, expected: set[str]) -> None:
    assert numeric_tokens(text) == expected


@pytest.mark.parametrize(
    ("word", "stem"),
    [("collected", "collect"), ("collect", "collect"), ("elements", "element"),
     ("retained", "retain"), ("clogs", "clog")],
)
def test_light_stemming(word: str, stem: str) -> None:
    assert light_stem(word) == stem


@pytest.mark.parametrize(
    ("inflected", "base"),
    [
        ("integrated", "integrate"),
        ("collected", "collect"),
        ("visualization", "visualize"),
        ("retained", "retain"),
    ],
)
def test_inflections_and_their_base_form_share_a_stem(
    inflected: str, base: str
) -> None:
    """The property V3 actually depends on. Lemma prettiness is irrelevant."""
    assert light_stem(inflected) == light_stem(base)


def test_content_words_drop_stop_words_and_domain_boilerplate() -> None:
    assert content_words("The DT shall collect the position in XYZ") == {
        "collect",
        "position",
        "xyz",
    }


@pytest.mark.parametrize(
    "text",
    [
        "The system shall minimize data loss",
        "The DT shall be reliable",
        "Response shall be fast",
        "The DT shall optimize the print",
    ],
)
def test_vague_predicates_are_detected_through_inflection(text: str) -> None:
    assert vague_predicate(text) is not None


def test_concrete_requirements_are_not_called_vague() -> None:
    assert vague_predicate("The DT shall pause the operation when clogs are identified") is None


def test_overlap_ratio_is_normalised_by_the_shorter_text() -> None:
    assert overlap_ratio({"a", "b", "c"}, {"a", "b"}) == 1.0
    assert overlap_ratio(set(), {"a"}) == 0.0


# --------------------------------------------------------------------------
# V1 - fabrication. Five cases, all mandatory.
# --------------------------------------------------------------------------


def test_v1_goal_with_a_valid_reference_passes(session: SessionState) -> None:
    session.add_turn("agent", "How accurate does the temperature have to be?")
    session.add_turn("stakeholder", "Within plus or minus 2.5 degrees C of target.")

    verdict = validate_candidate(
        candidate(
            designGoal="Accuracy of +/- 2.5 C from target",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=1,
        ),
        session,
    )

    assert verdict.findings == []
    assert not verdict.rejected


def test_v1_goal_with_no_reference_is_rejected(session: SessionState) -> None:
    session.add_turn("stakeholder", "Within plus or minus 2.5 degrees C.")

    verdict = validate_candidate(
        candidate(
            designGoal="Accuracy of +/- 2.5 C from target",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=None,
        ),
        session,
    )

    assert verdict.rejected
    assert codes(verdict) == {"fabricated_goal"}


def test_v1_number_absent_from_the_cited_turn_is_rejected(
    session: SessionState,
) -> None:
    """The headline case: the model invented "95%" out of a numberless answer."""
    session.add_turn("agent", "How reliably does clog detection have to work?")
    session.add_turn("stakeholder", "It just has to be good enough to trust.")

    verdict = validate_candidate(
        candidate(
            text="The DT shall detect clogs using vibration data.",
            designGoal="Accuracy >=95%",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=1,
        ),
        session,
    )

    assert verdict.rejected
    assert codes(verdict) == {"fabricated_goal"}
    assert "95" in verdict.hard_findings[0].message


def test_v1_reference_to_an_agent_turn_is_rejected(session: SessionState) -> None:
    """A number the agent said is not a source, even if it is in the transcript."""
    session.add_turn("agent", "Most shops aim for 90% detection accuracy.")
    session.add_turn("stakeholder", "That sounds about right.")

    verdict = validate_candidate(
        candidate(
            designGoal="Accuracy >=90%",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=0,
        ),
        session,
    )

    assert verdict.rejected
    assert codes(verdict) == {"fabricated_goal"}
    assert "agent speaking" in verdict.hard_findings[0].message


def test_v1_reference_out_of_range_is_rejected(session: SessionState) -> None:
    verdict = validate_candidate(
        candidate(
            designGoal="Accuracy >=90%",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=42,
        ),
        session,
    )
    assert verdict.rejected
    assert codes(verdict) == {"fabricated_goal"}


def test_v1_a_goal_carrying_no_number_still_needs_a_stakeholder_source(
    session: SessionState,
) -> None:
    session.add_turn("agent", "Any target for that?")
    verdict = validate_candidate(
        candidate(
            designGoal="no dropped frames",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=0,
        ),
        session,
    )
    assert verdict.rejected


def test_v1_says_nothing_when_there_is_no_goal(session: SessionState) -> None:
    verdict = validate_candidate(candidate(type="functional"), session)
    assert not verdict.rejected


# --------------------------------------------------------------------------
# V2 - the published UR1.11 defect
# --------------------------------------------------------------------------


def test_v2_flags_the_published_ur1_11_minimize_defect(session: SessionState) -> None:
    """UR1.11, verbatim from Figure 4 of the paper, as first stated."""
    verdict = validate_candidate(
        candidate(
            text="The system shall minimize data loss",
            type="performance",
            designGoal=None,
        ),
        session,
    )

    assert not verdict.rejected  # FLAG commits; it only blocks termination
    assert codes(verdict) == {"unverifiable_predicate"}
    assert "minimize" in verdict.soft_findings[0].message


def test_v2_is_silent_once_a_number_pins_the_predicate_down(
    session: SessionState,
) -> None:
    session.add_turn("stakeholder", "No more than 10% of the data may be lost.")
    verdict = validate_candidate(
        candidate(
            text="The system shall minimize data loss",
            type="performance",
            designGoal="data loss <=10%",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=0,
        ),
        session,
    )
    assert "unverifiable_predicate" not in codes(verdict)


# --------------------------------------------------------------------------
# V3 - the published UR1.2 / UR1.7 defect
# --------------------------------------------------------------------------

UR1_2 = (
    "The data for temperature, position, and vibration shall be collected "
    "continuously for every print job"
)
UR1_7 = "The DT shall collect bed and nozzle temperature continuously"


def test_v3_flags_the_published_ur1_2_ur1_7_duplicate(session: SessionState) -> None:
    """Both texts verbatim from Figure 4 of the paper."""
    session.commit_requirement(make_requirement("UR1.2", UR1_2))

    verdict = validate_candidate(
        candidate(text=UR1_7, type="performance"), session
    )

    assert not verdict.rejected
    assert "duplicate_obligation" in codes(verdict)
    finding = next(f for f in verdict.findings if f.code == "duplicate_obligation")
    assert finding.related_requirement_id == "UR1.2"


def test_the_threshold_is_tuned_to_that_known_pair() -> None:
    """Documents *why* the threshold is 0.6, so the tuning is not invisible.

    This pair is the calibration point named in the spec's honest-limitations
    section. If the stop-word list or the stemmer changes, this test is the one
    that says so.
    """
    score = overlap_ratio(content_words(UR1_2), content_words(UR1_7))
    assert score == pytest.approx(0.6, abs=1e-9)
    assert score >= DUPLICATE_THRESHOLD


def test_v3_does_not_flag_genuinely_different_obligations(
    session: SessionState,
) -> None:
    session.commit_requirement(make_requirement("UR1.2", UR1_2))
    session.commit_requirement(
        make_requirement("UR1.3", "All data elements shall be retained for every print job")
    )

    verdict = validate_candidate(
        candidate(text="The data collected from the physical system shall be integrated with the DT"),
        session,
    )

    assert "duplicate_obligation" not in codes(verdict)


def test_v3_ignores_two_short_requirements_sharing_only_generic_words(
    session: SessionState,
) -> None:
    """Reported from a live run: this pair looped the interview.

    Both reduce to three content words and share exactly two of them,
    "collect" and "job" - the two commonest words in this domain - which
    scores 0.67 on the ratio alone. The duration of a job and the type of a
    job are plainly different data elements, so this is a false positive, and
    each one cost a model call and a near-identical question to the human.
    """
    duration = "The DT shall collect data on the duration of the job"
    kind = "The DT shall collect data on the type of job"

    assert len(content_words(duration) & content_words(kind)) == 2
    assert overlap_ratio(content_words(duration), content_words(kind)) > DUPLICATE_THRESHOLD

    session.commit_requirement(make_requirement("UR1.1", duration))
    verdict = validate_candidate(candidate(text=kind, type="functional"), session)

    assert "duplicate_obligation" not in codes(verdict)


def test_v3_needs_shared_words_as_well_as_a_ratio(session: SessionState) -> None:
    """The evidence floor is what separates the two cases above and below it."""
    session.commit_requirement(make_requirement("UR1.2", UR1_2))
    shared = content_words(UR1_2) & content_words(UR1_7)
    assert len(shared) == DUPLICATE_MIN_SHARED_WORDS
    assert "duplicate_obligation" in codes(
        validate_candidate(candidate(text=UR1_7, type="performance"), session)
    )


def test_v3_raises_at_most_one_finding_per_requirement(
    session: SessionState,
) -> None:
    """N similar requirements must cost N questions, not N-squared.

    Every finding becomes a flag, every flag becomes a model call and a
    stakeholder turn. Asking about all 28 pairings of 8 requirements is how
    the interview turns into a loop that burns tokens.
    """
    texts = [
        f"The DT shall provide real-time visualization of the {part} readings"
        for part in ("nozzle", "bed", "extruder", "chamber", "filament",
                     "motor", "fan", "spool")
    ]
    verdicts = validate_batch(
        [candidate(text=t, type="functional", dimension="virtual_environment")
         for t in texts],
        session,
    )

    for verdict in verdicts:
        duplicates = [f for f in verdict.findings if f.code == "duplicate_obligation"]
        assert len(duplicates) <= 1

    total = sum(
        1 for v in verdicts for f in v.findings if f.code == "duplicate_obligation"
    )
    assert total < len(texts)          # linear, not the 28 pairings


def test_v3_names_the_closest_match_when_several_overlap(
    session: SessionState,
) -> None:
    """One question, about the requirement it most resembles."""
    session.commit_requirement(
        make_requirement("UR1.1", "The DT shall collect bed and nozzle temperature continuously")
    )
    session.commit_requirement(make_requirement("UR1.2", UR1_2))

    verdict = validate_candidate(candidate(text=UR1_7, type="performance"), session)
    duplicates = [f for f in verdict.findings if f.code == "duplicate_obligation"]

    assert len(duplicates) == 1
    assert duplicates[0].related_requirement_id == "UR1.1"   # the 1.00 match
    assert "sharing" in duplicates[0].message               # shows its evidence


def test_v3_only_compares_within_one_dimension(session: SessionState) -> None:
    session.commit_requirement(
        make_requirement("UR1.2", UR1_2, dimension="automation_feedback")
    )
    verdict = validate_candidate(
        candidate(text=UR1_7, dimension="data_collection_integration"), session
    )
    assert "duplicate_obligation" not in codes(verdict)


def test_v3_compares_candidates_from_the_same_answer_against_each_other(
    session: SessionState,
) -> None:
    """Two obligations stated in one breath are still two obligations."""
    verdicts = validate_batch(
        [
            candidate(text=UR1_2, type="functional"),
            candidate(text=UR1_7, type="functional"),
        ],
        session,
    )
    assert "duplicate_obligation" not in codes(verdicts[0])
    assert "duplicate_obligation" in codes(verdicts[1])


# --------------------------------------------------------------------------
# V4 - orphan goal
# --------------------------------------------------------------------------


def test_v4_rejects_a_goal_for_a_requirement_that_does_not_exist(
    session: SessionState,
) -> None:
    session.add_turn("stakeholder", "At least 90%.")
    verdict = validate_candidate(
        candidate(
            text="",
            goal_target_id="UR1.99",
            designGoal="Accuracy >=90%",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=0,
        ),
        session,
    )
    assert verdict.rejected
    assert "orphan_goal" in codes(verdict)


def test_v4_accepts_a_goal_for_a_committed_requirement(
    session: SessionState,
) -> None:
    session.add_turn("stakeholder", "At least 90%.")
    session.commit_requirement(
        make_requirement(
            "UR1.10",
            "The DT shall detect operation status using electric current data",
            dimension="intelligence_layer",
            requirement_type="performance",
        )
    )
    verdict = validate_candidate(
        candidate(
            text="",
            dimension="intelligence_layer",
            goal_target_id="UR1.10",
            designGoal="Accuracy >=90%",
            designGoal_provenance="stated",
            stakeholder_utterance_ref=0,
        ),
        session,
    )
    assert verdict.findings == []


# --------------------------------------------------------------------------
# V5 - maturity consistency
# --------------------------------------------------------------------------


def test_v5_flags_a_requirement_landing_in_a_dimension_ruled_out(
    session: SessionState,
) -> None:
    session.add_turn("agent", "Does the twin need to work anything out itself?")
    session.add_turn("stakeholder", "No, we do not want it deciding anything.")
    session.mark_not_applicable("intelligence_layer", 1)

    verdict = validate_candidate(
        candidate(
            text="The DT shall detect clogs using vibration data",
            dimension="intelligence_layer",
        ),
        session,
    )

    assert not verdict.rejected
    assert "maturity_inconsistency" in codes(verdict)
    finding = next(f for f in verdict.findings if f.code == "maturity_inconsistency")
    assert "we do not want it deciding anything" in finding.message


def test_v5_message_distinguishes_a_parked_dimension_from_a_ruled_out_one(
    session: SessionState,
) -> None:
    session.record_attempt("intelligence_layer")
    session.mark_parked("intelligence_layer", "no usable answer")

    verdict = validate_candidate(
        candidate(text="The DT shall detect clogs", dimension="intelligence_layer"),
        session,
    )

    finding = next(f for f in verdict.findings if f.code == "maturity_inconsistency")
    assert "parked" in finding.message


def test_v5_is_silent_for_a_covered_dimension(session: SessionState) -> None:
    session.coverage_status["intelligence_layer"] = "covered"
    verdict = validate_candidate(
        candidate(text="The DT shall detect clogs", dimension="intelligence_layer"),
        session,
    )
    assert "maturity_inconsistency" not in codes(verdict)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_the_validator_is_deterministic(session: SessionState) -> None:
    session.add_turn("stakeholder", "It just has to be good enough.")
    subject = candidate(
        designGoal="Accuracy >=95%",
        designGoal_provenance="stated",
        stakeholder_utterance_ref=0,
    )
    first = validate_candidate(subject, session)
    second = validate_candidate(subject, session)
    assert [f.message for f in first.findings] == [f.message for f in second.findings]
