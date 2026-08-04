"""Phase 3 - the prompts and the knowledge base they carry.

These tests guard the properties the architecture depends on. They cannot check
that Llama 3.3 obeys the prompt - that is the validator's job, and deliberately
so - but they can check that the instructions and the grounding are present.
"""

from __future__ import annotations

import pytest

from dtv_rea.prompts import (
    P1_SYSTEM,
    P2_CAPTURE_PURPOSE,
    P3_PROPOSE_MATURITY,
    P4_GENERATE_QUESTION,
    P5_EXTRACT,
    P5_RETRY_SUFFIX,
    P6_RESOLVE_FLAG,
    committed_summary,
)
from dtv_rea.settings import DIMENSIONS, MATURITY_LEVELS

from tests.conftest import make_requirement


def says(haystack: str, phrase: str) -> bool:
    """Substring test that ignores where the prompt happens to wrap its lines.

    Prompts are hand-wrapped prose. Asserting on raw substrings would make
    these tests fail every time a sentence is re-flowed, which tests the
    formatter rather than the instruction.
    """

    def collapse(text: str) -> str:
        return " ".join(text.split()).lower()

    return collapse(phrase) in collapse(haystack)


# --------------------------------------------------------------------------
# P1 carries the whole knowledge base - there is no retrieval step to fall back
# on, so anything missing here is missing at inference time.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("level", MATURITY_LEVELS)
def test_p1_defines_every_4r_level(level: str) -> None:
    assert level in P1_SYSTEM


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_p1_names_every_requirement_dimension(dimension: str) -> None:
    assert dimension in P1_SYSTEM


@pytest.mark.parametrize(
    "requirement_id",
    [f"UR1.{n}" for n in range(1, 13)],
)
def test_p1_carries_all_twelve_fdm_exemplar_requirements(
    requirement_id: str,
) -> None:
    assert requirement_id in P1_SYSTEM


@pytest.mark.parametrize(
    "goal", ["+/- 2.5 C", "+/- 10%", ">=90%", "<=10%", "100%"]
)
def test_p1_carries_the_published_design_goals(goal: str) -> None:
    assert goal in P1_SYSTEM


def test_p1_carries_the_three_layer_conceptual_pattern() -> None:
    for layer in ("Data layer", "Service layer", "Model layer"):
        assert layer in P1_SYSTEM


def test_p1_states_the_no_invented_numbers_rule() -> None:
    assert says(P1_SYSTEM, "never invent a number")


def test_p1_warns_against_the_specific_failure_mode() -> None:
    """"Sensible default" is the exact shape fabrication takes in practice."""
    assert says(P1_SYSTEM, "sensible default")
    assert says(P1_SYSTEM, "industry standard")
    assert says(P1_SYSTEM, "reasonable starting point")


def test_p1_forbids_framework_vocabulary_towards_the_stakeholder() -> None:
    assert says(P1_SYSTEM, "plain language")


def test_p1_records_that_levels_do_not_partition_dimensions() -> None:
    """The FDM case is Replication and still pauses the machine."""
    assert says(P1_SYSTEM, "Only the stakeholder can rule a dimension out")


def test_p1_names_the_two_seeded_defects_as_things_to_surface() -> None:
    assert "minimize" in P1_SYSTEM
    assert says(P1_SYSTEM, "UR1.2 and UR1.7")


def test_p1_shows_that_exemplars_are_solution_free() -> None:
    for tool in ("MQTT", "InfluxDB", "Grafana", "Unity"):
        assert tool in P1_SYSTEM
    assert says(P1_SYSTEM, "Solutions belong to a later DTV step.")


def test_knowledge_base_fits_comfortably_in_context() -> None:
    """No-RAG is only defensible while the corpus stays small.

    Roughly four characters per token. If this ever approaches the 128k window,
    the revisit trigger documented in the module docstring has fired.
    """
    assert len(P1_SYSTEM) / 4 < 20_000


# --------------------------------------------------------------------------
# P2 - P6 substitute cleanly and say the load-bearing things
# --------------------------------------------------------------------------


def test_p2_asks_for_exactly_one_open_question() -> None:
    rendered = P2_CAPTURE_PURPOSE.substitute()
    assert says(rendered, "One question")
    assert says(rendered, "any number")


def test_p3_demands_a_description_not_just_a_label() -> None:
    rendered = P3_PROPOSE_MATURITY.substitute(purpose="Catch clogs mid-print.")
    assert "Catch clogs mid-print." in rendered
    assert says(rendered, "A bare label is not acceptable")
    assert says(rendered, "IN PLAIN LANGUAGE")
    assert says(rendered, "what it will NOT do")


def test_p4_substitutes_and_carries_the_fabrication_recovery_rule() -> None:
    rendered = P4_GENERATE_QUESTION.substitute(
        purpose="Catch clogs mid-print.",
        maturity="Replication",
        committed="(none yet)",
        focus="A design goal was rejected for UR1.9.",
    )
    assert says(rendered, "FABRICATION RECOVERY")
    assert says(rendered, "ask the stakeholder directly for that value")
    assert "A design goal was rejected for UR1.9." in rendered


def test_p4_forbids_putting_a_number_in_the_stakeholders_mouth() -> None:
    rendered = P4_GENERATE_QUESTION.substitute(
        purpose="p", maturity="Replication", committed="c", focus="f"
    )
    assert "90%" in rendered  # the worked example of what NOT to ask
    assert says(rendered, "Never propose a number")


def test_p5_substitutes_around_its_json_braces() -> None:
    rendered = P5_EXTRACT.substitute(
        purpose="Catch clogs mid-print.",
        focus="data_collection_integration",
        committed="(none yet)",
        turn_index=7,
        answer="Temperature to within 2.5 degrees.",
    )
    assert '"requirements"' in rendered
    assert "turn number 7" in rendered
    assert "Temperature to within 2.5 degrees." in rendered


def test_p5_states_every_rule_the_schema_relies_on() -> None:
    rendered = P5_EXTRACT.substitute(
        purpose="p", focus="f", committed="c", turn_index=1, answer="a"
    )
    assert says(rendered, "NEVER INVENT A NUMBER")
    assert '"designGoal": null' in rendered
    assert '"status": "needs_clarification"' in rendered   # Q1
    assert '"goal_target_id"' in rendered
    assert "The DT shall" in rendered
    assert says(rendered, "Never infer it from silence")


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_p5_lists_every_dimension_as_a_legal_value(dimension: str) -> None:
    rendered = P5_EXTRACT.substitute(
        purpose="p", focus="f", committed="c", turn_index=1, answer="a"
    )
    assert dimension in rendered


def test_p5_retry_forbids_changing_values_to_make_them_parse() -> None:
    rendered = P5_RETRY_SUFFIX.substitute(error="Expecting ',' delimiter")
    assert "Expecting ',' delimiter" in rendered
    assert says(rendered, "do not add a number that the stakeholder did not say")


def test_p6_refuses_to_let_the_agent_merge_anything() -> None:
    rendered = P6_RESOLVE_FLAG.substitute(
        code="duplicate_obligation", message="UR1.2 overlaps UR1.7 by 0.60."
    )
    assert "UR1.2 overlaps UR1.7 by 0.60." in rendered
    assert says(rendered, "never merge")
    assert says(rendered, "keeping both is a legitimate answer")
    assert says(rendered, "Do not suggest one.")


# --------------------------------------------------------------------------
# committed_summary
# --------------------------------------------------------------------------


def test_committed_summary_is_readable_when_empty() -> None:
    assert committed_summary([]) == "(none yet)"


def test_committed_summary_shows_ids_types_and_goals() -> None:
    rendered = committed_summary(
        [
            make_requirement(
                "UR1.7",
                "The DT shall collect bed and nozzle temperature continuously",
                requirement_type="performance",
                design_goal="Accuracy of +/- 2.5 C from target",
                ref=3,
            ),
            make_requirement("UR1.3", "All data elements shall be retained"),
        ]
    )
    assert "UR1.7 [performance/data_collection_integration]" in rendered
    assert "Accuracy of +/- 2.5 C from target" in rendered
    assert "no design goal recorded" in rendered
