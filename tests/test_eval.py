"""Phase 7 - personas, ground truth, and the evaluation harness.

The whole harness runs offline in stub mode, which is what wires it into CI:
the deterministic core and the edge rules are regression-guarded by the same
code that produces the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dtv_rea.persona import PersonaError, PersonaScript
from dtv_rea.settings import (
    ATTEMPT_CAP,
    DIMENSIONS,
    MODEL_NAME,
    ground_truth_dir,
)

from eval.metrics import (
    coverage_completeness,
    defect_detection,
    extraction_fidelity,
    fabricated_goal_rate,
    interception_rate,
    load_ground_truth,
    robustness,
    well_formedness,
)
from eval.run_eval import PERSONA_ORDER, main, render_report, run_one, score

from tests.conftest import make_requirement


# --------------------------------------------------------------------------
# The ground-truth file
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fdm_truth() -> dict:
    return load_ground_truth(ground_truth_dir() / "fdm.json")


def test_ground_truth_has_all_twelve_published_requirements(fdm_truth: dict) -> None:
    ids = [item["id"] for item in fdm_truth["requirements"]]
    assert ids == [f"UR1.{n}" for n in range(1, 13)]


def test_ground_truth_carries_the_six_published_design_goals(
    fdm_truth: dict,
) -> None:
    goals = {
        item["designGoal"]
        for item in fdm_truth["requirements"]
        if item["designGoal"]
    }
    assert goals == {
        "Accuracy of +/- 2.5 C from target",
        "Accuracy of +/- 10% from target",
        "Accuracy >=90%",
        "data loss <=10%",
        "100% of data sent is received by the DT",
    }


def test_ground_truth_cites_its_source(fdm_truth: dict) -> None:
    """A case with no citation cannot anchor an extraction-fidelity claim."""
    source = fdm_truth["source"]
    assert source["doi"] == "10.1080/00207543.2025.2524516"
    assert "Figure 4" in source["figure"]
    assert "verbatim" in source["note"]


def test_ground_truth_records_both_real_paper_defects(fdm_truth: dict) -> None:
    defects = {d["code"]: d["requirements"] for d in fdm_truth["seeded_defects"]}
    assert defects["unverifiable_predicate"] == ["UR1.11"]
    assert defects["duplicate_obligation"] == ["UR1.2", "UR1.7"]


def test_only_performance_requirements_carry_design_goals(fdm_truth: dict) -> None:
    for item in fdm_truth["requirements"]:
        if item["designGoal"]:
            assert item["type"] == "performance"


def test_every_ground_truth_dimension_is_a_real_dimension(fdm_truth: dict) -> None:
    for item in fdm_truth["requirements"]:
        assert item["dimension"] in DIMENSIONS


def test_a_new_case_needs_no_code_change() -> None:
    """The spec's stated fix for n=1 is to drop in another <case>.json pair."""
    assert ground_truth_dir().is_dir()
    assert sorted(p.stem for p in ground_truth_dir().glob("*.json")) == ["fdm"]


# --------------------------------------------------------------------------
# The persona files
# --------------------------------------------------------------------------


def test_all_six_personas_load() -> None:
    assert sorted(PersonaScript.available()) == sorted(PERSONA_ORDER)
    for name in PERSONA_ORDER:
        script = PersonaScript.by_name(name)
        assert script.description
        assert script.purpose_answer


def test_a_missing_persona_raises_a_clear_error() -> None:
    with pytest.raises(PersonaError):
        PersonaScript.by_name("no_such_persona")


def test_personas_never_hardcode_an_agent_assigned_requirement_id() -> None:
    """Ids are allocated in commit order; a persona pinning one would be brittle."""
    for name in PERSONA_ORDER:
        path = Path(PersonaScript.by_name(name).source)
        for entry in json.loads(path.read_text(encoding="utf-8")).get("turns", []):
            for candidate in entry.get("extract", {}).get("requirements", []):
                assert "goal_target_id" not in candidate, name


# --------------------------------------------------------------------------
# Metrics, on hand-built inputs
# --------------------------------------------------------------------------


def test_fabrication_metric_counts_an_untraceable_goal(session) -> None:
    session.add_turn("agent", "Most shops use 90%.")
    session.commit_requirement(
        make_requirement(
            "UR1.1", "The DT shall detect clogs", requirement_type="performance",
            design_goal="Accuracy >=90%", ref=0,   # an AGENT turn
        )
    )
    result = fabricated_goal_rate(session)
    assert result["fabricated"] == 1
    assert result["fabricated_ids"] == ["UR1.1"]
    assert not result["passes"]


def test_interception_is_reported_separately_from_attempts(session) -> None:
    from dtv_rea.state import Flag

    session.add_flag(
        Flag(id="F1", code="fabricated_goal", severity="hard", message="x")
    )
    result = interception_rate(session)
    assert result["attempted_by_model"] == 1
    assert result["rejected_pre_commit"] == 1
    assert result["rate"] == 1.0
    assert result["passes"]


def test_well_formedness_catches_a_named_solution(session) -> None:
    session.commit_requirement(
        make_requirement("UR1.1", "The DT shall stream readings over MQTT")
    )
    result = well_formedness(session)
    assert result["well_formed"] == 0
    assert "names a solution: mqtt" in result["problems"][0]


def test_well_formedness_catches_a_missing_shall(session) -> None:
    session.commit_requirement(make_requirement("UR1.1", "Collect the temperature"))
    assert "not in shall form" in well_formedness(session)["problems"][0]


def test_coverage_reports_parked_separately_from_covered(session) -> None:
    for dimension in DIMENSIONS:
        session.coverage_status[dimension] = "covered"
    session.mark_parked("virtual_environment", "capped")
    result = coverage_completeness(session)
    assert result["resolved"] == 4
    assert result["parked"] == ["virtual_environment"]


def test_robustness_reports_n_a_rather_than_zero_when_unmeasured() -> None:
    result = robustness([{"call": "ask"}])
    assert result["latency_p50_ms"] is None
    assert result["tokens_total"] is None
    assert result["parse_retry_rate"] is None


def test_robustness_computes_the_parse_retry_rate() -> None:
    result = robustness(
        [
            {"call": "extract", "parse_retries": 0, "latency_ms": 100},
            {"call": "extract", "parse_retries": 1, "latency_ms": 300},
        ]
    )
    assert result["parse_retry_rate"] == 0.5
    assert result["latency_p50_ms"] == 300
    assert not result["passes"]     # 50% is far above the 10% bar


def test_fidelity_matches_by_text_not_by_id(session, fdm_truth: dict) -> None:
    """The agent's UR1.4 can legitimately be the paper's UR1.7."""
    session.add_turn("stakeholder", "Within plus or minus 2.5 degrees C.")
    session.commit_requirement(
        make_requirement(
            "UR1.4",
            "The DT shall collect bed and nozzle temperature continuously",
            requirement_type="performance",
            design_goal="Accuracy of +/- 2.5 C from target",
            ref=0,
        )
    )
    result = extraction_fidelity(session, fdm_truth)
    row = next(r for r in result["rows"] if r["expected_id"] == "UR1.7")
    assert row["matched_id"] == "UR1.4"
    assert row["goal_ok"] is True
    assert row["ref_ok"] is True


def test_fidelity_compares_goals_numerically_not_as_strings(
    session, fdm_truth: dict
) -> None:
    session.add_turn("stakeholder", "At least 90 percent.")
    session.commit_requirement(
        make_requirement(
            "UR1.1",
            "The DT shall detect clogs using vibration data",
            dimension="intelligence_layer",
            requirement_type="performance",
            design_goal="accuracy of at least 90 %",
            ref=0,
        )
    )
    row = next(
        r
        for r in extraction_fidelity(session, fdm_truth)["rows"]
        if r["expected_id"] == "UR1.9"
    )
    assert row["goal_ok"] is True


def test_fidelity_reports_a_wrong_number_as_a_miss(session, fdm_truth: dict) -> None:
    session.add_turn("stakeholder", "About 95 percent.")
    session.commit_requirement(
        make_requirement(
            "UR1.1",
            "The DT shall detect clogs using vibration data",
            dimension="intelligence_layer",
            requirement_type="performance",
            design_goal="Accuracy >=95%",
            ref=0,
        )
    )
    row = next(
        r
        for r in extraction_fidelity(session, fdm_truth)["rows"]
        if r["expected_id"] == "UR1.9"
    )
    assert row["goal_ok"] is False


def test_defect_detection_needs_the_flag_to_be_present(session, fdm_truth: dict) -> None:
    assert defect_detection(session, fdm_truth)["detected"] == 0


# --------------------------------------------------------------------------
# The harness end to end, offline
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scored() -> dict[str, dict]:
    """Every persona replayed through the real graph, once."""
    import os
    import tempfile

    directory = tempfile.mkdtemp()
    previous = os.environ.get("DTV_REA_RUNS_DIR")
    os.environ["DTV_REA_RUNS_DIR"] = str(Path(directory) / "runs")
    try:
        results = {}
        for name in PERSONA_ORDER:
            session, records = run_one(name, True, f"test-eval-{name}")
            results[name] = score(name, session, records)
        return results
    finally:
        if previous is None:
            os.environ.pop("DTV_REA_RUNS_DIR", None)
        else:
            os.environ["DTV_REA_RUNS_DIR"] = previous


def test_no_persona_commits_a_fabricated_goal(scored: dict) -> None:
    """The headline claim, across every scripted stakeholder."""
    for name, result in scored.items():
        assert result["fabrication"]["fabricated"] == 0, name
        assert result["fabrication"]["passes"], name


def test_every_fabrication_attempt_was_intercepted(scored: dict) -> None:
    for name, result in scored.items():
        assert result["interception"]["slipped_through"] == 0, name


def test_fabrication_attempts_are_actually_being_made(scored: dict) -> None:
    """A zero rate proves nothing if nothing ever tried to fabricate."""
    attempted = sum(r["interception"]["attempted_by_model"] for r in scored.values())
    assert attempted >= 5


def test_the_fdm_case_meets_every_headline_target(scored: dict) -> None:
    fdm = scored["fdm_stakeholder"]
    assert fdm["fidelity"]["matched"] == 12
    assert fdm["fidelity"]["goals_exact"] == fdm["fidelity"]["goals_expected"] == 6
    assert fdm["defects"]["detected"] == 2
    assert fdm["coverage"]["resolved"] == 4
    assert fdm["efficiency"]["stakeholder_turns"] <= 12
    assert fdm["maturity"]["proposed"] == "Replication"
    assert fdm["well_formed"]["rate"] == 1.0


def test_q1_parks_a_missing_number_without_inventing_one(scored: dict) -> None:
    edges = scored["dont_know"]["edges"]
    assert edges["q1_needs_clarification"]
    assert edges["q1_all_have_null_goals"]
    assert scored["dont_know"]["fabrication"]["fabricated"] == 0


def test_q2_marks_the_stopped_session_partial(scored: dict) -> None:
    assert scored["quitter"]["edges"]["q2_status"] == "partial"
    assert scored["quitter"]["coverage"]["resolved"] < 4


def test_q3_parks_after_exactly_three_attempts(scored: dict) -> None:
    edges = scored["evasive"]["edges"]
    assert edges["q3_parked"] == ["virtual_environment"]
    assert edges["q3_attempts_on_parked"]["virtual_environment"] == ATTEMPT_CAP
    assert edges["q3_respects_cap"]


def test_the_contradiction_persona_raises_v5(scored: dict) -> None:
    session_flags = scored["contradictory"]
    assert session_flags["coverage"]["resolved"] == 4
    assert session_flags["hitl"]["hitl3_open_flags_at_end"] == 0


def test_the_no_numbers_persona_intercepts_everything(scored: dict) -> None:
    interception = scored["no_numbers"]["interception"]
    assert interception["attempted_by_model"] == 4
    assert interception["rejected_pre_commit"] == 4
    assert interception["rate"] == 1.0


def test_no_dimension_is_ever_asked_about_more_than_the_cap(scored: dict) -> None:
    for name, result in scored.items():
        assert result["edges"]["q3_respects_cap"], name


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_the_report_states_the_headline_metric_and_its_target(scored: dict) -> None:
    report = render_report(list(scored.values()), use_stub=True)
    assert "**Fabricated-goal rate**" in report
    assert "| **0** |" in report


def test_the_report_separates_interception_from_model_behaviour(
    scored: dict,
) -> None:
    report = render_report(list(scored.values()), use_stub=True)
    assert "LLM fabrication-attempt rate" in report
    assert "Interception measures this system" in report


def test_the_report_refuses_to_overclaim_from_one_case(scored: dict) -> None:
    report = render_report(list(scored.values()), use_stub=True)
    assert "n = 1 real scenario" in report
    assert "No statistical claim should be made" in report
    assert "Ronanki et al. (2023)" in report
    assert "keyword heuristic" in report
    assert "tuned to the known case" in report


def test_the_report_says_a_stub_run_measures_the_core_not_the_model(
    scored: dict,
) -> None:
    report = render_report(list(scored.values()), use_stub=True)
    assert "proves nothing about how" in report
    assert MODEL_NAME in report


def test_the_harness_exits_zero_when_nothing_was_fabricated(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    assert main(["--stub", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# DTV-REA evaluation report")


def test_a_single_persona_can_be_scored_on_its_own(tmp_path: Path) -> None:
    out = tmp_path / "one.md"
    assert main(["--stub", "--persona", "quitter", "--out", str(out)]) == 0
    assert "`quitter`" in out.read_text(encoding="utf-8")


def test_the_harness_can_emit_machine_readable_results(tmp_path: Path) -> None:
    out = tmp_path / "r.md"
    main(["--stub", "--persona", "evasive", "--out", str(out), "--json"])
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert payload[0]["persona"] == "evasive"
