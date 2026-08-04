"""Replay every persona through the real graph and score it (spec section 5).

Two modes, one code path:

``--stub``   the scripted model. No network, no API key. This is what CI runs.
(default)    the real Groq model, so the live numbers can be diffed against the
             stub run.

Usage, identical on macOS, Linux and Windows::

    python -m eval.run_eval --stub
    python -m eval.run_eval --stub --persona fdm_stakeholder
    python -m eval.run_eval                      (live, needs GROQ_API_KEY)

Writes ``eval/report.md``: one table per persona plus an aggregate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct `python eval/run_eval.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtv_rea.graph import build_graph, open_checkpointer
from dtv_rea.llm import CallLogger, StubModel
from dtv_rea.outputs import write_all
from dtv_rea.persona import PersonaScript
from dtv_rea.runner import persona_answerer, run_session
from dtv_rea.settings import (
    ATTEMPT_CAP,
    DUPLICATE_MIN_SHARED_WORDS,
    DUPLICATE_THRESHOLD,
    MODEL_NAME,
    eval_dir,
    ground_truth_dir,
    runs_dir,
)
from dtv_rea.state import SessionState

from eval.metrics import (
    coverage_completeness,
    defect_detection,
    edge_rules,
    elicitation_efficiency,
    extraction_fidelity,
    fabricated_goal_rate,
    hitl_conformance,
    interception_rate,
    load_ground_truth,
    maturity_quality,
    robustness,
    well_formedness,
)

#: Which persona is the ground-truth case, and which edge rule each one exists
#: to exercise. Adding a case means adding a persona file and a line here.
PERSONA_ORDER = (
    "fdm_stakeholder",
    "evasive",
    "dont_know",
    "contradictory",
    "no_numbers",
    "quitter",
)

GROUND_TRUTH_FOR = {"fdm_stakeholder": "fdm"}

EDGE_RULE_FOR = {
    "dont_know": "Q1",
    "quitter": "Q2",
    "evasive": "Q3",
}


def run_one(
    name: str, use_stub: bool, session_id: str
) -> tuple[SessionState, list[dict[str, Any]]]:
    """Replay one persona through the real graph."""
    script = PersonaScript.by_name(name)
    logger = CallLogger(runs_dir() / session_id / "llm_calls.jsonl")

    if use_stub:
        model: Any = StubModel(script, logger=logger)
    else:
        from dtv_rea.groq_model import GroqModel

        model = GroqModel(logger=logger)

    saver, connection = open_checkpointer()
    try:
        graph = build_graph(model, checkpointer=saver)
        session = run_session(graph, session_id, persona_answerer(script))
    finally:
        connection.close()

    write_all(session)
    return session, logger.records


def score(name: str, session: SessionState, records: list[dict[str, Any]]) -> dict[str, Any]:
    script = PersonaScript.by_name(name)
    result: dict[str, Any] = {
        "persona": name,
        "description": script.description,
        "edge_rule": EDGE_RULE_FOR.get(name),
        "fabrication": fabricated_goal_rate(session),
        "interception": interception_rate(session),
        "coverage": coverage_completeness(session),
        "efficiency": elicitation_efficiency(session),
        "maturity": maturity_quality(session, script.expected_level),
        "well_formed": well_formedness(session),
        "edges": edge_rules(session),
        "hitl": hitl_conformance(session),
        "ops": robustness(records),
        "status": session.status,
    }

    case = GROUND_TRUTH_FOR.get(name)
    if case:
        ground_truth = load_ground_truth(ground_truth_dir() / f"{case}.json")
        result["fidelity"] = extraction_fidelity(session, ground_truth)
        result["defects"] = defect_detection(session, ground_truth)
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _mark(passed: bool | None) -> str:
    if passed is None:
        return "n/a"
    return "PASS" if passed else "**FAIL**"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value: Any, suffix: str = "") -> str:
    """Render a measurement that the stub mode legitimately does not produce."""
    return "n/a" if value is None else f"{value}{suffix}"


def render_report(results: list[dict[str, Any]], use_stub: bool) -> str:
    lines: list[str] = []
    lines.append("# DTV-REA evaluation report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(
        f"- Mode: **{'stub (offline, scripted model)' if use_stub else f'live ({MODEL_NAME} on Groq)'}**"
    )
    lines.append(f"- Personas: {len(results)}")
    lines.append(f"- Attempt cap N = {ATTEMPT_CAP}; duplicate threshold = {DUPLICATE_THRESHOLD}")
    lines.append("")

    total_goals = sum(r["fabrication"]["committed_goals"] for r in results)
    total_fabricated = sum(r["fabrication"]["fabricated"] for r in results)
    attempted = sum(r["interception"]["attempted_by_model"] for r in results)
    rejected = sum(r["interception"]["rejected_pre_commit"] for r in results)

    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value | Target | Verdict |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| **Fabricated-goal rate** | {total_fabricated} of {total_goals} "
        f"committed goals | **0** | {_mark(total_fabricated == 0)} |"
    )
    lines.append(
        f"| Validator interception rate | {rejected} of {attempted} fabricated "
        f"candidates caught pre-commit | 100% | "
        f"{_mark(attempted == 0 or rejected == attempted)} |"
    )
    lines.append(
        f"| LLM fabrication-attempt rate | {attempted} attempts across "
        f"{len(results)} sessions | reported, not targeted | n/a |"
    )
    lines.append("")
    lines.append(
        "The first two rows measure different things. Interception measures "
        "this system; the attempt rate measures the model, and is reported "
        "separately so improvements in one are never mistaken for the other. "
        "The fabricated-goal rate is re-derived from each transcript "
        "independently of the validator's own verdicts."
    )
    lines.append("")

    lines.append("## Per persona")
    lines.append("")
    lines.append(
        "| Persona | Edge rule | Status | Reqs | Turns | Coverage | "
        "Fabricated | Intercepted | Well-formed | Open flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for result in results:
        coverage = result["coverage"]
        lines.append(
            f"| `{result['persona']}` "
            f"| {result['edge_rule'] or '-'} "
            f"| {result['status']} "
            f"| {result['efficiency']['requirements']} "
            f"| {result['efficiency']['stakeholder_turns']} "
            f"| {coverage['resolved']}/{coverage['total']} "
            f"| {result['fabrication']['fabricated']} "
            f"| {result['interception']['rejected_pre_commit']}"
            f"/{result['interception']['attempted_by_model']} "
            f"| {_pct(result['well_formed']['rate'])} "
            f"| {result['hitl']['hitl3_open_flags_at_end']} |"
        )
    lines.append("")

    for result in results:
        lines.extend(_render_persona(result))

    lines.extend(_render_limitations())
    return "\n".join(lines)


def _render_persona(result: dict[str, Any]) -> list[str]:
    lines = [f"## `{result['persona']}`", ""]
    if result["description"]:
        lines.append(f"_{result['description']}_")
        lines.append("")

    lines.append("| Metric | Value | Verdict |")
    lines.append("|---|---|---|")

    fabrication = result["fabrication"]
    offenders = (
        f" ({', '.join(fabrication['fabricated_ids'])})"
        if fabrication["fabricated_ids"]
        else ""
    )
    lines.append(
        f"| Fabricated-goal rate | {fabrication['fabricated']}"
        f"/{fabrication['committed_goals']} committed goals{offenders} "
        f"| {_mark(fabrication['passes'])} |"
    )
    interception = result["interception"]
    lines.append(
        f"| Validator interception | {interception['rejected_pre_commit']} caught, "
        f"{interception['slipped_through']} slipped | {_mark(interception['passes'])} |"
    )
    coverage = result["coverage"]
    lines.append(
        f"| Coverage completeness | {coverage['resolved']}/{coverage['total']}"
        f"{' (parked: ' + ', '.join(coverage['parked']) + ')' if coverage['parked'] else ''}"
        f" | {_mark(coverage['passes'])} |"
    )
    efficiency = result["efficiency"]
    lines.append(
        f"| Elicitation efficiency | {efficiency['stakeholder_turns']} stakeholder "
        f"turns for {efficiency['requirements']} requirements "
        f"({efficiency['requirements_per_turn']}/turn) | {_mark(efficiency['passes'])} |"
    )
    maturity = result["maturity"]
    lines.append(
        f"| Maturity proposal | proposed {maturity['proposed']}, expected "
        f"{maturity['expected']}, description shown: "
        f"{maturity['description_shown']} ({maturity.get('description_chars', 0)} chars) "
        f"| {_mark(maturity['passes'])} |"
    )
    well_formed = result["well_formed"]
    lines.append(
        f"| Requirement well-formedness | {well_formed['well_formed']}"
        f"/{well_formed['total']} ({_pct(well_formed['rate'])}) "
        f"| {_mark(well_formed['passes'])} |"
    )
    hitl = result["hitl"]
    lines.append(
        f"| HITL conformance | maturity confirmed: "
        f"{hitl['hitl1_maturity_confirmed']}, open flags at end: "
        f"{hitl['hitl3_open_flags_at_end']}, duplicates human-resolved only: "
        f"{hitl['duplicates_resolved_by_human_only']} | {_mark(hitl['passes'])} |"
    )
    ops = result["ops"]
    lines.append(
        f"| Robustness | {ops['calls']} calls, parse-retry rate "
        f"{_pct(ops['parse_retry_rate'])}, latency p50 "
        f"{_num(ops['latency_p50_ms'], ' ms')} / p95 "
        f"{_num(ops['latency_p95_ms'], ' ms')}, tokens "
        f"{_num(ops['tokens_total'])} | {_mark(ops['passes'])} |"
    )

    if "fidelity" in result:
        fidelity = result["fidelity"]
        lines.append(
            f"| Extraction fidelity (vs published FDM set) | "
            f"{fidelity['matched']}/{fidelity['expected']} requirements matched, "
            f"{fidelity['goals_exact']}/{fidelity['goals_expected']} design goals "
            f"exact | {_mark(fidelity['passes'])} |"
        )
        defects = result["defects"]
        lines.append(
            f"| Defect detection | {defects['detected']}/{defects['expected']} "
            f"seeded paper defects flagged | {_mark(defects['passes'])} |"
        )

    edges = result["edges"]
    lines.append(
        f"| Edge rules | Q1 parked: {edges['q1_needs_clarification'] or 'none'}; "
        f"Q2 status: {edges['q2_status']}; Q3 parked: {edges['q3_parked'] or 'none'} "
        f"{edges['q3_attempts_on_parked'] or ''} | "
        f"{_mark(edges['q3_respects_cap'] and edges['q1_all_have_null_goals'])} |"
    )
    lines.append("")

    if well_formed["problems"]:
        lines.append("Well-formedness problems:")
        lines.append("")
        for problem in well_formed["problems"]:
            lines.append(f"- {problem}")
        lines.append("")

    if "fidelity" in result:
        lines.append("### Requirement-by-requirement against the published set")
        lines.append("")
        lines.append("| Published | Matched | Type | Dimension | Goal | Ref |")
        lines.append("|---|---|---|---|---|---|")
        for row in result["fidelity"]["rows"]:
            lines.append(
                f"| {row['expected_id']} | {row['matched_id'] or '**none**'} "
                f"| {'ok' if row['type_ok'] else 'no'} "
                f"| {'ok' if row['dimension_ok'] else 'no'} "
                f"| {'-' if row['goal_ok'] is None else ('ok' if row['goal_ok'] else 'no')} "
                f"| {'-' if row['ref_ok'] is None else ('ok' if row['ref_ok'] else 'no')} |"
            )
        lines.append("")
        lines.append(
            "Matching is by requirement text, not by id: the agent numbers "
            "requirements in the order the stakeholder raised them, which is "
            "not the order the paper lists them in."
        )
        lines.append("")
        lines.append(f"_{result['fidelity']['caveat']}_")
        lines.append("")

    return lines


def _render_limitations() -> list[str]:
    return [
        "## What these numbers do not show",
        "",
        "- **n = 1 real scenario.** One reverse-engineered ground-truth case "
        "(FDM) is not a sample. No statistical claim should be made from this "
        "report until there are at least three ground-truth cases. The other "
        "five personas are synthetic and were written to exercise specific "
        "code paths, so they measure conformance, not generalisation.",
        "",
        "- **Semantic text quality is not scored here.** Requirement matching "
        "above is lexical. Judging whether a requirement means the same thing "
        "as the published one, and whether it is clearly written, needs the "
        "human rubric adapted from Ronanki et al. (2023).",
        "",
        "- **V2 is a keyword heuristic.** It catches vague predicates from a "
        "word list. It cannot tell that a requirement is unverifiable for any "
        "other reason.",
        "",
        "- **V3's threshold is tuned to the known case.** It sits at "
        f"{DUPLICATE_THRESHOLD} because the published UR1.2 / UR1.7 pair scores "
        "exactly that. It will flag pairs a human considers distinct - that is "
        "why V3 only raises a question and never merges anything. Two guards "
        f"bound the noise: a pair must share at least {DUPLICATE_MIN_SHARED_WORDS} "
        "content words as well as clearing the ratio, and each requirement "
        "raises at most one question, about its closest match.",
        "",
        "- **Single stakeholder.** The agent interviews one person. Nothing "
        "here says anything about reconciling several stakeholders who "
        "disagree.",
        "",
        "- **In stub mode the model is scripted.** A stub run proves the "
        "deterministic core - routing, validation, edge rules, output - and "
        "proves nothing about how Llama 3.3 actually behaves. Only the live "
        "run measures the model.",
        "",
    ]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="Replay the personas through the graph and score every metric.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline scripted model. No API key, no internet.",
    )
    parser.add_argument(
        "--persona", action="append", metavar="NAME", help="Score only this persona."
    )
    parser.add_argument(
        "--out", metavar="PATH", help="Where to write the report (default eval/report.md)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Also write report.json beside the report."
    )
    args = parser.parse_args(argv)

    names = args.persona or list(PERSONA_ORDER)
    prefix = "stub" if args.stub else "live"

    results: list[dict[str, Any]] = []
    for name in names:
        session, records = run_one(name, args.stub, f"eval-{prefix}-{name}")
        results.append(score(name, session, records))

    report = render_report(results, args.stub)
    out_path = Path(args.out) if args.out else (eval_dir() / "report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    if args.json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    fabricated = sum(r["fabrication"]["fabricated"] for r in results)
    slipped = sum(r["interception"]["slipped_through"] for r in results)

    print(f"Scored {len(results)} persona(s) in {prefix} mode.")
    print(f"Report: {out_path}")
    print(f"Fabricated goals committed: {fabricated} (target 0)")

    # A nonzero fabricated-goal count is the headline negative result. Failing
    # the exit code makes CI say so instead of quietly filing the report.
    return 1 if (fabricated or slipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
