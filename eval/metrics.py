"""Every metric in spec section 5, computed from a finished session.

The headline metric is deliberately computed by re-reading the transcript, not
by counting the validator's own verdicts. A metric that trusts the component it
is measuring measures nothing.

Where a metric genuinely needs human judgement - semantic text matching against
ground truth - this module says so rather than inventing a proxy and reporting
it as if it were the same thing.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from dtv_rea.settings import ATTEMPT_CAP, BANNED_SOLUTION_WORDS, DIMENSIONS
from dtv_rea.state import Requirement, SessionState
from dtv_rea.validator import (
    audit_committed_goals,
    content_words,
    numeric_tokens,
    overlap_ratio,
)

#: Two requirement texts at or above this content-word overlap are treated as
#: the same requirement for automated matching. Anything below it is reported
#: as unmatched and left for the human rubric - the harness never claims a
#: semantic match it cannot demonstrate.
TEXT_MATCH_THRESHOLD = 0.75


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


# --------------------------------------------------------------------------
# Headline: fabrication
# --------------------------------------------------------------------------


def fabricated_goal_rate(session: SessionState) -> dict[str, Any]:
    """Committed design goals that cannot be traced to a stakeholder utterance.

    Target: 0. Not "low" - zero. Any nonzero value is the headline negative
    result and must be reported as such.
    """
    committed = [r for r in session.requirements if r.designGoal is not None]
    fabricated = audit_committed_goals(session)
    rate = len(fabricated) / len(committed) if committed else 0.0
    return {
        "committed_goals": len(committed),
        "fabricated": len(fabricated),
        "fabricated_ids": [r.id for r in fabricated],
        "rate": rate,
        "passes": not fabricated,
    }


def interception_rate(session: SessionState) -> dict[str, Any]:
    """Fabricated candidates V1 caught before commit, over all that were made.

    The two halves measure different things and are reported separately:
    *attempts* measure the model, *interception* measures this system.
    """
    rejected = [f for f in session.flags if f.code == "fabricated_goal"]
    slipped = audit_committed_goals(session)
    attempted = len(rejected) + len(slipped)
    return {
        "attempted_by_model": attempted,
        "rejected_pre_commit": len(rejected),
        "slipped_through": len(slipped),
        "rate": (len(rejected) / attempted) if attempted else None,
        "passes": not slipped,
    }


# --------------------------------------------------------------------------
# Coverage, efficiency, maturity
# --------------------------------------------------------------------------


def coverage_completeness(session: SessionState) -> dict[str, Any]:
    resolved = [
        dimension
        for dimension in DIMENSIONS
        if session.coverage_status.get(dimension, "uncovered") != "uncovered"
    ]
    return {
        "resolved": len(resolved),
        "total": len(DIMENSIONS),
        "detail": {d: session.coverage_status.get(d, "uncovered") for d in DIMENSIONS},
        "parked": [d for d in DIMENSIONS if session.is_parked(d)],
        "passes": len(resolved) == len(DIMENSIONS),
    }


def elicitation_efficiency(session: SessionState) -> dict[str, Any]:
    turns = len(session.stakeholder_turns())
    return {
        "stakeholder_turns": turns,
        "requirements": len(session.requirements),
        "requirements_per_turn": (
            round(len(session.requirements) / turns, 2) if turns else 0.0
        ),
        "target": 12,
        "passes": turns <= 12,
    }


def maturity_quality(
    session: SessionState, expected_level: str | None
) -> dict[str, Any]:
    maturity = session.maturity
    if maturity is None:
        return {"proposed": None, "description_shown": False, "passes": False}
    description = maturity.description_shown_to_human.strip()
    correct = expected_level is None or maturity.value == expected_level
    return {
        "proposed": maturity.value,
        "expected": expected_level,
        "correct_level": correct,
        "human_response": maturity.human_response,
        "description_shown": bool(description),
        "description_chars": len(description),
        # A description that just repeats the label is not a description.
        "jargon_free": not any(
            term in description for term in ("4R", "dimension", "intelligence layer")
        ),
        "passes": correct and bool(description),
    }


# --------------------------------------------------------------------------
# Well-formedness
# --------------------------------------------------------------------------


def well_formedness(session: SessionState) -> dict[str, Any]:
    """Fields populated, "shall" form, and no solution named in the text."""
    problems: list[str] = []
    good = 0
    for requirement in session.requirements:
        faults = []
        if " shall " not in requirement.text.lower():
            faults.append("not in shall form")
        if not requirement.text.strip():
            faults.append("empty text")
        if not requirement.dimension or not requirement.verifyMethod:
            faults.append("missing fields")
        named = [
            word
            for word in BANNED_SOLUTION_WORDS
            if word in requirement.text.lower()
        ]
        if named:
            faults.append(f"names a solution: {', '.join(named)}")
        if faults:
            problems.append(f"{requirement.id}: {'; '.join(faults)}")
        else:
            good += 1
    total = len(session.requirements)
    return {
        "well_formed": good,
        "total": total,
        "rate": (good / total) if total else None,
        "problems": problems,
        "passes": total == 0 or (good / total) >= 0.95,
    }


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def _match(expected_text: str, committed: Iterable[Requirement]) -> Requirement | None:
    expected_words = content_words(expected_text)
    normalised = _normalise(expected_text)
    best: tuple[float, Requirement] | None = None
    for requirement in committed:
        if _normalise(requirement.text) == normalised:
            return requirement
        score = overlap_ratio(expected_words, content_words(requirement.text))
        if best is None or score > best[0]:
            best = (score, requirement)
    if best is not None and best[0] >= TEXT_MATCH_THRESHOLD:
        return best[1]
    return None


def extraction_fidelity(
    session: SessionState, ground_truth: dict[str, Any]
) -> dict[str, Any]:
    """Committed requirements against the published FDM set.

    Matching is by requirement *text*, never by id: the agent numbers
    requirements in the order they are committed, which is the order the
    stakeholder happened to raise them, and that will not be the paper's order.

    Goal values are compared on their numeric content, so "Accuracy >=90%" and
    ">= 90 %" agree while "90%" and "95%" do not.
    """
    expected = ground_truth.get("requirements", [])
    rows: list[dict[str, Any]] = []
    matched = 0
    goals_expected = 0
    goals_exact = 0

    for item in expected:
        found = _match(item["text"], session.requirements)
        row: dict[str, Any] = {
            "expected_id": item["id"],
            "expected_text": item["text"],
            "matched_id": found.id if found else None,
            "type_ok": bool(found and found.type == item["type"]),
            "dimension_ok": bool(found and found.dimension == item["dimension"]),
            "goal_ok": None,
            "ref_ok": None,
        }
        if found:
            matched += 1
        if item.get("designGoal"):
            goals_expected += 1
            if found and found.designGoal:
                same = numeric_tokens(found.designGoal) == numeric_tokens(
                    item["designGoal"]
                )
                row["goal_ok"] = same
                row["found_goal"] = found.designGoal
                if same:
                    goals_exact += 1
            else:
                row["goal_ok"] = False
        if found and found.designGoal:
            turn = session.turn(found.stakeholder_utterance_ref)
            row["ref_ok"] = bool(turn and turn.role == "stakeholder")
        rows.append(row)

    return {
        "matched": matched,
        "expected": len(expected),
        "target": 10,
        "goals_expected": goals_expected,
        "goals_exact": goals_exact,
        "goal_exact_rate": (goals_exact / goals_expected) if goals_expected else None,
        "rows": rows,
        "passes": matched >= 10 and goals_exact == goals_expected,
        "caveat": (
            "Text matching here is lexical. Semantic equivalence and clarity "
            "scoring require the human rubric adapted from Ronanki et al. "
            "(2023) and are not automated."
        ),
    }


def defect_detection(
    session: SessionState, ground_truth: dict[str, Any]
) -> dict[str, Any]:
    """Were the two real defects in the published set surfaced?"""
    codes = {flag.code for flag in session.flags}
    results = []
    for defect in ground_truth.get("seeded_defects", []):
        results.append(
            {
                "code": defect["code"],
                "requirements": defect["requirements"],
                "detected": defect["code"] in codes,
            }
        )
    detected = sum(1 for row in results if row["detected"])
    return {
        "detected": detected,
        "expected": len(results),
        "rows": results,
        "passes": detected == len(results),
    }


# --------------------------------------------------------------------------
# Edge rules Q1 / Q2 / Q3
# --------------------------------------------------------------------------


def edge_rules(session: SessionState) -> dict[str, Any]:
    """What each edge rule actually did in this run.

    Reported per persona; only the persona built to exercise a rule is
    expected to trip it.
    """
    parked = [d for d in DIMENSIONS if session.is_parked(d)]
    return {
        "q1_needs_clarification": [
            requirement.id
            for requirement in session.requirements
            if requirement.status == "needs_clarification"
        ],
        "q1_all_have_null_goals": all(
            requirement.designGoal is None
            for requirement in session.requirements
            if requirement.status == "needs_clarification"
        ),
        "q2_status": session.status,
        "q3_parked": parked,
        "q3_attempts_on_parked": {d: session.attempts.get(d, 0) for d in parked},
        "q3_respects_cap": all(
            session.attempts.get(d, 0) <= ATTEMPT_CAP for d in DIMENSIONS
        ),
    }


def hitl_conformance(session: SessionState) -> dict[str, Any]:
    """No autonomous path around any checkpoint (spec section 4)."""
    duplicates = [f for f in session.flags if f.code == "duplicate_obligation"]
    return {
        "hitl1_maturity_confirmed": session.maturity is not None,
        "hitl2_every_answer_from_human": len(session.stakeholder_turns()) > 0,
        "hitl3_open_flags_at_end": len(session.open_flags()),
        "duplicates_resolved_by_human_only": all(
            flag.resolved_by == "human"
            for flag in duplicates
            if flag.status == "resolved"
        ),
        "passes": (
            session.maturity is not None
            and (session.status == "partial" or not session.open_flags())
        ),
    }


# --------------------------------------------------------------------------
# Robustness / ops
# --------------------------------------------------------------------------


def robustness(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse-retry rate, latency and tokens, from the model call log."""
    extractions = [r for r in records if r.get("call") == "extract"]
    retried = [r for r in extractions if (r.get("parse_retries") or 0) > 0]
    latencies = [
        float(r["latency_ms"]) for r in records if isinstance(r.get("latency_ms"), (int, float))
    ]
    tokens = [
        int(r["input_tokens"]) + int(r.get("output_tokens") or 0)
        for r in records
        if isinstance(r.get("input_tokens"), int)
    ]

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(int(fraction * len(ordered)), len(ordered) - 1)
        return round(ordered[index], 1)

    return {
        "calls": len(records),
        "extractions": len(extractions),
        "parse_retry_rate": (len(retried) / len(extractions)) if extractions else None,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "tokens_total": sum(tokens) if tokens else None,
        "tokens_mean": round(statistics.mean(tokens), 1) if tokens else None,
        "passes": (not extractions) or (len(retried) / len(extractions)) < 0.10,
    }


def load_ground_truth(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "TEXT_MATCH_THRESHOLD",
    "coverage_completeness",
    "defect_detection",
    "edge_rules",
    "elicitation_efficiency",
    "extraction_fidelity",
    "fabricated_goal_rate",
    "hitl_conformance",
    "interception_rate",
    "load_ground_truth",
    "maturity_quality",
    "robustness",
    "well_formedness",
]
