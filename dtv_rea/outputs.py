"""Writing a finished session to disk (spec section 1.2, ``finalize``).

Four files per session, all under ``./runs/<session_id>/``:

``session.json``     the full state, exactly as the agent held it
``requirements.md``  the human-readable document, grouped by dimension
``snapshots.jsonl``  one row per turn - this log *is* the evaluation dataset
``llm_calls.jsonl``  one row per model call (written by the model, not here)

Every path is built with :class:`pathlib.Path` and every file is written as
explicit UTF-8, so the same bytes land on macOS and on Windows.
"""

from __future__ import annotations

import json
from pathlib import Path

from dtv_rea.settings import DIMENSION_LABELS, DIMENSIONS, session_dir
from dtv_rea.state import Requirement, SessionState

PARTIAL_BANNER = (
    "> **STATUS: PARTIAL.** The interview was stopped before it finished. "
    "Topics may be unexplored, numbers may be missing, and flagged items may "
    "be unresolved. Do not treat this as a completed requirements document."
)


def write_session_json(session: SessionState, directory: Path) -> Path:
    path = directory / "session.json"
    payload = session.model_dump(mode="json")
    # "status" first, so a partial run is obvious in the first line of the file.
    ordered = {"status": payload.pop("status")}
    ordered.update(payload)
    path.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def write_snapshots(session: SessionState, directory: Path) -> Path:
    path = directory / "snapshots.jsonl"
    lines = [
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False)
        for snapshot in session.snapshots
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _quote_for(session: SessionState, requirement: Requirement) -> str:
    """The exact words the number came from. This is the audit trail."""
    turn = session.turn(requirement.stakeholder_utterance_ref)
    if turn is None:
        return "_no source turn recorded_"
    return f'turn {turn.index}, stakeholder: "{turn.text.strip()}"'


def _render_requirement(session: SessionState, requirement: Requirement) -> list[str]:
    lines = [f"#### {requirement.id} - {requirement.text}", ""]
    lines.append(f"- Type: {requirement.type}")
    lines.append(
        f"- Verification method: {requirement.verifyMethod} "
        f"({'confirmed' if requirement.verifyMethod_confirmed else 'proposed'})"
    )
    lines.append(f"- Priority: {requirement.priority}")
    if requirement.designGoal is not None:
        lines.append(f"- **Design goal: {requirement.designGoal}**")
        lines.append(f"  - Source: {_quote_for(session, requirement)}")
    elif requirement.type == "performance":
        if requirement.status == "needs_clarification":
            lines.append(
                "- Design goal: **NOT SET - the stakeholder did not have a "
                "value.** This requirement cannot be verified until one is "
                "supplied."
            )
        else:
            lines.append("- Design goal: none recorded")
    if requirement.rationale:
        lines.append(f"- Rationale: {requirement.rationale}")
    if requirement.status != "complete":
        lines.append(f"- Status: **{requirement.status}**")
    lines.append("")
    return lines


def render_requirements_md(session: SessionState) -> str:
    lines: list[str] = []
    lines.append(f"# DT requirements - session `{session.session_id}`")
    lines.append("")
    if session.status == "partial":
        lines.append(PARTIAL_BANNER)
        lines.append("")

    lines.append(f"- Status: **{session.status}**")
    lines.append(f"- Requirements captured: {len(session.requirements)}")
    lines.append(f"- Stakeholder turns: {len(session.stakeholder_turns())}")
    lines.append(f"- Open flags: {len(session.open_flags())}")
    lines.append("")

    lines.append("## Purpose")
    lines.append("")
    if session.purpose is None:
        lines.append("_Not captured._")
    else:
        lines.append(session.purpose.statement)
        if session.purpose.rationale:
            lines.append("")
            lines.append(f"**Why:** {session.purpose.rationale}")
        if session.purpose.user_roles:
            lines.append("")
            lines.append(f"**Who uses it:** {', '.join(session.purpose.user_roles)}")
        if session.purpose.context_of_use:
            lines.append("")
            lines.append(f"**When:** {session.purpose.context_of_use}")
    lines.append("")

    lines.append("## Capability level")
    lines.append("")
    if session.maturity is None:
        lines.append("_Not agreed._")
    else:
        maturity = session.maturity
        lines.append(f"**{maturity.value}** ({maturity.human_response} by the stakeholder)")
        lines.append("")
        lines.append(f"What the stakeholder was shown: {maturity.description_shown_to_human}")
        lines.append("")
        lines.append(f"Agent reasoning: {maturity.agent_reasoning}")
        if maturity.overridden_to:
            lines.append("")
            lines.append(f"Overridden to **{maturity.overridden_to}** by the stakeholder.")
    lines.append("")

    lines.append("## Requirements")
    lines.append("")
    for dimension in DIMENSIONS:
        label = DIMENSION_LABELS.get(dimension, dimension)
        status = session.coverage_status.get(dimension, "uncovered")
        if session.is_parked(dimension):
            status = "parked (no usable answer after repeated attempts)"
        elif status == "not_applicable":
            confirming = session.not_applicable_confirmations.get(dimension)
            status = f"not applicable - ruled out by the stakeholder at turn {confirming}"

        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"_Coverage: {status}._")
        lines.append("")
        items = session.requirements_in(dimension)
        if not items:
            lines.append("_No requirements captured for this topic._")
            lines.append("")
            continue
        for requirement in items:
            lines.extend(_render_requirement(session, requirement))

    lines.append("## Open and resolved flags")
    lines.append("")
    if not session.flags:
        lines.append("_No validator findings._")
        lines.append("")
    else:
        for flag in session.flags:
            marker = "OPEN" if flag.status == "open" else "resolved"
            lines.append(f"- **[{marker}] {flag.code}** ({flag.severity}) - {flag.message}")
            if flag.resolution:
                lines.append(f"  - Resolution ({flag.resolved_by}): {flag.resolution}")
        lines.append("")

    lines.append("## Provenance note")
    lines.append("")
    lines.append(
        "Every design goal above is quoted back to the stakeholder turn it came "
        "from. A design goal that could not be traced to something the "
        "stakeholder actually said was rejected before it reached this "
        "document, and appears in the flag list instead. The agent does not "
        "supply numbers."
    )
    lines.append("")
    return "\n".join(lines)


def write_requirements_md(session: SessionState, directory: Path) -> Path:
    path = directory / "requirements.md"
    path.write_text(render_requirements_md(session), encoding="utf-8")
    return path


def write_all(session: SessionState, directory: Path | None = None) -> dict[str, Path]:
    """Write every output file for a session and return where each one went."""
    target = directory or session_dir(session.session_id)
    target.mkdir(parents=True, exist_ok=True)
    return {
        "session_json": write_session_json(session, target),
        "requirements_md": write_requirements_md(session, target),
        "snapshots": write_snapshots(session, target),
        "directory": target,
    }


__all__ = [
    "PARTIAL_BANNER",
    "render_requirements_md",
    "write_all",
    "write_requirements_md",
    "write_session_json",
    "write_snapshots",
]
