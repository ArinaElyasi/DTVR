"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtv_rea.state import Requirement, SessionState, new_session


@pytest.fixture(autouse=True)
def offline_and_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs with no API key and writes only into ``tmp_path``.

    This is what makes "the deterministic core is testable offline" a fact the
    suite proves rather than a claim in a README.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("DTV_REA_RUNS_DIR", str(tmp_path / "runs"))


@pytest.fixture
def session() -> SessionState:
    return new_session("test-session")


def make_requirement(
    requirement_id: str,
    text: str,
    dimension: str = "data_collection_integration",
    requirement_type: str = "functional",
    design_goal: str | None = None,
    ref: int | None = None,
    status: str = "complete",
) -> Requirement:
    """Build a committed requirement without repeating boilerplate."""
    return Requirement(
        id=requirement_id,
        text=text,
        type=requirement_type,  # type: ignore[arg-type]
        dimension=dimension,  # type: ignore[arg-type]
        verifyMethod="Test" if requirement_type == "performance" else "Inspection",
        designGoal=design_goal,
        designGoal_provenance="stated" if design_goal is not None else None,
        stakeholder_utterance_ref=ref,
        status=status,  # type: ignore[arg-type]
    )
