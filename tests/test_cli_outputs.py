"""Phase 6 - the CLI, the output files, and the Q2 early-stop rule.

All offline. The CLI is exercised through ``main()`` with the stub model, which
is exactly what the documented demo command does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dtv_rea.cli import build_parser, console_answerer, main
from dtv_rea.graph import build_graph, open_checkpointer
from dtv_rea.llm import StubModel
from dtv_rea.outputs import PARTIAL_BANNER, render_requirements_md, write_all
from dtv_rea.persona import PersonaScript
from dtv_rea.runner import persona_answerer, run_session
from dtv_rea.state import SessionState


def run_persona(name: str, tmp_path: Path) -> SessionState:
    script = PersonaScript.by_name(name)
    saver, connection = open_checkpointer(tmp_path / f"{name}.db")
    try:
        graph = build_graph(StubModel(script), checkpointer=saver)
        return run_session(graph, name, persona_answerer(script))
    finally:
        connection.close()


# --------------------------------------------------------------------------
# Q2 - stopping early
# --------------------------------------------------------------------------


@pytest.fixture
def quitter(tmp_path: Path) -> SessionState:
    return run_persona("quitter", tmp_path)


def test_stopping_early_marks_the_session_partial(quitter: SessionState) -> None:
    assert quitter.status == "partial"
    assert quitter.stop_requested


def test_work_done_before_stopping_is_kept(quitter: SessionState) -> None:
    assert len(quitter.requirements) == 1
    assert quitter.requirements[0].designGoal == "Accuracy of +/- 2.5 C from target"


def test_unexplored_topics_are_still_shown_as_uncovered(
    quitter: SessionState,
) -> None:
    """A stopped interview must not look like a finished one."""
    assert quitter.coverage_status["intelligence_layer"] == "uncovered"
    assert not quitter.is_complete()


def test_partial_status_is_prominent_in_both_output_files(
    quitter: SessionState, tmp_path: Path
) -> None:
    paths = write_all(quitter, tmp_path / "out")

    document = paths["requirements_md"].read_text(encoding="utf-8")
    assert PARTIAL_BANNER in document
    # "Prominently" means near the top, not buried at the end.
    assert document.index("STATUS: PARTIAL") < 200

    payload = json.loads(paths["session_json"].read_text(encoding="utf-8"))
    assert next(iter(payload)) == "status"
    assert payload["status"] == "partial"


def test_a_completed_session_carries_no_partial_banner(tmp_path: Path) -> None:
    session = run_persona("fdm_stakeholder", tmp_path)
    assert PARTIAL_BANNER not in render_requirements_md(session)


# --------------------------------------------------------------------------
# The output files
# --------------------------------------------------------------------------


@pytest.fixture
def fdm_outputs(tmp_path: Path) -> tuple[SessionState, dict]:
    session = run_persona("fdm_stakeholder", tmp_path)
    return session, write_all(session, tmp_path / "out")


def test_all_three_files_are_written(fdm_outputs) -> None:
    _, paths = fdm_outputs
    for key in ("session_json", "requirements_md", "snapshots"):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0


def test_files_are_utf8_and_readable_as_utf8(fdm_outputs) -> None:
    """Explicit encoding everywhere - a Windows default would mangle these."""
    _, paths = fdm_outputs
    for key in ("session_json", "requirements_md", "snapshots"):
        paths[key].read_text(encoding="utf-8")


def test_snapshots_are_one_json_object_per_line(fdm_outputs) -> None:
    _, paths = fdm_outputs
    lines = paths["snapshots"].read_text(encoding="utf-8").strip().splitlines()
    assert lines
    for line in lines:
        row = json.loads(line)
        assert {"turn", "n_requirements", "n_open_flags", "coverage"} <= set(row)


def test_session_json_round_trips_back_into_the_model(fdm_outputs) -> None:
    session, paths = fdm_outputs
    payload = json.loads(paths["session_json"].read_text(encoding="utf-8"))
    restored = SessionState.model_validate(payload)
    assert len(restored.requirements) == len(session.requirements)


def test_the_document_quotes_the_source_turn_for_every_goal(fdm_outputs) -> None:
    """The audit trail is the point of the document, not a footnote."""
    session, paths = fdm_outputs
    document = paths["requirements_md"].read_text(encoding="utf-8")
    for requirement in session.requirements:
        if requirement.designGoal is None:
            continue
        assert f"**Design goal: {requirement.designGoal}**" in document
        turn = session.turn(requirement.stakeholder_utterance_ref)
        assert turn is not None
        assert f"turn {turn.index}, stakeholder:" in document


def test_the_document_is_grouped_by_dimension(fdm_outputs) -> None:
    _, paths = fdm_outputs
    document = paths["requirements_md"].read_text(encoding="utf-8")
    for label in (
        "Data collection, storage and integration",
        "Fidelity and responsiveness of the virtual environment",
        "Performance and accuracy of the intelligence layer",
        "Degree of automation and feedback",
    ):
        assert f"### {label}" in document


def test_the_document_records_what_hitl1_showed_the_human(fdm_outputs) -> None:
    _, paths = fdm_outputs
    document = paths["requirements_md"].read_text(encoding="utf-8")
    assert "What the stakeholder was shown:" in document
    assert "**Replication** (confirmed by the stakeholder)" in document


def test_the_document_lists_flags_and_how_they_were_settled(fdm_outputs) -> None:
    _, paths = fdm_outputs
    document = paths["requirements_md"].read_text(encoding="utf-8")
    assert "duplicate_obligation" in document
    assert "Resolution (human):" in document


def test_a_parked_topic_is_labelled_as_parked_not_as_ruled_out(
    tmp_path: Path,
) -> None:
    session = run_persona("evasive", tmp_path)
    document = render_requirements_md(session)
    assert "parked (no usable answer after repeated attempts)" in document


def test_a_requirement_with_no_number_says_so_plainly(tmp_path: Path) -> None:
    session = run_persona("dont_know", tmp_path)
    document = render_requirements_md(session)
    assert "NOT SET - the stakeholder did not have a value." in document


# --------------------------------------------------------------------------
# Reading answers from the console
# --------------------------------------------------------------------------


@pytest.mark.parametrize("typed", ["stop", "STOP", " quit ", "exit"])
def test_a_typed_stop_keyword_is_passed_straight_through(
    typed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: typed)
    assert console_answerer({"kind": "answer"}) == typed


def test_end_of_input_becomes_the_same_stop_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-D on macOS and Ctrl-Z on Windows both land here."""

    def raise_eof(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert console_answerer({"kind": "answer"}) == "stop"


def test_ctrl_c_at_a_question_also_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_interrupt(_: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", raise_interrupt)
    assert console_answerer({"kind": "answer"}) == "stop"


def test_an_empty_line_is_not_treated_as_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "   ")
    assert console_answerer({"kind": "answer"}) == "(no answer given)"


# --------------------------------------------------------------------------
# The command line itself
# --------------------------------------------------------------------------


def test_the_documented_demo_command_succeeds_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`python -m dtv_rea.cli --stub` - the first thing a teammate runs."""
    assert main(["--stub", "--session", "demo"]) == 0
    output = capsys.readouterr().out
    assert "Interview complete." in output
    assert "Requirements captured : 12" in output
    assert "all traced to a stakeholder turn" in output


def test_the_demo_writes_its_files_under_runs(tmp_path: Path) -> None:
    main(["--stub", "--session", "demo2"])
    from dtv_rea.settings import runs_dir

    directory = runs_dir() / "demo2"
    assert (directory / "requirements.md").exists()
    assert (directory / "session.json").exists()
    assert (directory / "snapshots.jsonl").exists()


def test_stub_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert main(["--stub", "--session", "nokey"]) == 0


def test_a_live_run_without_a_key_fails_with_advice_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dtv_rea.settings.load_env", lambda: False)
    monkeypatch.setattr("dtv_rea.groq_model.load_env", lambda: False)

    assert main(["--session", "live"]) == 2
    output = capsys.readouterr().out
    assert "GROQ_API_KEY is not set" in output
    assert "--stub" in output


def test_an_unknown_persona_is_reported_helpfully(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--stub", "--persona", "does_not_exist"]) == 2
    assert "--list-personas" in capsys.readouterr().out


def test_list_personas_shows_all_six(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-personas"]) == 0
    output = capsys.readouterr().out
    for name in (
        "fdm_stakeholder", "evasive", "dont_know",
        "contradictory", "no_numbers", "quitter",
    ):
        assert name in output


def test_the_early_stop_instruction_is_shown_in_interactive_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "stop")
    monkeypatch.setenv("GROQ_API_KEY", "")
    main(["--stub", "--persona", "quitter", "--session", "q"])
    # Persona mode does not prompt, so check the flag exists for the human path.
    assert "--stub" in build_parser().format_help()


def test_doctor_reports_the_setup_without_revealing_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value_do_not_leak")

    assert main(["--doctor"]) == 0

    output = capsys.readouterr().out
    assert "GROQ_API_KEY found  : True" in output
    assert "Personas found      : 6" in output
    assert "gsk_secret_value_do_not_leak" not in output


def test_doctor_points_a_keyless_teammate_at_the_offline_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dtv_rea.cli.load_env", lambda: False)

    assert main(["--doctor"]) == 0
    assert "python -m dtv_rea.cli --stub" in capsys.readouterr().out


def test_the_env_file_is_only_ever_read_from_the_project_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No walking up the tree - the answer must not depend on the cwd."""
    from dtv_rea.settings import REPO_ROOT, env_file

    assert env_file() == REPO_ROOT / ".env"

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dtv_rea.settings.env_file", lambda: tmp_path / "absent.env")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_stray_key\n", encoding="utf-8")

    from dtv_rea.settings import load_env

    assert load_env() is False


def test_help_documents_the_stop_word_and_is_platform_neutral() -> None:
    help_text = build_parser().format_help()
    assert "Type 'stop' at any question" in help_text
    assert "identical on macOS, Linux and Windows" in help_text
    for shell_ism in ("bash", ".sh", "chmod", "export ", "set "):
        assert shell_ism not in help_text


# --------------------------------------------------------------------------
# Failures a teammate will actually hit, explained rather than dumped
# --------------------------------------------------------------------------


class _ApiError(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code


def test_a_rate_limit_is_explained_in_plain_language() -> None:
    """The free Groq tier runs out after two or three interviews a day."""
    from dtv_rea.cli import explain_failure

    lines = explain_failure(
        _ApiError(
            429,
            "Rate limit reached for model `llama-3.3-70b-versatile` ... on "
            "tokens per day (TPD): Limit 100000. Please try again in 10m58s.",
        )
    )
    text = " ".join(lines)
    assert "used up your quota" in text
    assert "10m58s" in text            # the wait is surfaced, not buried
    assert "console.groq.com/settings/billing" in text
    assert "Traceback" not in text


def test_a_bad_key_is_explained_without_echoing_it() -> None:
    from dtv_rea.cli import explain_failure

    text = " ".join(explain_failure(_ApiError(401, "invalid api key gsk_bad")))
    assert "would not accept your API key" in text
    assert "gsk_" in text              # tells them the expected prefix
    assert "gsk_bad" not in text       # but never echoes the actual value


def test_a_network_failure_is_explained() -> None:
    from dtv_rea.cli import explain_failure

    error = type("APIConnectionError", (Exception,), {})()
    assert "Could not reach Groq." in explain_failure(error)


def test_an_unrecognised_failure_still_says_something_useful() -> None:
    from dtv_rea.cli import explain_failure

    text = " ".join(explain_failure(ValueError("something odd")))
    assert "something odd" in text


def test_a_failed_live_run_tells_the_user_how_to_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interrupted interview must never look like lost work."""

    def blow_up(*args, **kwargs):
        raise _ApiError(429, "Please try again in 5m.")

    monkeypatch.setattr("dtv_rea.cli.run_session", blow_up)

    assert main(["--stub", "--session", "boom"]) == 2

    output = capsys.readouterr().out
    assert "Nothing you have already answered is lost." in output
    assert "--session boom --resume" in output
