"""The command-line interface: ``python -m dtv_rea.cli``.

The same command works identically on macOS, Linux and Windows. There is no
shell script to invoke, no entry point to put on ``PATH``, and no platform
branch anywhere in this file.

Three cross-platform details worth knowing about:

* **Console encoding.** The requirements carry characters like the degree sign
  and plus-or-minus. On a Windows console still running a legacy code page,
  printing them would raise ``UnicodeEncodeError`` and kill the interview
  mid-answer. Both streams are reconfigured to UTF-8 with replacement, so at
  worst a character renders as ``?``.

* **Stopping early (Q2).** The primary signal is the typed word ``stop``,
  because the end-of-input key differs by platform: Ctrl-D on macOS and Linux,
  Ctrl-Z then Enter on Windows. Both are accepted, and both produce the same
  clearly-marked partial output.

* **Paths.** Every path here is a :class:`pathlib.Path`.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dtv_rea import __version__
from dtv_rea.graph import build_graph, open_checkpointer
from dtv_rea.llm import CallLogger, StubModel
from dtv_rea.outputs import write_all
from dtv_rea.persona import PersonaError, PersonaScript
from dtv_rea.runner import persona_answerer, run_session
from dtv_rea.settings import (
    GROQ_API_KEY_VAR,
    MODEL_NAME,
    REPO_ROOT,
    checkpoint_db,
    env_file,
    load_env,
    runs_dir,
    session_dir,
)
from dtv_rea.state import SessionState, Turn
from dtv_rea.validator import audit_committed_goals

RULE = "-" * 68


def _use_utf8() -> None:
    """Make both console streams UTF-8 on every platform."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed/odd streams
            pass


def _say(text: str = "") -> None:
    print(text, flush=True)


def _default_session_id() -> str:
    return f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


# --------------------------------------------------------------------------
# Answering from the console
# --------------------------------------------------------------------------


def console_answerer(payload: dict[str, Any]) -> str:
    """Read one stakeholder answer from the terminal.

    End-of-input (Ctrl-D on macOS/Linux, Ctrl-Z then Enter on Windows) and
    Ctrl-C are both folded into the same "stop" the typed keyword produces, so
    there is exactly one early-exit path to reason about.
    """
    kind = payload.get("kind")
    if kind == "maturity":
        _say()
        _say("Does that match what you need? Say yes, or tell me what is wrong.")
    elif kind == "flag":
        _say()
        _say("(Your answer settles this. Keeping things as they are is fine.)")

    try:
        answer = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        _say()
        _say("(Input closed - stopping here and saving what we have.)")
        return "stop"

    if not answer.strip():
        return "(no answer given)"
    return answer


def explain_failure(error: BaseException) -> list[str]:
    """Turn an API failure into something a non-expert can act on.

    A raw traceback is not an error message. These are the four failures that
    actually happen in normal use, and each one gets told what to do next.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    name = type(error).__name__
    text = str(error)

    if status == 429 or "RateLimit" in name:
        wait = re.search(r"try again in ([0-9hms.]+)", text)
        lines = [
            "Groq says you have used up your quota for now.",
            "",
            "Free Groq accounts have a daily token allowance, and one full "
            "interview uses roughly 35,000 tokens - so about two or three "
            "interviews a day.",
        ]
        if wait:
            lines.append(f"It suggests trying again in {wait.group(1)}.")
        lines += [
            "",
            "You can either wait, or raise the allowance at",
            "  https://console.groq.com/settings/billing",
        ]
        return lines

    if status in (401, 403) or "Authentication" in name or "Permission" in name:
        return [
            "Groq would not accept your API key.",
            "",
            "Open the .env file and check the key is complete and current.",
            "Groq keys begin with 'gsk_'. You can make a new one at",
            "  https://console.groq.com/keys",
        ]

    if "Connection" in name or "Timeout" in name or "APIConnection" in name:
        return [
            "Could not reach Groq.",
            "",
            "Check your internet connection and try again. If you are on a "
            "work network, a firewall may be blocking api.groq.com.",
        ]

    return [f"Something went wrong talking to the model: {name}", "", text[:400]]


def print_agent_turn(turn: Turn) -> None:
    _say()
    _say(RULE)
    _say(turn.text)


def _echoing(answerer: Any) -> Any:
    """Print a scripted stakeholder's answers, so a persona run reads as a
    conversation rather than as a monologue."""

    def answer(payload: dict[str, Any]) -> str:
        reply = answerer(payload)
        _say()
        _say(f"> {reply}")
        return reply

    return answer


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_summary(session: SessionState, directory: Path) -> None:
    _say()
    _say(RULE)
    if session.status == "partial":
        _say("STOPPED EARLY - the output below is marked PARTIAL.")
    else:
        _say("Interview complete.")
    _say(RULE)
    _say(f"  Requirements captured : {len(session.requirements)}")
    _say(f"  Stakeholder turns     : {len(session.stakeholder_turns())}")
    _say(f"  Open flags            : {len(session.open_flags())}")

    covered = sum(
        1 for value in session.coverage_status.values() if value != "uncovered"
    )
    _say(f"  Topics resolved       : {covered}/{len(session.coverage_status)}")

    fabricated = audit_committed_goals(session)
    goals = sum(1 for r in session.requirements if r.designGoal is not None)
    _say(f"  Design goals recorded : {goals}, all traced to a stakeholder turn"
         if not fabricated else
         f"  Design goals recorded : {goals}, {len(fabricated)} UNTRACEABLE")

    _say()
    _say("Your files are in:")
    _say(f"  {directory}")
    for name in ("requirements.md", "session.json", "snapshots.jsonl"):
        if (directory / name).exists():
            _say(f"    - {name}")
    _say()
    _say("Open requirements.md first - that is the readable document.")


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dtv_rea.cli",
        description=(
            "Interview a stakeholder and produce a digital-twin requirements "
            "document. The agent never invents a number."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (identical on macOS, Linux and Windows):\n"
            "  python -m dtv_rea.cli --stub            offline demo, no API key\n"
            "  python -m dtv_rea.cli                   live interview\n"
            "  python -m dtv_rea.cli --session my-run --resume\n"
            "\n"
            "Type 'stop' at any question to finish early and keep what you have."
        ),
    )
    parser.add_argument(
        "--session",
        metavar="ID",
        help="Name for this interview. Output goes to ./runs/<ID>/. "
        "Defaults to a timestamp.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline scripted model. Needs no API key and no internet.",
    )
    parser.add_argument(
        "--persona",
        metavar="NAME",
        help="Answer automatically from a scripted stakeholder in data/personas "
        "(name or path). Implied by --stub.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a session that was left unfinished.",
    )
    parser.add_argument(
        "--list-personas", action="store_true", help="Show the scripted personas."
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check the setup: where files will go, and whether a key was found.",
    )
    parser.add_argument("--version", action="version", version=f"DTV-REA {__version__}")
    return parser


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _use_utf8()
    args = build_parser().parse_args(argv)

    if args.doctor:
        _describe_environment()
        return 0

    if args.list_personas:
        available = PersonaScript.available()
        if not available:
            _say("No personas found in data/personas.")
            return 0
        _say("Scripted personas:")
        for name in available:
            try:
                script = PersonaScript.by_name(name)
            except PersonaError:  # pragma: no cover - malformed file
                _say(f"  {name}  (could not be read)")
                continue
            _say(f"  {name}")
            if script.description:
                _say(f"      {script.description}")
        return 0

    session_id = args.session or _default_session_id()

    # --stub with no persona would have nothing to extract from, so the demo
    # everyone runs first defaults to the ground-truth case.
    persona_name = args.persona or ("fdm_stakeholder" if args.stub else None)

    script: PersonaScript | None = None
    if persona_name:
        try:
            script = PersonaScript.by_name(persona_name)
        except PersonaError as error:
            _say(f"Could not load persona '{persona_name}': {error}")
            _say("Run with --list-personas to see what is available.")
            return 2

    if args.stub:
        model: Any = StubModel(script)
        _say("Running the offline scripted model - no API key, no internet.")
    else:
        from dtv_rea.groq_model import GroqModel, MissingApiKey

        logger = CallLogger(session_dir(session_id) / "llm_calls.jsonl")
        try:
            model = GroqModel(logger=logger)
        except MissingApiKey as error:
            _say("Cannot start a live interview.")
            _say()
            _say(str(error))
            return 2
        _say(f"Running live against {MODEL_NAME} on Groq.")

    if script is not None:
        _say(f"Answers come from the scripted persona '{script.name}'.")
        answerer = _echoing(persona_answerer(script))
    else:
        answerer = console_answerer

    _say()
    _say(f"Session: {session_id}")
    _say(f"Output : {session_dir(session_id)}")
    if script is None:
        _say("Type 'stop' at any point to finish early and keep what you have.")

    saver, connection = open_checkpointer()
    try:
        graph = build_graph(model, checkpointer=saver)
        session = run_session(
            graph,
            session_id,
            answerer,
            resume=args.resume,
            on_agent_turn=print_agent_turn,
        )
    except KeyboardInterrupt:  # pragma: no cover - needs a real terminal
        _say()
        _say("Interrupted. The interview is saved and can be picked up with:")
        _say(f"  python -m dtv_rea.cli --session {session_id} --resume")
        return 130
    except Exception as error:
        # Nothing is lost: every answer so far is already in the checkpoint
        # database, so the interview can be picked up exactly where it stopped.
        _say()
        _say(RULE)
        for line in explain_failure(error):
            _say(line)
        _say(RULE)
        _say()
        _say("Nothing you have already answered is lost. Pick up where you")
        _say("left off with:")
        _say(f"  python -m dtv_rea.cli --session {session_id} --resume")
        return 2
    finally:
        connection.close()

    paths = write_all(session)
    print_summary(session, paths["directory"])
    return 0


def _describe_environment() -> None:
    """Answer the questions the troubleshooting section asks, from one command.

    Reports only *whether* a key was found. The key itself is never printed.
    """
    key_found = load_env()
    _say(f"DTV-REA {__version__}")
    _say(f"  Python              : {sys.version.split()[0]}")
    _say(f"  Project folder      : {REPO_ROOT}")
    _say(f"  Output goes to      : {runs_dir()}")
    _say(f"  Checkpoints         : {checkpoint_db()}")
    _say(f"  Personas found      : {len(PersonaScript.available())}")
    _say(f"  .env file present   : {env_file().exists()}  ({env_file()})")
    _say(f"  {GROQ_API_KEY_VAR} found  : {key_found}")
    _say()
    if key_found:
        _say("Ready for a live interview:  python -m dtv_rea.cli")
    else:
        _say("No API key found, so live interviews are unavailable.")
        _say("Everything offline still works:  python -m dtv_rea.cli --stub")


if __name__ == "__main__":
    raise SystemExit(main())
