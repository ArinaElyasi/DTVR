"""Phase 5 - the Groq port's parse, repair and retry paths.

No network, no API key. The model is driven through an injected transport, so
every branch that matters offline - malformed JSON, the retry ladder, the
give-up-and-park path, the backoff predicate - is exercised for real.
"""

from __future__ import annotations

import pytest

from dtv_rea.groq_model import (
    GroqModel,
    JsonParseError,
    MissingApiKey,
    extraction_from_payload,
    is_transient,
    maturity_from_payload,
    parse_json_object,
    strip_fences,
)
from dtv_rea.llm import CallLogger, Focus
from dtv_rea.settings import MAX_PARSE_RETRIES
from dtv_rea.state import Purpose, SessionState

GOOD_JSON = """
{"requirements": [{"text": "The DT shall collect bed and nozzle temperature continuously",
  "type": "performance", "dimension": "data_collection_integration",
  "designGoal": "Accuracy of +/- 2.5 C from target", "designGoal_provenance": "stated",
  "stakeholder_utterance_ref": 3, "priority": "must", "rationale": "drift ruins prints",
  "status": "complete", "goal_target_id": null}], "not_applicable": [], "note": ""}
"""


def transport_returning(*replies: str):
    """A fake transport that hands back canned replies in order."""
    calls: list[str] = []

    def transport(system: str, user: str, temperature: float, json_mode: bool) -> str:
        calls.append(user)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


@pytest.fixture
def state() -> SessionState:
    session = SessionState(session_id="groq")
    session.purpose = Purpose(statement="Catch clogs mid-print.")
    session.add_turn("agent", "How accurate?")
    session.add_turn("stakeholder", "Within plus or minus 2.5 degrees C.")
    return session


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_clean_json_parses() -> None:
    assert parse_json_object('{"level": "Replication"}') == {"level": "Replication"}


def test_a_code_fence_is_stripped() -> None:
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_the_object_is_tolerated() -> None:
    reply = 'Sure! Here is the JSON you asked for:\n{"a": 1}\nLet me know if that helps.'
    assert parse_json_object(reply) == {"a": 1}


@pytest.mark.parametrize(
    "reply",
    ["", "   ", "I cannot help with that.", "{not json at all", "[1, 2, 3]", "null"],
)
def test_genuinely_unreadable_replies_raise(reply: str) -> None:
    """Guessing at malformed output is how invented values get in."""
    with pytest.raises(JsonParseError):
        parse_json_object(reply)


# --------------------------------------------------------------------------
# Coercion into the schema
# --------------------------------------------------------------------------


def test_a_well_formed_extraction_round_trips() -> None:
    result = extraction_from_payload(parse_json_object(GOOD_JSON), turn_index=3)
    assert len(result.requirements) == 1
    candidate = result.requirements[0]
    assert candidate.designGoal == "Accuracy of +/- 2.5 C from target"
    assert candidate.designGoal_provenance == "stated"
    assert candidate.stakeholder_utterance_ref == 3


def test_an_uncitable_goal_is_pinned_to_the_turn_so_v1_can_judge_it() -> None:
    """Dropping the citation would hide a fabrication instead of catching it."""
    payload = {
        "requirements": [
            {
                "text": "The DT shall detect clogs using vibration data",
                "type": "performance",
                "dimension": "intelligence_layer",
                "designGoal": "Accuracy >=95%",
            }
        ]
    }
    candidate = extraction_from_payload(payload, turn_index=7).requirements[0]
    assert candidate.stakeholder_utterance_ref == 7


def test_an_invented_provenance_is_normalised_not_accepted() -> None:
    payload = {
        "requirements": [
            {
                "text": "The DT shall hold temperature",
                "type": "performance",
                "dimension": "data_collection_integration",
                "designGoal": "+/- 2.5 C",
                "designGoal_provenance": "agent_derived",
            }
        ]
    }
    candidate = extraction_from_payload(payload, turn_index=1).requirements[0]
    assert candidate.designGoal_provenance == "stated"


def test_a_requirement_carrying_a_number_is_typed_performance_whatever_the_model_said() -> None:
    """Observed live: Llama 3.3 attaches goals to "functional" requirements.

    DTV section 3.2 ties design goals to performance requirements, so the type
    follows from the goal. Left uncorrected this gives the requirement the
    wrong verification method and hides it from the missing-goal check.
    """
    payload = {
        "requirements": [
            {
                "text": "The DT shall collect bed and nozzle temperature readings",
                "type": "functional",
                "dimension": "data_collection_integration",
                "designGoal": "+/- 2.5 degrees C of target",
                "stakeholder_utterance_ref": 5,
            }
        ]
    }
    candidate = extraction_from_payload(payload, turn_index=5).requirements[0]
    assert candidate.type == "performance"
    assert candidate.default_verify_method() == "Test"


def test_a_requirement_with_no_number_keeps_the_type_the_model_gave_it() -> None:
    payload = {
        "requirements": [
            {
                "text": "The DT shall pause the print when it spots a clog",
                "type": "functional",
                "dimension": "automation_feedback",
                "designGoal": None,
            }
        ]
    }
    candidate = extraction_from_payload(payload, turn_index=1).requirements[0]
    assert candidate.type == "functional"
    assert candidate.default_verify_method() == "Inspection"


def test_an_empty_goal_string_becomes_a_real_absence() -> None:
    payload = {
        "requirements": [
            {
                "text": "The DT shall collect the current",
                "type": "functional",
                "dimension": "data_collection_integration",
                "designGoal": "   ",
            }
        ]
    }
    candidate = extraction_from_payload(payload, turn_index=1).requirements[0]
    assert candidate.designGoal is None
    assert candidate.designGoal_provenance is None


def test_a_requirement_with_no_usable_dimension_is_dropped_not_guessed() -> None:
    payload = {
        "requirements": [
            {"text": "The DT shall do something", "type": "functional"},
            {"text": "x", "type": "functional", "dimension": "not_a_dimension"},
        ]
    }
    assert extraction_from_payload(payload, turn_index=1).requirements == []


def test_needs_clarification_survives_coercion() -> None:
    payload = {
        "requirements": [
            {
                "text": "The DT shall detect clogs",
                "type": "performance",
                "dimension": "intelligence_layer",
                "designGoal": None,
                "status": "needs_clarification",
            }
        ]
    }
    assert (
        extraction_from_payload(payload, turn_index=2).requirements[0].status
        == "needs_clarification"
    )


def test_not_applicable_claims_are_read_and_junk_ones_dropped() -> None:
    payload = {
        "requirements": [],
        "not_applicable": [
            {"dimension": "intelligence_layer", "quote": "we don't want that"},
            {"dimension": "nonsense"},
        ],
    }
    result = extraction_from_payload(payload, turn_index=5)
    assert len(result.not_applicable) == 1
    assert result.not_applicable[0].stakeholder_utterance_ref == 5


def test_a_missing_requirements_key_is_not_an_error() -> None:
    assert extraction_from_payload({}, turn_index=0).requirements == []


def test_maturity_payload_falls_back_to_a_valid_level() -> None:
    assert maturity_from_payload({"level": "Wishful"}).level == "Replication"
    assert maturity_from_payload({"level": "reality"}).level == "Reality"
    proposal = maturity_from_payload(
        {"level": "Replication", "user_roles": ["operator"], "description": "d"}
    )
    assert proposal.user_roles == ["operator"]


# --------------------------------------------------------------------------
# The retry ladder
# --------------------------------------------------------------------------


def test_a_first_pass_parse_needs_no_retry(state: SessionState) -> None:
    transport = transport_returning(GOOD_JSON)
    model = GroqModel(transport=transport, logger=CallLogger())
    result = model.extract(state, state.turns[1], Focus(kind="dimension", description="d"))
    assert len(result.requirements) == 1
    assert len(transport.calls) == 1  # type: ignore[attr-defined]


def test_malformed_output_is_retried_with_the_error_attached(
    state: SessionState,
) -> None:
    transport = transport_returning("{oops", GOOD_JSON)
    model = GroqModel(transport=transport, logger=CallLogger())
    result = model.extract(state, state.turns[1], Focus(kind="dimension", description="d"))

    assert len(result.requirements) == 1
    assert len(transport.calls) == 2  # type: ignore[attr-defined]
    assert "could not be parsed" in transport.calls[1]  # type: ignore[attr-defined]


def test_retries_are_capped_and_the_answer_is_parked_not_guessed(
    state: SessionState,
) -> None:
    """Out of retries means "left unprocessed", never "filled in"."""
    transport = transport_returning("still not json")
    logger = CallLogger()
    model = GroqModel(transport=transport, logger=logger)

    result = model.extract(state, state.turns[1], Focus(kind="dimension", description="d"))

    assert result.requirements == []
    assert result.not_applicable == []
    assert "could not be structured" in result.note
    assert len(transport.calls) == MAX_PARSE_RETRIES + 1  # type: ignore[attr-defined]
    assert any(record.get("error") == "parse_failed" for record in logger.records)


def test_the_retry_prompt_forbids_changing_values_to_make_them_parse(
    state: SessionState,
) -> None:
    transport = transport_returning("{oops", GOOD_JSON)
    model = GroqModel(transport=transport, logger=CallLogger())
    model.extract(state, state.turns[1], Focus(kind="dimension", description="d"))
    retry_prompt = " ".join(transport.calls[1].split())  # type: ignore[attr-defined]
    assert "do not add a number that the stakeholder did not say" in retry_prompt


# --------------------------------------------------------------------------
# Backoff policy
# --------------------------------------------------------------------------


class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(str(status_code))
        self.status_code = status_code


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_rate_limits_and_server_errors_are_retried(status: int) -> None:
    assert is_transient(_Status(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    """Retrying a bad key or a bad request just burns the stakeholder's time."""
    assert not is_transient(_Status(status))


def test_connection_failures_are_retried_by_name() -> None:
    assert is_transient(type("APIConnectionError", (Exception,), {})())


def test_a_plain_bug_is_not_retried() -> None:
    assert not is_transient(ValueError("boom"))


# --------------------------------------------------------------------------
# Logging and key handling
# --------------------------------------------------------------------------


def test_calls_are_logged_with_a_prompt_hash_not_the_prompt(
    state: SessionState, tmp_path
) -> None:
    logger = CallLogger(tmp_path / "llm_calls.jsonl")
    model = GroqModel(transport=transport_returning(GOOD_JSON), logger=logger)
    model.extract(state, state.turns[1], Focus(kind="dimension", description="d"))

    record = logger.records[0]
    assert record["call"] == "extract"
    assert record["temperature"] == 0.0
    assert record["json_mode"] is True
    assert len(str(record["prompt_sha256"])) == 16
    assert "prompt" not in record
    assert isinstance(record["latency_ms"], float)
    assert (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").strip()


def test_phrasing_calls_run_warmer_than_extraction(state: SessionState) -> None:
    logger = CallLogger()
    model = GroqModel(transport=transport_returning("What has to be captured?"), logger=logger)
    model.ask(state, Focus(kind="dimension", description="data collection"))
    assert logger.records[0]["temperature"] == 0.3


def test_a_live_model_refuses_to_start_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dtv_rea.groq_model.load_env", lambda: False)
    with pytest.raises(MissingApiKey) as error:
        GroqModel()
    assert "--stub" in str(error.value)


def test_the_key_never_appears_in_the_error_message(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value_do_not_leak")
    monkeypatch.setattr("dtv_rea.groq_model.load_env", lambda: False)
    with pytest.raises(MissingApiKey) as error:
        GroqModel()
    assert "gsk_secret_value_do_not_leak" not in str(error.value)


def test_an_unparsable_maturity_proposal_still_reaches_the_human(
    state: SessionState,
) -> None:
    """HITL-1 is the safety net: the human sees whatever the model actually said."""
    model = GroqModel(
        transport=transport_returning("I think Replication is right here."),
        logger=CallLogger(),
    )
    proposal = model.propose_maturity(state)
    assert proposal.level == "Replication"
    assert "I think Replication is right here." in proposal.description
