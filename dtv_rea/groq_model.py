"""The Groq-hosted model - the only network dependency (spec section 1.5).

Which model is a setting, not a fact about this file: see ``MODEL_NAME`` in
:mod:`dtv_rea.settings`, overridable with ``DTV_REA_MODEL``. The requirement
this module places on it is JSON mode, because every extraction uses it.

Configuration, exactly as specified:

* ``extract`` runs at ``temperature=0`` in JSON mode
  (``response_format={"type": "json_object"}``), and its output is parsed into
  the Pydantic schema.
* On a parse error the extraction is retried at most twice with the error
  appended, and then the answer is **parked** as a note rather than guessed at.
* Everything else - question phrasing, the maturity proposal, flag wording -
  runs at ``temperature=0.3``. Mild variation in phrasing is fine because the
  *content* is constrained by the prompt and, where it matters, by the
  validator.
* 429 and 5xx responses are retried with exponential backoff, five attempts.
* Every call is logged to ``llm_calls.jsonl``: prompt hash, latency, tokens.

A parse retry is legitimate - the model produced malformed JSON and is being
asked to format it properly. A **fabrication** retry is not, and does not exist
anywhere in this file. When V1 rejects a number the recovery is always to ask
the human, because a regenerated number is still an invented number.

The API key comes from the ``GROQ_API_KEY`` environment variable, loaded from a
local ``.env``. It is never hardcoded, never logged and never printed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from dtv_rea.llm import CallLogger, Focus, MaturityProposal
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
from dtv_rea.settings import (
    DIMENSION_LABELS,
    GROQ_API_KEY_VAR,
    MATURITY_LEVELS,
    MAX_API_ATTEMPTS,
    MAX_PARSE_RETRIES,
    MODEL_NAME,
    TEMPERATURE_EXTRACT,
    TEMPERATURE_PHRASE,
    load_env,
)
from dtv_rea.state import (
    ExtractionResult,
    Flag,
    NotApplicableClaim,
    RequirementCandidate,
    SessionState,
    Turn,
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class MissingApiKey(RuntimeError):
    """Raised when a live run is attempted with no ``GROQ_API_KEY`` set."""


class JsonParseError(ValueError):
    """The model's reply could not be read as a JSON object."""


# --------------------------------------------------------------------------
# Pure parsing helpers - unit-tested offline against canned malformed output
# --------------------------------------------------------------------------


def strip_fences(text: str) -> str:
    """Remove a markdown code fence if the model wrapped its JSON in one."""
    without_open = _FENCE_RE.sub("", text.strip(), count=1)
    return _FENCE_RE.sub("", without_open, count=1).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Read a JSON object out of a model reply.

    Tolerates the two things models actually do wrong in JSON mode: wrapping
    the object in a code fence, and bracketing it with a sentence of prose.
    Anything beyond that is a genuine parse failure and is reported as one -
    guessing at malformed output is how invented values get in.
    """
    if not text or not text.strip():
        raise JsonParseError("The model returned an empty reply.")

    candidate = strip_fences(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JsonParseError(str(first_error)) from first_error
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as second_error:
            raise JsonParseError(str(second_error)) from second_error

    if not isinstance(parsed, dict):
        raise JsonParseError(
            f"Expected a JSON object, got {type(parsed).__name__}."
        )
    return parsed


def _clean_goal(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Normalise the goal/provenance pair coming back from the model.

    ``designGoal_provenance`` is forced to ``"stated"`` whenever a goal is
    present, and to ``None`` when it is not. The model is not permitted to
    invent a third provenance, and the schema would reject one anyway - this
    just turns a model slip into a clean value instead of a crash. Whether the
    goal is *legitimate* is V1's question, not this function's.
    """
    goal = payload.get("designGoal")
    if isinstance(goal, str) and not goal.strip():
        goal = None
    if goal is None:
        return None, None
    return str(goal), "stated"


def extraction_from_payload(
    payload: dict[str, Any], turn_index: int
) -> ExtractionResult:
    """Coerce P5's JSON into the Pydantic schema, dropping unusable entries.

    Deliberately forgiving about *shape* and completely unforgiving about
    *content*: a requirement missing its dimension is dropped rather than
    guessed at, and no field is ever filled in with a plausible default.
    """
    candidates: list[RequirementCandidate] = []
    for raw in payload.get("requirements") or []:
        if not isinstance(raw, dict):
            continue
        dimension = raw.get("dimension")
        target = raw.get("goal_target_id")
        if dimension not in DIMENSION_LABELS:
            if not target:
                continue
            dimension = "data_collection_integration"

        goal, provenance = _clean_goal(raw)
        reference = raw.get("stakeholder_utterance_ref")
        if goal is not None and not isinstance(reference, int):
            # A goal with no usable citation is still a goal. Pin it to the
            # turn being extracted so V1 checks it against the real answer
            # rather than rejecting it on a technicality.
            reference = turn_index

        requirement_type = raw.get("type")
        if goal is not None:
            # DTV section 3.2: "design goals should also be associated with
            # each performance requirement". A requirement carrying a number
            # therefore *is* a performance requirement, whatever the model
            # labelled it. Observed in the live FDM run: Llama 3.3 typed four
            # requirements "functional" while attaching design goals to them,
            # which would have given each the wrong verification method
            # (Inspection instead of Test) and hidden them from the
            # missing-goal check. Classification is a decision, so it is made
            # here in code rather than trusted to the model.
            requirement_type = "performance"
        elif requirement_type not in {"functional", "performance"}:
            requirement_type = "functional"

        status = raw.get("status")
        if status not in {"complete", "needs_clarification"}:
            status = "complete"

        priority = raw.get("priority")
        if priority not in {"must", "should", "could"}:
            priority = "must"

        try:
            candidates.append(
                RequirementCandidate(
                    text=str(raw.get("text") or ""),
                    type=requirement_type,
                    dimension=dimension,
                    designGoal=goal,
                    designGoal_provenance=provenance,
                    stakeholder_utterance_ref=(
                        reference if isinstance(reference, int) else None
                    ),
                    priority=priority,
                    rationale=str(raw.get("rationale") or ""),
                    status=status,
                    goal_target_id=str(target) if target else None,
                )
            )
        except Exception:  # pragma: no cover - schema guards the rest
            continue

    claims: list[NotApplicableClaim] = []
    for raw in payload.get("not_applicable") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("dimension") not in DIMENSION_LABELS:
            continue
        reference = raw.get("stakeholder_utterance_ref")
        claims.append(
            NotApplicableClaim(
                dimension=raw["dimension"],
                stakeholder_utterance_ref=(
                    reference if isinstance(reference, int) else turn_index
                ),
                quote=str(raw.get("quote") or ""),
            )
        )

    return ExtractionResult(
        requirements=candidates,
        not_applicable=claims,
        note=str(payload.get("note") or ""),
    )


def maturity_from_payload(payload: dict[str, Any]) -> MaturityProposal:
    """Coerce P3's JSON into a proposal, defaulting only the level's spelling."""
    level = str(payload.get("level") or "").strip().title()
    if level not in MATURITY_LEVELS:
        level = "Replication"
    roles = payload.get("user_roles")
    return MaturityProposal(
        level=level,
        reasoning=str(payload.get("reasoning") or ""),
        description=str(payload.get("description") or ""),
        rationale=str(payload.get("rationale") or ""),
        user_roles=[str(role) for role in roles] if isinstance(roles, list) else [],
        context_of_use=str(payload.get("context_of_use") or ""),
    )


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def is_transient(error: BaseException) -> bool:
    """True for 429 and 5xx responses and for connection failures.

    Deliberately duck-typed rather than importing Groq's exception classes, so
    that the policy can be unit-tested without the SDK and does not break when
    the SDK reorganises its exception hierarchy.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    name = type(error).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ConnectionError",
        "Timeout",
    }


_with_backoff = retry(
    retry=retry_if_exception(is_transient),
    stop=stop_after_attempt(MAX_API_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class GroqModel:
    """:class:`~dtv_rea.llm.ModelPort` backed by a Groq-hosted model."""

    name = MODEL_NAME

    def __init__(
        self,
        logger: CallLogger | None = None,
        model_name: str | None = None,
        transport: Callable[[str, str, float, bool], str] | None = None,
    ) -> None:
        """``transport`` exists so tests can drive the parse and retry logic
        without a network. Production passes nothing and gets ChatGroq."""
        self.model_name = model_name or MODEL_NAME
        self.logger = logger or CallLogger()
        self._transport = transport
        self._clients: dict[tuple[float, bool], Any] = {}
        if transport is None:
            if not load_env():
                raise MissingApiKey(
                    f"{GROQ_API_KEY_VAR} is not set. Copy .env.example to .env "
                    f"and put your Groq API key in it, or run with --stub to "
                    f"use the offline scripted model."
                )

    # -- transport ------------------------------------------------------

    def _client(self, temperature: float, json_mode: bool) -> Any:
        key = (temperature, json_mode)
        if key not in self._clients:
            from langchain_groq import ChatGroq

            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["model_kwargs"] = {
                    "response_format": {"type": "json_object"}
                }
            self._clients[key] = ChatGroq(**kwargs)
        return self._clients[key]

    @_with_backoff
    def _call_api(
        self, system: str, user: str, temperature: float, json_mode: bool
    ) -> str:
        if self._transport is not None:
            return self._transport(system, user, temperature, json_mode)
        response = self._client(temperature, json_mode).invoke(
            [("system", system), ("human", user)]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):  # pragma: no cover - multimodal shape
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        self._last_usage = getattr(response, "usage_metadata", None) or {}
        return str(content)

    def _complete(
        self,
        call: str,
        user: str,
        temperature: float,
        json_mode: bool = False,
        **extra: Any,
    ) -> str:
        self._last_usage: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            content = self._call_api(P1_SYSTEM, user, temperature, json_mode)
        except Exception as error:
            self.logger.log(
                call=call,
                model=self.model_name,
                ok=False,
                error=type(error).__name__,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                **extra,
            )
            raise
        usage = self._last_usage or {}
        self.logger.log(
            call=call,
            model=self.model_name,
            ok=True,
            temperature=temperature,
            json_mode=json_mode,
            # A hash, not the prompt. This log is an operational record, not a
            # second copy of the stakeholder's transcript.
            prompt_sha256=hashlib.sha256(user.encode("utf-8")).hexdigest()[:16],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            **extra,
        )
        return content

    # -- P2 -------------------------------------------------------------

    def opener(self, state: SessionState) -> str:
        return self._complete(
            "opener", P2_CAPTURE_PURPOSE.substitute(), TEMPERATURE_PHRASE
        ).strip()

    # -- P3 -------------------------------------------------------------

    def propose_maturity(self, state: SessionState) -> MaturityProposal:
        purpose = state.purpose.statement if state.purpose else ""
        reply = self._complete(
            "propose_maturity",
            P3_PROPOSE_MATURITY.substitute(purpose=purpose),
            TEMPERATURE_PHRASE,
            json_mode=True,
        )
        try:
            return maturity_from_payload(parse_json_object(reply))
        except JsonParseError:
            # A malformed proposal is recoverable: the human confirms or
            # overrides it at HITL-1 anyway, and the raw reply is what they see.
            return MaturityProposal(
                level="Replication",
                reasoning="The model's proposal could not be parsed.",
                description=reply.strip(),
            )

    # -- P4 -------------------------------------------------------------

    def ask(self, state: SessionState, focus: Focus) -> str:
        return self._complete(
            "ask",
            P4_GENERATE_QUESTION.substitute(
                purpose=state.purpose.statement if state.purpose else "",
                maturity=state.maturity.value if state.maturity else "not yet agreed",
                committed=committed_summary(state.requirements),
                focus=_describe_focus(state, focus),
            ),
            TEMPERATURE_PHRASE,
            focus=focus.kind,
        ).strip()

    # -- P5 -------------------------------------------------------------

    def extract(
        self, state: SessionState, answer: Turn, focus: Focus
    ) -> ExtractionResult:
        prompt = P5_EXTRACT.substitute(
            purpose=state.purpose.statement if state.purpose else "",
            focus=_describe_focus(state, focus),
            committed=committed_summary(state.requirements),
            turn_index=answer.index,
            answer=answer.text,
        )

        last_error = ""
        for attempt in range(MAX_PARSE_RETRIES + 1):
            user = prompt if attempt == 0 else (
                prompt + P5_RETRY_SUFFIX.substitute(error=last_error)
            )
            reply = self._complete(
                "extract",
                user,
                TEMPERATURE_EXTRACT,
                json_mode=True,
                turn=answer.index,
                parse_retries=attempt,
            )
            try:
                payload = parse_json_object(reply)
            except JsonParseError as error:
                last_error = str(error)
                continue
            return extraction_from_payload(payload, answer.index)

        # Out of retries. Park the answer rather than guess at what it meant -
        # a requirement invented from unparsable output is exactly the failure
        # this whole system exists to prevent.
        self.logger.log(
            call="extract",
            model=self.model_name,
            ok=False,
            error="parse_failed",
            turn=answer.index,
            parse_retries=MAX_PARSE_RETRIES,
        )
        return ExtractionResult(
            requirements=[],
            not_applicable=[],
            note=(
                f"The answer at turn {answer.index} could not be structured "
                f"after {MAX_PARSE_RETRIES + 1} attempts ({last_error}). It has "
                f"been left unprocessed rather than guessed at."
            ),
        )

    # -- P6 -------------------------------------------------------------

    def phrase_flag(self, state: SessionState, flag: Flag) -> str:
        return self._complete(
            "phrase_flag",
            P6_RESOLVE_FLAG.substitute(code=flag.code, message=flag.message),
            TEMPERATURE_PHRASE,
            code=flag.code,
        ).strip()


def _describe_focus(state: SessionState, focus: Focus) -> str:
    """Turn the code's decision into the sentence P4/P5 read."""
    if focus.kind == "fabrication_recovery":
        return (
            f"A design goal for \"{focus.description}\" was rejected because "
            f"the number in it was not something the stakeholder said. Ask them "
            f"directly what the value should be. Do not propose one."
        )
    if focus.kind == "goal":
        requirement = state.requirement(focus.requirement_id or "")
        text = requirement.text if requirement else focus.description
        return (
            f"The requirement \"{text}\" has no design goal. Ask the "
            f"stakeholder what number it has to meet."
        )
    if focus.kind == "retry_dimension":
        return (
            f"{focus.description}. The stakeholder has already been asked about "
            f"this {focus.attempt - 1} time(s) and has not given anything "
            f"usable. Ask something narrower, or ask whether it applies at all."
        )
    return focus.description


__all__ = [
    "GroqModel",
    "JsonParseError",
    "MissingApiKey",
    "extraction_from_payload",
    "is_transient",
    "maturity_from_payload",
    "parse_json_object",
    "strip_fences",
]
