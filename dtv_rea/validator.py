"""V1-V5, the deterministic validator (spec section 1.3).

This module is the whole research claim. It is pure Python: no model call, no
network, no randomness. Given the same candidate and the same session it always
returns the same verdict, which is what makes "fabricated-goal rate = 0" a
measurable property rather than a hope about prompt compliance.

Two severities:

``hard``
    Blocks the commit. The candidate is rejected outright.

``flag``
    Commits the candidate but blocks *termination* until a human resolves it.

The single most important rule in the file, V1: when a design goal cannot be
traced to something the stakeholder actually said, the answer is never to ask
the model again. **A regenerated number is still an invented number.** The only
legitimate resolution is to ask the human.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from dtv_rea.settings import (
    DUPLICATE_MIN_CONTENT_WORDS,
    DUPLICATE_MIN_SHARED_WORDS,
    DUPLICATE_THRESHOLD,
    STOP_WORDS,
    VAGUE_ROOTS,
)
from dtv_rea.state import FlagCode, Requirement, RequirementCandidate, SessionState

Severity = Literal["hard", "flag"]


class Finding(BaseModel):
    """One validator result. Turned into a :class:`~dtv_rea.state.Flag` later.

    Pydantic rather than a dataclass because findings travel between graph
    nodes and therefore through the checkpointer.
    """

    code: FlagCode
    severity: Severity
    message: str
    related_requirement_id: str | None = None
    detail: dict[str, str] = Field(default_factory=dict)


class Verdict(BaseModel):
    """Everything V1-V5 concluded about one candidate."""

    candidate: RequirementCandidate
    findings: list[Finding] = Field(default_factory=list)

    @property
    def hard_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "hard"]

    @property
    def soft_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "flag"]

    @property
    def rejected(self) -> bool:
        """True when a HARD check failed, so the candidate must not commit."""
        return bool(self.hard_findings)


# --------------------------------------------------------------------------
# Text normalisation helpers
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+")


def numeric_tokens(text: str | None) -> set[str]:
    """Every number in ``text``, normalised to a canonical decimal string.

    Normalisation deliberately discards everything around the digits, so that
    ``90%``, ``>= 90 %`` and ``90`` all reduce to ``"90"``, and ``+/- 2.5``,
    ``±2.5`` and ``2.50`` all reduce to ``"2.5"``. Comparison operators and
    units are not part of the claim being traced - the *number* is.
    """
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        try:
            value = Decimal(raw).normalize()
        except InvalidOperation:  # pragma: no cover - regex cannot produce this
            continue
        tokens.add(format(value, "f"))
    return tokens


def light_stem(word: str) -> str:
    """A deliberately small stemmer: ``collected`` -> ``collect``.

    The goal is *consistency*, not dictionary lemmas. ``integrate`` and
    ``integrated`` both reduce to ``integrat``, which is ugly and correct - if
    the two forms stemmed differently, V3 would miss the overlap between a
    requirement that says "shall integrate" and one that says "integrated".
    That is why the trailing ``e`` comes off at the end.

    Nothing cleverer is justified here. V3 is a first-pass heuristic and a
    heavier stemmer would make its behaviour harder to explain in a paper.
    """
    stem = word
    for suffix in ("ations", "ation", "ements", "ement", "ings", "ing", "ies",
                   "ed", "es", "s"):
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
            stem = stem[: -len(suffix)]
            if suffix == "ies":
                stem += "y"
            break
    if stem.endswith("e") and len(stem) >= 4:
        stem = stem[:-1]
    return stem


def content_words(text: str) -> set[str]:
    """Lower-case, stop-word-free, lightly stemmed content words of ``text``."""
    words = _WORD_RE.findall((text or "").lower())
    return {
        light_stem(word)
        for word in words
        if word not in STOP_WORDS and light_stem(word) not in STOP_WORDS
    }


def overlap_ratio(left: set[str], right: set[str]) -> float:
    """Overlap coefficient - the shared words over the *smaller* word set.

    Chosen over Jaccard because requirements differ wildly in length: the
    published UR1.2 bundles six data elements while UR1.7 names one, and a
    length-symmetric measure buries that pair far below any usable threshold.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def vague_predicate(text: str) -> str | None:
    """Return the first vague term in ``text``, or ``None``.

    Prefix matching, so ``minimize`` / ``minimizing`` / ``minimised`` all match
    the root ``minimi``.
    """
    for word in _WORD_RE.findall((text or "").lower()):
        for root in VAGUE_ROOTS:
            if word.startswith(root):
                return word
    return None


# --------------------------------------------------------------------------
# The five checks
# --------------------------------------------------------------------------


def check_v1_fabrication(
    candidate: RequirementCandidate, state: SessionState
) -> list[Finding]:
    """V1 (HARD) - a design goal must trace to a real stakeholder utterance.

    Four conditions, all required: there is a reference; it points at a real
    turn; that turn is the *stakeholder's*; and that turn's text contains every
    numeric token of the goal.
    """
    goal = candidate.designGoal
    if goal is None:
        return []

    subject = candidate.goal_target_id or candidate.text or "this requirement"
    reference = candidate.stakeholder_utterance_ref

    if reference is None:
        return [
            Finding(
                code="fabricated_goal",
                severity="hard",
                message=(
                    f"The design goal \"{goal}\" for {subject} cites no "
                    f"stakeholder turn. It has to be asked for, not inferred."
                ),
                detail={"goal": goal, "reason": "no_reference"},
            )
        ]

    turn = state.turn(reference)
    if turn is None:
        return [
            Finding(
                code="fabricated_goal",
                severity="hard",
                message=(
                    f"The design goal \"{goal}\" for {subject} cites turn "
                    f"{reference}, which does not exist."
                ),
                detail={"goal": goal, "reason": "reference_out_of_range"},
            )
        ]

    if turn.role != "stakeholder":
        return [
            Finding(
                code="fabricated_goal",
                severity="hard",
                message=(
                    f"The design goal \"{goal}\" for {subject} cites turn "
                    f"{reference}, which is the agent speaking, not the "
                    f"stakeholder. A number the agent said is not a source."
                ),
                detail={"goal": goal, "reason": "reference_is_agent_turn"},
            )
        ]

    missing = numeric_tokens(goal) - numeric_tokens(turn.text)
    if missing:
        return [
            Finding(
                code="fabricated_goal",
                severity="hard",
                message=(
                    f"The design goal \"{goal}\" for {subject} contains "
                    f"{', '.join(sorted(missing))}, which does not appear in "
                    f"the cited turn {reference}. Ask the stakeholder for the "
                    f"value; do not regenerate it."
                ),
                detail={"goal": goal, "reason": "number_absent_from_cited_turn"},
            )
        ]

    return []


def check_v2_vague_predicate(candidate: RequirementCandidate) -> list[Finding]:
    """V2 (FLAG) - a vague predicate with no number to pin it down.

    This is the published UR1.11 case: "The system shall minimize data loss".
    There is no test that can pass or fail such a requirement until somebody
    says what "minimize" means numerically.
    """
    if candidate.goal_target_id is not None:
        return []
    if candidate.designGoal is not None:
        return []

    term = vague_predicate(candidate.text)
    if term is None:
        return []

    return [
        Finding(
            code="unverifiable_predicate",
            severity="flag",
            message=(
                f"\"{candidate.text}\" says \"{term}\" but gives no number, so "
                f"there is no way to test whether it has been met."
            ),
            detail={"term": term},
        )
    ]


def check_v3_duplicate_obligation(
    candidate: RequirementCandidate, peers: Sequence[Requirement]
) -> list[Finding]:
    """V3 (FLAG) - two requirements in one dimension carrying one obligation.

    This is the published UR1.2 / UR1.7 case. The agent never merges: it points
    the pair out and a human decides (spec section 4, HITL-3).

    Two properties keep this from drowning the interview in questions.

    **An evidence floor as well as a ratio.** The overlap coefficient is
    unstable on short requirements: three content words that share two generic
    ones score 0.67 and look like a duplicate. Requiring
    ``DUPLICATE_MIN_SHARED_WORDS`` shared words as well means a ratio only
    counts when there is something substantive behind it.

    **At most one finding per candidate.** Comparing against every peer
    produces a finding per *pair*, so N similar requirements cost N-squared
    flags - and every flag is a model call plus a near-identical question to
    the stakeholder. The human only needs asking once per requirement, about
    its closest match; answering that settles how they think about the rest.
    """
    if candidate.goal_target_id is not None:
        return []

    words = content_words(candidate.text)
    if len(words) < DUPLICATE_MIN_CONTENT_WORDS:
        return []

    best: tuple[float, set[str], Requirement] | None = None
    for peer in peers:
        if peer.dimension != candidate.dimension:
            continue
        peer_words = content_words(peer.text)
        if len(peer_words) < DUPLICATE_MIN_CONTENT_WORDS:
            continue
        shared = words & peer_words
        if len(shared) < DUPLICATE_MIN_SHARED_WORDS:
            continue
        score = overlap_ratio(words, peer_words)
        if score < DUPLICATE_THRESHOLD:
            continue
        if best is None or score > best[0]:
            best = (score, shared, peer)

    if best is None:
        return []

    score, shared, peer = best
    return [
        Finding(
            code="duplicate_obligation",
            severity="flag",
            message=(
                f"\"{candidate.text}\" overlaps {peer.id} (\"{peer.text}\") by "
                f"{score:.2f} of its content words, sharing "
                f"{', '.join(sorted(shared))}. These may be one obligation "
                f"stated twice."
            ),
            related_requirement_id=peer.id,
            detail={"overlap": f"{score:.2f}", "shared": str(len(shared))},
        )
    ]


def check_v4_orphan_goal(
    candidate: RequirementCandidate, state: SessionState
) -> list[Finding]:
    """V4 (HARD) - a goal attached to a requirement that does not exist."""
    target = candidate.goal_target_id
    if target is None:
        return []
    if state.requirement(target) is not None:
        return []
    return [
        Finding(
            code="orphan_goal",
            severity="hard",
            message=(
                f"A design goal was offered for {target}, but no requirement "
                f"with that id has been committed."
            ),
            detail={"target": target},
        )
    ]


def check_v5_maturity_consistency(
    candidate: RequirementCandidate, state: SessionState
) -> list[Finding]:
    """V5 (FLAG) - a requirement landing in a dimension already ruled out.

    Surfaces both statements so the human can say which one stands. Note the
    two ways a dimension can be ``not_applicable``: the stakeholder ruled it
    out, or the attempt cap parked it. The message distinguishes them.
    """
    if candidate.goal_target_id is not None:
        return []
    if state.coverage_status.get(candidate.dimension) != "not_applicable":
        return []

    if state.is_parked(candidate.dimension):
        standing = (
            "that topic was parked after "
            f"{state.attempts.get(candidate.dimension, 0)} attempts with no "
            "usable answer"
        )
    else:
        confirming = state.not_applicable_confirmations.get(candidate.dimension)
        turn = state.turn(confirming)
        quote = turn.text if turn is not None else "(turn not recorded)"
        standing = f'earlier the stakeholder said: "{quote}"'

    return [
        Finding(
            code="maturity_inconsistency",
            severity="flag",
            message=(
                f"\"{candidate.text}\" belongs to a topic marked as not "
                f"applicable - {standing}. Which statement stands?"
            ),
            detail={"dimension": candidate.dimension},
        )
    ]


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def validate_candidate(
    candidate: RequirementCandidate,
    state: SessionState,
    peers: Sequence[Requirement] | None = None,
) -> Verdict:
    """Run V1-V5 over one candidate and return its verdict.

    ``peers`` defaults to everything already committed. The graph passes a
    wider list so that two candidates extracted from the *same* answer are
    compared against each other as well.
    """
    if peers is None:
        peers = state.requirements

    findings: list[Finding] = []
    findings.extend(check_v1_fabrication(candidate, state))
    findings.extend(check_v4_orphan_goal(candidate, state))
    findings.extend(check_v2_vague_predicate(candidate))
    findings.extend(check_v3_duplicate_obligation(candidate, peers))
    findings.extend(check_v5_maturity_consistency(candidate, state))
    return Verdict(candidate=candidate, findings=findings)


def validate_batch(
    candidates: Sequence[RequirementCandidate], state: SessionState
) -> list[Verdict]:
    """Validate one answer's worth of candidates, in order.

    Candidates are compared against the committed set *and* against the
    candidates ahead of them in the same batch, so a stakeholder who states the
    same obligation twice in one breath is still flagged.
    """
    verdicts: list[Verdict] = []
    peers: list[Requirement] = list(state.requirements)
    provisional_number = state.next_requirement_number

    for candidate in candidates:
        verdict = validate_candidate(candidate, state, peers)
        verdicts.append(verdict)
        if not verdict.rejected and candidate.goal_target_id is None:
            # Stand the accepted candidate in as a peer for the ones that
            # follow it. The id is provisional; it is never persisted from here.
            peers.append(
                Requirement(
                    id=f"UR1.{provisional_number}",
                    text=candidate.text,
                    type=candidate.type,
                    dimension=candidate.dimension,
                    verifyMethod=candidate.default_verify_method(),  # type: ignore[arg-type]
                    designGoal=candidate.designGoal,
                    designGoal_provenance=(
                        "stated" if candidate.designGoal is not None else None
                    ),
                    stakeholder_utterance_ref=candidate.stakeholder_utterance_ref,
                    priority=candidate.priority,
                    rationale=candidate.rationale,
                    status=candidate.status,
                )
            )
            provisional_number += 1

    return verdicts


def audit_committed_goals(state: SessionState) -> list[Requirement]:
    """Independent re-check for the evaluation harness (spec section 5).

    Re-derives, from the transcript alone, which committed design goals cannot
    be traced to a stakeholder utterance. This deliberately does **not** read
    the validator's own verdicts - a metric that trusts the component it is
    measuring measures nothing.
    """
    fabricated: list[Requirement] = []
    for requirement in state.requirements:
        if requirement.designGoal is None:
            continue
        turn = state.turn(requirement.stakeholder_utterance_ref)
        if turn is None or turn.role != "stakeholder":
            fabricated.append(requirement)
            continue
        if numeric_tokens(requirement.designGoal) - numeric_tokens(turn.text):
            fabricated.append(requirement)
    return fabricated


__all__ = [
    "Finding",
    "Verdict",
    "audit_committed_goals",
    "check_v1_fabrication",
    "check_v2_vague_predicate",
    "check_v3_duplicate_obligation",
    "check_v4_orphan_goal",
    "check_v5_maturity_consistency",
    "content_words",
    "light_stem",
    "numeric_tokens",
    "overlap_ratio",
    "vague_predicate",
    "validate_batch",
    "validate_candidate",
]
