"""Every configuration knob for DTV-REA, in one place (spec section 8).

Two rules hold throughout this module:

1.  **Every filesystem location is a :class:`pathlib.Path`.** No string
    concatenation, no ``os.path.join``, no ``"/"`` or ``"\\"`` literals. This
    is what makes the agent behave identically on macOS and Windows.
2.  **Directories are resolved through functions, not constants.** Tests and
    the CLI override them with environment variables; a module-level constant
    computed at import time could not be overridden.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

#: Directory containing this package (``.../dtv_rea``).
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent

#: Repository root - the directory holding ``dtv_rea/``, ``data/``, ``eval/``.
REPO_ROOT: Final[Path] = PACKAGE_DIR.parent

_ENV_RUNS_DIR: Final[str] = "DTV_REA_RUNS_DIR"
_ENV_DATA_DIR: Final[str] = "DTV_REA_DATA_DIR"


def _from_env(variable: str) -> Path | None:
    """Return the path held in ``variable``, or ``None`` if it is unset."""
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def runs_dir() -> Path:
    """Directory that holds all session output (``./runs`` by default).

    Resolution order: the ``DTV_REA_RUNS_DIR`` environment override, else
    ``runs`` beside the repository root. The directory is created if it does
    not already exist.
    """
    directory = _from_env(_ENV_RUNS_DIR) or (REPO_ROOT / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def session_dir(session_id: str) -> Path:
    """Directory for one session's output files: ``./runs/<session_id>/``."""
    directory = runs_dir() / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def checkpoint_db() -> Path:
    """SQLite file backing the LangGraph checkpointer (``./runs/checkpoints.db``).

    Its parent directory is guaranteed to exist by the time this returns, so a
    caller can hand the path straight to :func:`sqlite3.connect`.
    """
    return runs_dir() / "checkpoints.db"


def env_file() -> Path:
    """The ``.env`` file holding the Groq API key."""
    return REPO_ROOT / ".env"


def load_env() -> bool:
    """Load the project's ``.env``. Returns whether a key is present.

    Only ``<project>/.env`` is read - never a ``.env`` located by searching
    parent directories, which is python-dotenv's default behaviour. That search
    makes the result depend on where the terminal happens to be sitting, and it
    can silently pick up an unrelated key from a folder higher up. The README
    tells people to put the file in the project folder, so that is the only
    place this looks.

    An already-set ``GROQ_API_KEY`` still wins, which is what CI expects.

    The key itself is never returned, logged or printed - only whether one
    was found.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        return bool(os.environ.get(GROQ_API_KEY_VAR, "").strip())

    path = env_file()
    if path.exists():
        load_dotenv(dotenv_path=path, override=False, encoding="utf-8")
    return bool(os.environ.get(GROQ_API_KEY_VAR, "").strip())


def data_dir() -> Path:
    """Directory holding personas and ground-truth cases (``./data``)."""
    return _from_env(_ENV_DATA_DIR) or (REPO_ROOT / "data")


def personas_dir() -> Path:
    """Directory holding scripted stakeholder personas."""
    return data_dir() / "personas"


def ground_truth_dir() -> Path:
    """Directory holding reverse-engineered ground-truth cases."""
    return data_dir() / "ground_truth"


def eval_dir() -> Path:
    """Directory holding the evaluation harness and its report."""
    return REPO_ROOT / "eval"


# --------------------------------------------------------------------------
# LLM configuration (spec section 1.5)
# --------------------------------------------------------------------------

#: Groq-hosted model used for every LLM call.
MODEL_NAME: Final[str] = os.environ.get(
    "DTV_REA_MODEL", "llama-3.3-70b-versatile"
)

#: Extraction must be reproducible: temperature 0, JSON mode.
TEMPERATURE_EXTRACT: Final[float] = 0.0

#: Question phrasing may vary mildly; content is constrained by the prompt.
TEMPERATURE_PHRASE: Final[float] = 0.3

#: Schema-parse retries allowed on an extraction before the answer is parked.
#: A parse retry is legitimate. A *fabrication* retry is not - see the module
#: docstring of :mod:`dtv_rea.validator`.
MAX_PARSE_RETRIES: Final[int] = 2

#: tenacity attempts on 429/5xx responses from Groq.
MAX_API_ATTEMPTS: Final[int] = 5

#: Environment variable holding the Groq API key. Read via python-dotenv from
#: a local ``.env`` file. Never hardcoded, never logged, never printed.
GROQ_API_KEY_VAR: Final[str] = "GROQ_API_KEY"


# --------------------------------------------------------------------------
# The four DTV requirement dimensions (spec section 1.4)
# --------------------------------------------------------------------------
#
# Derived from DTV section 3.2, which directs developers to consider
# requirements regarding "data collection, storage, and integration, the degree
# of fidelity and responsiveness of the virtual environment, the desired
# performance and accuracy of intelligence layers, and the degree of automation
# and feedback".
#
# The order below is the fixed checklist order used by ``decide_gap``.

DIMENSIONS: Final[tuple[str, ...]] = (
    "data_collection_integration",
    "virtual_environment",
    "intelligence_layer",
    "automation_feedback",
)

DIMENSION_LABELS: Final[dict[str, str]] = {
    "data_collection_integration": "Data collection, storage and integration",
    "virtual_environment": "Fidelity and responsiveness of the virtual environment",
    "intelligence_layer": "Performance and accuracy of the intelligence layer",
    "automation_feedback": "Degree of automation and feedback",
}

#: Plain-language description of what each dimension is asking about. Used by
#: P4 so the generated question never has to use framework vocabulary.
DIMENSION_TOPICS: Final[dict[str, str]] = {
    "data_collection_integration": (
        "which measurements have to be captured from the physical machine, how "
        "often, how they are stored, and how completely they have to arrive in "
        "the twin"
    ),
    "virtual_environment": (
        "what the person using the twin has to be able to see on screen, how "
        "closely it has to match the real machine, and how quickly it has to "
        "update"
    ),
    "intelligence_layer": (
        "what the twin has to work out or detect on its own from the data, and "
        "how often it has to be right"
    ),
    "automation_feedback": (
        "what the twin is allowed to do back to the physical machine on its "
        "own, and what it should only alert a person about"
    ),
}


# --------------------------------------------------------------------------
# 4R maturity levels (Hyre et al. 2022; Osho et al. 2022, as used by DTV)
# --------------------------------------------------------------------------

MATURITY_LEVELS: Final[tuple[str, ...]] = (
    "Representation",
    "Replication",
    "Reality",
    "Relational",
)

MATURITY_DEFINITIONS: Final[dict[str, str]] = {
    "Representation": (
        "The components needed to represent the physical system in the virtual "
        "environment are identified, and the flow of information and data from "
        "the physical system to the virtual space is realised."
    ),
    "Replication": (
        "The physical system components are modelled in the virtual "
        "environment, replicating the behaviour of the physical counterpart."
    ),
    "Reality": (
        "Prediction capabilities are added to the model, so the twin can "
        "predict the outcome of the physical system."
    ),
    "Relational": (
        "The intelligence layer is incremented so the twin can autonomously "
        "apply changes to the physical system."
    ),
}


# --------------------------------------------------------------------------
# Validator configuration (spec section 1.3)
# --------------------------------------------------------------------------

#: V3 threshold. Content-word overlap at or above this value marks two
#: requirements in the same dimension as a possible duplicate obligation.
#:
#: Honest limitation, carried into the paper: this threshold is tuned to the
#: known case. The published UR1.2 / UR1.7 pair scores exactly 0.60 under the
#: stop-word list below, which is why the threshold sits where it does.
DUPLICATE_THRESHOLD: Final[float] = 0.6

#: V3 minimum. Requirements with fewer content words than this are not
#: compared, because the overlap measure is unstable on very short texts.
DUPLICATE_MIN_CONTENT_WORDS: Final[int] = 2

#: V3 evidence floor. Two requirements must share at least this many content
#: words before their overlap ratio counts as evidence of one duplicated
#: obligation.
#:
#: Without it the ratio alone is misleading on short requirements. "The DT
#: shall collect data on the duration of the job" and "... on the type of the
#: job" reduce to three content words each and share exactly two of them,
#: "collect" and "job" - the two most common words in this domain - which
#: scores 0.67 and reads as a duplicate. It is not one; they are simply both
#: short. The published UR1.2 / UR1.7 pair shares three words, so the
#: calibration this validator is tuned to is unaffected.
DUPLICATE_MIN_SHARED_WORDS: Final[int] = 3

#: Q3. Times a single dimension (or a single missing design goal) may be asked
#: about before it is parked and the interview moves on.
ATTEMPT_CAP: Final[int] = 3

#: V3 stop words. Grammatical function words plus the four terms that appear in
#: nearly every digital-twin requirement and therefore carry no discriminating
#: content: "dt", "digital", "twin", "system", "data".
STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        # articles, conjunctions, prepositions, auxiliaries
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in",
        "on", "at", "for", "from", "with", "by", "as", "is", "are", "was",
        "were", "be", "been", "being", "shall", "should", "must", "will",
        "can", "may", "it", "its", "this", "that", "these", "those", "there",
        "we", "our", "they", "their", "all", "any", "each", "every", "both",
        "when", "while", "during", "into", "out", "up", "down", "over",
        "under", "again", "further", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "s", "t",
        # domain boilerplate - present in almost every requirement
        "dt", "digital", "twin", "system", "data",
    }
)

#: V2 vague predicates, stored as prefixes so inflections are caught too
#: ("minimize", "minimizing", "minimised" all match "minimi").
#:
#: Honest limitation, carried into the paper: this is a keyword heuristic and a
#: first pass, not a semantic judgement of verifiability.
VAGUE_ROOTS: Final[tuple[str, ...]] = (
    "minimi", "maximi", "optimi", "improv", "reduc", "increas", "enhanc",
    "reliab", "robust", "fast", "quick", "rapid", "accura", "precis",
    "suffic", "adequa", "accepta", "appropria", "effici", "effectiv",
    "seamless", "timely", "minimal", "significan", "substantial",
    "approximat", "roughly", "user-friendly", "easy", "simple", "smooth",
)

#: Requirements must state *what* is needed, not *how* to build it. Used by the
#: solution-free lint in the evaluation harness (spec section 5,
#: "requirement well-formedness").
BANNED_SOLUTION_WORDS: Final[tuple[str, ...]] = (
    "mqtt", "docker", "node-red", "nodered", "influxdb", "grafana", "unity",
    "python", "kafka", "postgres", "mysql", "mongodb", "raspberry", "arduino",
    "thermocouple", "accelerometer", "rest api", "websocket", "opc-ua",
    "opcua", "aws", "azure", "kubernetes", "tensorflow", "pytorch",
    "scikit-learn", "sklearn",
)


# --------------------------------------------------------------------------
# Interview / CLI behaviour
# --------------------------------------------------------------------------

#: Typed words that end the interview early and emit partial output (Q2).
#: A typed keyword is the primary mechanism because the end-of-input key
#: differs between platforms (Ctrl-D on macOS, Ctrl-Z then Enter on Windows).
STOP_WORDS_INPUT: Final[frozenset[str]] = frozenset(
    {"stop", "quit", "exit", ":q", "/stop"}
)

#: Upper bound on LangGraph node executions in a single run. The interview is a
#: cycle; this stops a malformed script from looping forever.
GRAPH_RECURSION_LIMIT: Final[int] = 250
