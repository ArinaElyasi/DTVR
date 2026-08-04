"""Cross-platform guarantees, enforced as tests rather than as a promise.

Everything in this file is a property that would otherwise quietly break the
first time a teammate on the other operating system pulled the repository. A
convention that is only written down in a README is a convention that drifts.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

import pytest

from dtv_rea import settings
from dtv_rea.settings import (
    REPO_ROOT,
    checkpoint_db,
    data_dir,
    ground_truth_dir,
    personas_dir,
    runs_dir,
    session_dir,
)

SOURCE_DIRS = ("dtv_rea", "eval", "tests")

#: This module is the one file that has to *name* the forbidden patterns in
#: order to search for them, so it excludes itself from the pattern scans -
#: the same self-exemption a lint rule takes. It is still scanned by the
#: encoding, JSON and secret checks below, which have no such problem.
GUARD = Path(__file__).resolve()


def python_files(include_guard: bool = False) -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        files.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    if not include_guard:
        files = [path for path in files if path.resolve() != GUARD]
    assert files, "no source files found"
    return files


def source_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def code_of(path: Path) -> str:
    """The file with comments and string literals blanked out.

    Scanning raw text finds the rule written down in a docstring as readily as
    it finds the rule being broken. This looks at code only.
    """
    pieces: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(source_of(path)).readline)
    for token in tokens:
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


def string_literals(path: Path) -> list[tokenize.TokenInfo]:
    """Every string literal in the file, for checks that need the literals."""
    return [
        token
        for token in tokenize.generate_tokens(io.StringIO(source_of(path)).readline)
        if token.type == tokenize.STRING
    ]


# --------------------------------------------------------------------------
# Paths go through pathlib, always
# --------------------------------------------------------------------------


def test_no_module_uses_os_path() -> None:
    """``os.path`` joins with the host separator and invites string paths."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in python_files()
        if "os.path" in code_of(path)
    ]
    assert offenders == []


def test_no_module_uses_os_sep_or_makes_directories_by_hand() -> None:
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in python_files()
        if re.search(r"\bos\s*\.\s*(sep|getcwd|makedirs|mkdir)\b", code_of(path))
    ]
    assert offenders == []


def test_no_string_literal_contains_a_windows_path_separator() -> None:
    """A literal backslash separator is a path that only works on Windows."""
    offenders: list[str] = []
    for path in python_files():
        for token in string_literals(path):
            text = token.string
            prefix = text[: text.index(text.lstrip("rRbBuUfF")[0])].lower()
            if "r" in prefix:
                continue  # a raw string here is a regex, not a path
            if re.search(r"\\\\[a-zA-Z0-9_.]", text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{token.start[0]}")
    assert offenders == []


def test_no_string_literal_hardcodes_a_windows_drive() -> None:
    for path in python_files():
        for token in string_literals(path):
            assert not re.search(r"[A-Z]:[\\/]", token.string), path.name


def test_no_path_is_built_by_concatenating_a_separator() -> None:
    """Catches ``directory + "/" + name`` and ``f"{directory}/{name}"``."""
    concat = re.compile(r"\+\s*[\"']/[a-zA-Z0-9_.]|\{[a-z_]+\}/[a-zA-Z0-9_.]+[\"']")
    offenders = []
    for path in python_files():
        for number, line in enumerate(source_of(path).splitlines(), start=1):
            if "http" in line:
                continue
            if concat.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert offenders == []


def test_every_directory_helper_returns_a_path() -> None:
    for helper in (runs_dir, data_dir, personas_dir, ground_truth_dir, checkpoint_db):
        assert isinstance(helper(), Path), helper.__name__


def test_output_directories_are_created_with_parents(tmp_path, monkeypatch) -> None:
    """``parents=True, exist_ok=True`` - no manual mkdir ladder, no race."""
    monkeypatch.setenv("DTV_REA_RUNS_DIR", str(tmp_path / "a" / "b" / "runs"))
    directory = session_dir("deep-session")
    assert directory.is_dir()
    assert checkpoint_db().parent.is_dir()
    # Calling twice must not raise.
    assert session_dir("deep-session") == directory


def test_the_runs_directory_can_be_relocated_by_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DTV_REA_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert runs_dir() == (tmp_path / "elsewhere").resolve()


# --------------------------------------------------------------------------
# Text I/O is explicitly UTF-8
# --------------------------------------------------------------------------


def _calls_missing_encoding(path: Path) -> list[str]:
    """Find open/read_text/write_text calls with no ``encoding`` argument."""
    tree = ast.parse(source_of(path))
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in {"open", "read_text", "write_text"}:
            continue
        if any(keyword.arg == "encoding" for keyword in node.keywords):
            continue
        # Binary mode needs no encoding.
        if name == "open" and any(
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and "b" in argument.value
            for argument in node.args
        ):
            continue
        problems.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {name}()")
    return problems


def test_every_text_file_is_read_and_written_as_utf8() -> None:
    """The default encoding differs by platform; on Windows it is not UTF-8."""
    offenders: list[str] = []
    for path in python_files(include_guard=True):
        offenders.extend(_calls_missing_encoding(path))
    assert offenders == []


def test_data_files_are_valid_utf8_json() -> None:
    for path in sorted(data_dir().rglob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# No platform-specific behaviour, no shelling out
# --------------------------------------------------------------------------


def test_nothing_branches_on_the_operating_system() -> None:
    """One code path. ``sys.platform`` checks are how the two versions diverge."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in python_files()
        if re.search(
            r"sys\s*\.\s*platform|platform\s*\.\s*system|os\s*\.\s*name\b",
            code_of(path),
        )
    ]
    assert offenders == []


def test_nothing_shells_out() -> None:
    """A subprocess call is a shell dependency, and shells differ."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in python_files()
        if re.search(
            r"\bsubprocess\b|os\s*\.\s*(system|popen)\b", code_of(path)
        )
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# Packaging promises
# --------------------------------------------------------------------------


def test_requirements_txt_and_pyproject_agree() -> None:
    """A teammate installing either way must get the same thing."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    named = {
        line.split(">=")[0].split("[")[0].strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    }
    for package in (
        "langgraph",
        "langgraph-checkpoint-sqlite",
        "langchain-groq",
        "pydantic",
        "tenacity",
        "python-dotenv",
    ):
        assert package in named, f"{package} missing from requirements.txt"
        assert package in pyproject, f"{package} missing from pyproject.toml"

    # requirements.txt must also install the test runner, so that one install
    # command is enough to run the documented `python -m pytest`.
    assert "pytest" in named


def test_dependencies_are_ranges_not_exact_pins() -> None:
    """An exact pin resolved on macOS can fail to resolve on Windows."""
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" not in line, line
        assert ">=" in line, line


def test_python_311_is_the_stated_floor() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11"' in pyproject


def test_the_env_example_exists_and_documents_the_key() -> None:
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GROQ_API_KEY=" in example
    assert "console.groq.com" in example


def test_a_real_env_file_is_never_committed() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]
    assert "runs/" in [line.strip() for line in ignored]


def test_the_api_key_is_never_hardcoded_in_source() -> None:
    """The key belongs in .env, which is git-ignored - never in a module."""
    looks_like_a_key = re.compile(r"gsk_[A-Za-z0-9]{20,}")
    for path in python_files(include_guard=True):
        assert not looks_like_a_key.search(source_of(path)), path.relative_to(REPO_ROOT)


def test_the_key_is_never_printed_or_logged() -> None:
    for path in python_files(include_guard=True):
        text = source_of(path)
        for pattern in (
            r"print\([^)]*GROQ_API_KEY",
            r"log\([^)]*api_key",
            r'environ\["GROQ_API_KEY"\]\s*\)',
        ):
            assert not re.search(pattern, text), path.relative_to(REPO_ROOT)


# --------------------------------------------------------------------------
# Convenience scripts must exist for both platforms and agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["setup", "demo"])
def test_both_a_shell_and_a_powershell_script_are_provided(stem: str) -> None:
    scripts = REPO_ROOT / "scripts"
    assert (scripts / f"{stem}.sh").exists(), f"{stem}.sh missing"
    assert (scripts / f"{stem}.ps1").exists(), f"{stem}.ps1 missing"


@pytest.mark.parametrize("stem", ["setup", "demo"])
def test_the_two_scripts_run_the_same_python_commands(stem: str) -> None:
    """Identical behaviour, not merely a file with the same name."""
    scripts = REPO_ROOT / "scripts"
    pattern = re.compile(r"python -m [a-z_.]+[^\r\n\"']*")

    def commands(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8")
        return [match.strip() for match in pattern.findall(text)]

    shell = commands(scripts / f"{stem}.sh")
    powershell = commands(scripts / f"{stem}.ps1")
    assert shell, f"{stem}.sh runs no python commands"
    assert shell == powershell, f"{stem}.sh and {stem}.ps1 have diverged"


# --------------------------------------------------------------------------
# The README's promises
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_gives_both_macos_and_windows_instructions(readme: str) -> None:
    for step in ("macOS / Linux", "Windows"):
        # Every setup step must appear for both platforms.
        assert readme.count(step) >= 3, step


def test_the_readme_covers_every_setup_step_on_both_platforms(readme: str) -> None:
    assert "python3 -m venv .venv" in readme          # macOS/Linux
    assert "py -3 -m venv .venv" in readme            # Windows
    assert "source .venv/bin/activate" in readme      # macOS/Linux
    assert ".venv\\Scripts\\Activate.ps1" in readme   # Windows
    assert "cp .env.example .env" in readme           # macOS/Linux
    assert "copy .env.example .env" in readme         # Windows


def test_the_readme_gives_one_identical_run_command(readme: str) -> None:
    assert "python -m dtv_rea.cli" in readme
    assert "python -m pytest" in readme


def test_the_readme_says_where_the_output_goes(readme: str) -> None:
    assert "runs/<session_id>/" in readme
    assert "requirements.md" in readme


def test_the_readme_troubleshoots_the_four_named_problems(readme: str) -> None:
    lowered = readme.lower()
    assert "python not found" in lowered
    assert "pip not found" in lowered
    assert "set-executionpolicy" in lowered
    assert "groq_api_key" in lowered


def test_the_readme_links_to_python_org(readme: str) -> None:
    assert "python.org" in readme
    assert "Add Python to PATH" in readme


def test_the_readme_records_the_architecture_decisions(readme: str) -> None:
    for decision in ("LangGraph", "No RAG", "Human in the loop", "hybrid"):
        assert decision.lower() in readme.lower(), decision


def test_the_readme_states_the_no_rag_revisit_trigger(readme: str) -> None:
    assert "revisit" in readme.lower()
    assert "context window" in readme.lower()


def test_the_readme_is_honest_about_limitations(readme: str) -> None:
    lowered = readme.lower()
    assert "limitations" in lowered
    assert "n = 1" in lowered or "n=1" in lowered
    assert "keyword heuristic" in lowered
    assert "single stakeholder" in lowered
