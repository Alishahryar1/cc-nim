import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_text(name: str) -> str:
    return (_repo_root() / "scripts" / name).read_text(encoding="utf-8")


def _braced_body(text: str, declaration: str) -> str:
    start = text.index(declaration)
    brace_start = text.index("{", start)
    depth = 0

    for index, char in enumerate(text[brace_start:], start=brace_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index]

    raise AssertionError(f"Unclosed function body for {declaration}")


def _path_without_uv() -> str:
    uv_names = ("uv", "uv.exe", "uv.cmd", "uv.bat")
    entries = []
    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_entry:
            continue
        entry = Path(raw_entry)
        if any((entry / name).exists() for name in uv_names):
            continue
        entries.append(raw_entry)
    return os.pathsep.join(entries)


def _shell_interpreter() -> str:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("sh is not available on this platform")
    return sh


def _powershell_interpreter() -> str:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("PowerShell is not available on this platform")
    return pwsh


def test_ci_sh_runs_ci_checks_in_order() -> None:
    text = _script_text("ci.sh")

    assert 'CHECK_ORDER="suppressions ruff-format ruff-check ty pytest"' in text
    assert "grep -rE" in text
    assert "Fix the underlying type errors instead" in text
    assert "--exclude-dir=.venv" in text
    assert "--exclude-dir=.git" in text
    assert "uv run ruff format --check" in text
    assert "uv run ruff check" in text
    assert "uv run ty check" in text
    assert "uv run pytest -v --tb=short" in text
    assert "--only" in text
    assert "--skip" in text
    assert "--dry-run" in text
    assert "uv is required but was not found on PATH" in text
    assert "npm" not in text
    assert "smoke/" not in text
    assert "uv self update" not in text


def test_ci_sh_dry_run_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _shell_interpreter(),
            str(_repo_root() / "scripts" / "ci.sh"),
            "--only",
            "pytest",
            "--dry-run",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "+ uv run pytest -v --tb=short" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_sh_suppression_only_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _shell_interpreter(),
            str(_repo_root() / "scripts" / "ci.sh"),
            "--only",
            "suppressions",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Ban type ignore suppressions" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_sh_is_tracked_executable() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "scripts/ci.sh"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.startswith("100755 ")


def test_ci_sh_fail_fast_runs_checks_sequentially() -> None:
    text = _script_text("ci.sh")
    main = text[text.index('parse_args "$@"') :]

    suppress_index = text.index("run_suppressions()")
    ruff_format_index = text.index("run_ruff_format()")
    ruff_check_index = text.index("run_ruff_check()")
    ty_index = text.index("run_ty()")
    pytest_index = text.index("run_pytest()")

    assert (
        suppress_index < ruff_format_index < ruff_check_index < ty_index < pytest_index
    )
    assert "for check_id in $CHECK_ORDER" in main


def test_ci_ps1_runs_ci_checks_in_order() -> None:
    text = _script_text("ci.ps1")

    assert '"suppressions"' in text
    assert '"ruff-format"' in text
    assert '"ruff-check"' in text
    assert '"ty"' in text
    assert '"pytest"' in text
    assert "Select-String -Pattern" in text
    assert "Fix the underlying type errors instead" in text
    assert ".venv" in text
    assert ".git" in text
    assert '"format", "--check"' in text
    assert '"-v", "--tb=short"' in text
    assert "-Only" in text
    assert "-Skip" in text
    assert "-DryRun" in text
    assert "uv is required but was not found on PATH" in text
    assert "npm" not in text
    assert "smoke/" not in text
    assert "uv self update" not in text


def test_ci_ps1_dry_run_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            str(_repo_root() / "scripts" / "ci.ps1"),
            "-Only",
            "pytest",
            "-DryRun",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "+ uv run pytest -v --tb=short" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_ps1_suppression_only_does_not_require_uv() -> None:
    result = subprocess.run(
        [
            _powershell_interpreter(),
            "-NoProfile",
            "-File",
            str(_repo_root() / "scripts" / "ci.ps1"),
            "-Only",
            "suppressions",
        ],
        cwd=_repo_root(),
        env={**os.environ, "PATH": _path_without_uv()},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Ban type ignore suppressions" in result.stdout
    assert "uv is required" not in result.stderr


def test_ci_ps1_fail_fast_runs_checks_sequentially() -> None:
    text = _script_text("ci.ps1")

    assert "foreach ($checkId in $CheckOrder)" in text
    assert "Invoke-SuppressionsCheck" in text
    assert "Invoke-RuffFormatCheck" in text
    assert "Invoke-RuffLintCheck" in text
    assert "Invoke-TyCheck" in text
    assert "Invoke-PytestCheck" in text

    suppress_index = text.index("function Invoke-SuppressionsCheck")
    ruff_format_index = text.index("function Invoke-RuffFormatCheck")
    ruff_check_index = text.index("function Invoke-RuffLintCheck")
    ty_index = text.index("function Invoke-TyCheck")
    pytest_index = text.index("function Invoke-PytestCheck")

    assert (
        suppress_index < ruff_format_index < ruff_check_index < ty_index < pytest_index
    )
