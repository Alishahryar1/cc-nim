"""Tests for cli/entrypoints.py — fcc-init scaffolding logic."""

from pathlib import Path
from unittest.mock import patch


def _run_init(tmp_home: Path) -> tuple[str, Path]:
    """Run init() with home directory redirected to tmp_home. Returns (printed output, env_file path)."""
    from cli.entrypoints import init

    env_file = tmp_home / ".config" / "free-claude-code" / ".env"
    printed: list[str] = []

    with patch("pathlib.Path.home", return_value=tmp_home), patch(
        "builtins.print",
        side_effect=lambda *a: printed.append(" ".join(str(x) for x in a)),
    ):
        init()

    return "\n".join(printed), env_file


def test_init_creates_env_file(tmp_path: Path) -> None:
    """init() creates .env from the bundled template when it doesn't exist yet."""
    output, env_file = _run_init(tmp_path)

    assert env_file.exists()
    assert env_file.stat().st_size > 0
    assert str(env_file) in output


def test_init_copies_template_content(tmp_path: Path) -> None:
    """init() writes the actual bundled env.example content, not an empty file."""
    import importlib.resources

    template = (
        importlib.resources.files("config").joinpath("env.example").read_text("utf-8")
    )
    _, env_file = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == template


def test_init_creates_parent_directories(tmp_path: Path) -> None:
    """init() creates ~/.config/free-claude-code/ even if it doesn't exist."""
    config_dir = tmp_path / ".config" / "free-claude-code"
    assert not config_dir.exists()

    _run_init(tmp_path)

    assert config_dir.is_dir()


def test_init_skips_if_env_already_exists(tmp_path: Path) -> None:
    """init() does not overwrite an existing .env and prints a warning."""
    # Create it first
    _run_init(tmp_path)

    env_file = tmp_path / ".config" / "free-claude-code" / ".env"
    env_file.write_text("existing content", encoding="utf-8")

    output, _ = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == "existing content"
    assert "already exists" in output


def test_init_prints_next_step_hint(tmp_path: Path) -> None:
    """init() tells the user to run free-claude-code after editing .env."""
    output, _ = _run_init(tmp_path)

    assert "free-claude-code" in output
