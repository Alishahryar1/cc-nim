from pathlib import Path


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


def test_uninstall_sh_removes_uv_tool_and_purges_fcc_home() -> None:
    text = _script_text("uninstall.sh")
    tool_body = _braced_body(text, "uninstall_free_claude_code()")
    purge_body = _braced_body(text, "purge_fcc_home()")

    assert "Does not remove uv, Claude Code, Codex" in text
    assert "uv tool uninstall" in tool_body
    assert 'PACKAGE_NAME="free-claude-code"' in text
    assert "uv not found on PATH; skipping uv tool uninstall." in tool_body
    assert "rm -rf" in purge_body
    assert ".fcc" in purge_body
    assert "npm uninstall" not in text
    assert "uv self uninstall" not in text
    assert "uv python uninstall" not in text


def test_uninstall_sh_fails_when_fcc_commands_are_running() -> None:
    text = _script_text("uninstall.sh")
    guard_body = _braced_body(text, "assert_no_fcc_processes_running()")
    main = text[text.index('parse_args "$@"') :]

    for command in (
        "fcc-server",
        "fcc-claude",
        "fcc-codex",
        "fcc-init",
        "free-claude-code",
    ):
        assert command in text

    assert "FCC_COMMANDS" in text

    assert "Free Claude Code is still running" in guard_body
    assert (
        'step "Checking for running Free Claude Code processes"\nassert_no_fcc_processes_running'
        in main
    )


def test_uninstall_ps1_removes_uv_tool_and_purges_fcc_home() -> None:
    text = _script_text("uninstall.ps1")
    tool_body = _braced_body(text, "function Uninstall-FreeClaudeCode")
    purge_body = _braced_body(text, "function Purge-FccHome")

    assert "Does not remove uv, Claude Code, Codex" in text
    assert "uv tool uninstall" in tool_body
    assert '$PackageName = "free-claude-code"' in text
    assert "uv not found on PATH; skipping uv tool uninstall." in tool_body
    assert "Remove-Item" in purge_body
    assert '$FccHomeDirname = ".fcc"' in text
    assert "npm uninstall" not in text
    assert "uv self uninstall" not in text
    assert "uv python uninstall" not in text


def test_uninstall_ps1_fails_when_fcc_commands_are_running() -> None:
    text = _script_text("uninstall.ps1")
    guard_body = _braced_body(text, "function Assert-NoFccProcessesRunning")

    for command in (
        "fcc-server",
        "fcc-claude",
        "fcc-codex",
        "fcc-init",
        "free-claude-code",
    ):
        assert command in text

    assert "FccCommands" in text

    assert "Free Claude Code is still running" in guard_body
    assert (
        'Write-Step "Checking for running Free Claude Code processes"\n'
        "Assert-NoFccProcessesRunning" in text
    )
