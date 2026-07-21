import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

FCC_COMMANDS = (
    "fcc-desktop",
    "fcc-server",
    "fcc-claude",
    "fcc-codex",
    "fcc-pi",
    "fcc-init",
    "free-claude-code",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


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


def _posix_command(name: str) -> str:
    help_output = (
        '    echo "  --extension, -e <path>  Load an extension"\n'
        '    echo "  --models <patterns>     Scope models"'
        if name == "pi"
        else "    :"
    )
    return f"""#!/bin/sh
echo "{name}:$*" >> "$CALL_LOG"
if [ "$FAIL_STEP" = "{name}-verify" ]; then
    exit 31
fi
if [ "${{1:-}}" = "--version" ]; then
    echo "{name} 1.0.0"
fi
if [ "${{1:-}}" = "--help" ]; then
{help_output}
fi
"""


def _posix_npm_command() -> str:
    return """#!/bin/sh
echo "npm:$*" >> "$CALL_LOG"
if [ "${1:-}" = "prefix" ] && [ "${2:-}" = "-g" ]; then
    printf '%s\n' "$FAKE_NPM_PREFIX"
    exit 0
fi
if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ] && [ "${3:-}" = "prefix" ]; then
    printf '%s\n' "$FAKE_NPM_PREFIX"
    exit 0
fi
exit 71
"""


def _posix_uv_command(version: str) -> str:
    return f"""#!/bin/sh
echo "uv:$*" >> "$CALL_LOG"
if [ "${{1:-}}" = "--version" ]; then
    if [ "${{FCC_RUNNING_PHASE:-}}" = "late" ]; then
        : > "$FCC_PROCESS_MARKER"
    fi
    if [ "$FAIL_STEP" = "uv-verify" ]; then
        exit 32
    fi
    echo "uv {version}"
    exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "install" ]; then
    if [ "$FAIL_STEP" = "fcc-install" ]; then
        exit 33
    fi
    mkdir -p "$FAKE_TOOL_BIN"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$FAKE_TOOL_BIN/fcc-server"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$FAKE_TOOL_BIN/fcc-desktop"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$FAKE_TOOL_BIN/fcc-claude"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$FAKE_TOOL_BIN/fcc-pi"
    if [ "$FAIL_STEP" != "fcc-missing" ]; then
        cp "$FAKE_FIXTURES/fcc-command.sh" "$FAKE_TOOL_BIN/fcc-codex"
    fi
    chmod +x "$FAKE_TOOL_BIN"/fcc-*
    exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "update-shell" ]; then
    if [ "$FAIL_STEP" = "path-update" ]; then
        exit 34
    fi
    exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "dir" ] && [ "${{3:-}}" = "--bin" ]; then
    printf '%s\n' "$FAKE_TOOL_BIN"
    exit 0
fi
exit 35
"""


@dataclass
class PosixHarness:
    root: Path
    bin_dir: Path
    fixtures: Path
    tool_bin: Path
    log: Path
    env: dict[str, str]

    def add_client(self, name: str) -> None:
        _write_executable(self.bin_dir / name, _posix_command(name))

    def add_unrelated_pi(self) -> None:
        _write_executable(self.bin_dir / "pi", _posix_command("unrelated-pi"))

    def add_npm_prefix(self, prefix: Path) -> None:
        prefix.mkdir(parents=True)
        self.env["FAKE_NPM_PREFIX"] = str(prefix)
        _write_executable(self.bin_dir / "npm", _posix_npm_command())

    def add_uv(self, version: str) -> None:
        _write_executable(self.bin_dir / "uv", _posix_uv_command(version))

    def use_process_list_fallback(self, process_line: str) -> None:
        fallback_bin = self.root / "fallback-bin"
        fallback_bin.mkdir()
        _write_executable(
            fallback_bin / "ps",
            """#!/bin/sh
printf '%s\n' "$FCC_PS_OUTPUT"
""",
        )
        awk = shutil.which("awk", path=self.env["PATH"])
        if awk is None:
            pytest.skip("awk is required for the POSIX process fallback scenario")
        shutil.copy2(awk, fallback_bin / "awk")
        self.env["FCC_PS_OUTPUT"] = process_line
        self.env["PATH"] = str(fallback_bin)

    def run(self, *args: str, fail_step: str = "") -> subprocess.CompletedProcess[str]:
        env = self.env | {"FAIL_STEP": fail_step}
        return subprocess.run(
            ["/bin/sh", str(_repo_root() / "scripts" / "install.sh"), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def posix_harness(tmp_path: Path) -> PosixHarness:
    if os.name == "nt":
        pytest.skip("POSIX installer scenarios run on POSIX hosts")

    bin_dir = tmp_path / "bin"
    fixtures = tmp_path / "fixtures"
    tool_bin = tmp_path / "tool-bin"
    home = tmp_path / "home"
    log = tmp_path / "calls.log"
    for path in (bin_dir, fixtures, tool_bin, home):
        path.mkdir(parents=True)

    _write_executable(
        bin_dir / "pgrep",
        """#!/bin/sh
[ -n "${FCC_RUNNING_COMMAND:-}" ] || exit 1
if [ "${FCC_RUNNING_PHASE:-early}" = "late" ] && [ ! -e "$FCC_PROCESS_MARKER" ]; then
    exit 1
fi
case "$*" in
    *"$FCC_RUNNING_COMMAND"*) printf '4242\n'; exit 0 ;;
    *) exit 1 ;;
esac
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/bin/sh
url=""
output=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o)
            shift
            output=$1
            ;;
        http*)
            url=$1
            ;;
    esac
    shift
done
echo "download:$url" >> "$CALL_LOG"
case "$url:$FAIL_STEP" in
    *claude.ai*:claude-download|*chatgpt.com*:codex-download|*pi.dev*:pi-download|*astral.sh*:uv-download)
        exit 41
        ;;
esac
case "$url" in
    *claude.ai*) source="$FAKE_FIXTURES/claude-installer.sh" ;;
    *chatgpt.com*) source="$FAKE_FIXTURES/codex-installer.sh" ;;
    *pi.dev*) source="$FAKE_FIXTURES/pi-installer.sh" ;;
    *astral.sh*) source="$FAKE_FIXTURES/uv-installer.sh" ;;
    *) exit 42 ;;
esac
cp "$source" "$output"
""",
    )
    _write_executable(
        fixtures / "claude-installer.sh",
        """#!/bin/sh
echo "claude-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "claude-install" ] && exit 21
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/claude-command.sh" "$HOME/.local/bin/claude"
chmod +x "$HOME/.local/bin/claude"
""",
    )
    _write_executable(
        fixtures / "codex-installer.sh",
        """#!/bin/sh
echo "codex-install:$CODEX_NON_INTERACTIVE" >> "$CALL_LOG"
[ "$FAIL_STEP" = "codex-install" ] && exit 22
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/codex-command.sh" "$HOME/.local/bin/codex"
chmod +x "$HOME/.local/bin/codex"
""",
    )
    _write_executable(
        fixtures / "pi-installer.sh",
        """#!/bin/sh
echo "pi-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "pi-install" ] && exit 24
if [ -n "${FAKE_NPM_PREFIX:-}" ]; then
    pi_bin="$FAKE_NPM_PREFIX/bin"
else
    pi_bin="$HOME/.local/bin"
fi
mkdir -p "$pi_bin"
cp "$FAKE_FIXTURES/pi-command.sh" "$pi_bin/pi"
chmod +x "$pi_bin/pi"
""",
    )
    _write_executable(
        fixtures / "uv-installer.sh",
        """#!/bin/sh
echo "uv-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "uv-install" ] && exit 23
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/uv-command.sh" "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
""",
    )
    _write_executable(fixtures / "claude-command.sh", _posix_command("claude"))
    _write_executable(fixtures / "codex-command.sh", _posix_command("codex"))
    _write_executable(fixtures / "pi-command.sh", _posix_command("pi"))
    _write_executable(fixtures / "uv-command.sh", _posix_uv_command("0.11.28"))
    _write_executable(
        fixtures / "fcc-command.sh",
        """#!/bin/sh
name=${0##*/}
echo "$name:$*" >> "$CALL_LOG"
if [ "$FAIL_STEP" = "fcc-verify" ]; then
    exit 36
fi
if [ "$name" = "fcc-server" ] && [ "${1:-}" = "--version" ]; then
    echo "free-claude-code 3.5.18"
fi
""",
    )
    _write_executable(
        bin_dir / "uname",
        """#!/bin/sh
printf '%s\n' "${FAKE_UNAME:-Linux}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(home),
            "CALL_LOG": str(log),
            "FAKE_FIXTURES": str(fixtures),
            "FAKE_TOOL_BIN": str(tool_bin),
            "FCC_PROCESS_MARKER": str(tmp_path / "fcc-process-ready"),
            "FCC_RUNNING_COMMAND": "",
            "FCC_RUNNING_PHASE": "early",
            "FAKE_UNAME": "Linux",
            "FAIL_STEP": "",
        }
    )
    env.pop("XDG_BIN_HOME", None)
    return PosixHarness(tmp_path, bin_dir, fixtures, tool_bin, log, env)


def test_install_sh_fresh_install_is_verified(posix_harness: PosixHarness) -> None:
    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Free Claude Code is installed and verified." in result.stdout
    calls = posix_harness.calls()
    assert calls.index("claude-install") < calls.index("claude:--version")
    assert calls.index("codex-install:1") < calls.index("codex:--version")
    assert calls.index("pi-install") < calls.index("pi:--version")
    assert calls.index("uv-install") < calls.index("uv:--version")
    assert any(
        call.startswith(
            "uv:tool install --force --refresh-package free-claude-code "
            "--python 3.14.0 free-claude-code @ "
            "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
        )
        for call in calls
    )
    assert not any(call.startswith("git:") for call in calls)
    assert calls[-3:] == [
        "uv:tool update-shell",
        "uv:tool dir --bin",
        "fcc-server:--version",
    ]


def test_install_sh_creates_native_macos_app_and_desktop_link(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    tool_bin = posix_harness.root / "tool's bin"
    posix_harness.env["FAKE_TOOL_BIN"] = str(tool_bin)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    app = posix_harness.root / "home" / "Applications" / "Free Claude Code.app"
    plist = app / "Contents" / "Info.plist"
    owner_file = app / "Contents" / ".free-claude-code-owner"
    launcher = app / "Contents" / "MacOS" / "fcc-desktop"
    desktop_link = posix_harness.root / "home" / "Desktop" / "Free Claude Code.app"
    assert owner_file.read_text(encoding="utf-8").strip() == (
        "io.github.alishahryar1.free-claude-code"
    )
    assert "<key>LSUIElement</key>" in plist.read_text(encoding="utf-8")
    assert "<key>LSMultipleInstancesProhibited</key>" in plist.read_text(
        encoding="utf-8"
    )
    assert launcher.stat().st_mode & 0o111
    expected_command = str(tool_bin / "fcc-desktop").replace("'", "'\\''")
    assert f"exec '{expected_command}'" in launcher.read_text(encoding="utf-8")
    assert desktop_link.is_symlink()
    assert desktop_link.readlink() == app


def test_install_sh_rejects_unowned_macos_app_bundle(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    app = posix_harness.root / "home" / "Applications" / "Free Claude Code.app"
    contents = app / "Contents"
    contents.mkdir(parents=True)
    plist = contents / "Info.plist"
    plist.write_text("foreign app", encoding="utf-8")

    result = posix_harness.run()

    assert result.returncode != 0
    assert "not managed by Free Claude Code" in result.stderr
    assert plist.read_text(encoding="utf-8") == "foreign app"
    assert not (contents / ".free-claude-code-owner").exists()


def test_install_sh_preserves_unrelated_macos_desktop_link(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    desktop = posix_harness.root / "home" / "Desktop"
    desktop.mkdir()
    unrelated = posix_harness.root / "Unrelated.app"
    unrelated.mkdir()
    desktop_link = desktop / "Free Claude Code.app"
    desktop_link.symlink_to(unrelated, target_is_directory=True)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "non-FCC link" in result.stdout
    assert desktop_link.readlink() == unrelated


@pytest.mark.parametrize("uv_version", ("0.11.16", "0.11.16+build.1"))
def test_install_sh_preserves_valid_existing_tools(
    posix_harness: PosixHarness,
    uv_version: str,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv(uv_version)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("download:") for call in posix_harness.calls())
    assert "leaving it unchanged" in result.stdout


def test_install_sh_replaces_unrelated_pi_command(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_unrelated_pi()
    posix_harness.add_uv("0.11.16")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "is not Pi Coding Agent; installing Pi" in result.stdout
    assert "pi-install" in posix_harness.calls()


def test_install_sh_discovers_custom_pi_npm_prefix(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_npm_prefix(posix_harness.root / "custom-npm")
    posix_harness.add_uv("0.11.16")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    calls = posix_harness.calls()
    assert "npm:prefix -g" in calls
    assert "pi:--help" in calls
    assert "pi:--version" in calls


def test_install_sh_replaces_obsolete_uv(posix_harness: PosixHarness) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv("0.5.9")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "uv 0.5.9 does not satisfy stable >=0.11.16" in result.stdout
    assert "uv-install" in posix_harness.calls()


@pytest.mark.parametrize("version", ("0.11.16-alpha.1", "0.12.0-rc.1"))
def test_install_sh_replaces_prerelease_uv(
    posix_harness: PosixHarness,
    version: str,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv(version)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert f"uv {version} does not satisfy stable >=0.11.16" in result.stdout
    assert "uv-install" in posix_harness.calls()


@pytest.mark.parametrize(
    "failure",
    [
        "claude-download",
        "claude-install",
        "claude-verify",
        "codex-download",
        "codex-install",
        "codex-verify",
        "pi-download",
        "pi-install",
        "pi-verify",
        "uv-download",
        "uv-install",
        "uv-verify",
        "fcc-install",
        "path-update",
        "fcc-missing",
        "fcc-verify",
    ],
)
def test_install_sh_stops_without_success_on_each_failure(
    posix_harness: PosixHarness,
    failure: str,
) -> None:
    result = posix_harness.run(fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    forbidden = {
        "claude-download": "claude-install",
        "claude-install": "claude:--version",
        "claude-verify": "chatgpt.com",
        "codex-download": "codex-install",
        "codex-install": "codex:--version",
        "codex-verify": "pi.dev",
        "pi-download": "pi-install",
        "pi-install": "pi:--version",
        "pi-verify": "astral.sh",
        "uv-download": "uv-install",
        "uv-install": "uv:--version",
        "uv-verify": "uv:tool install",
        "fcc-install": "uv:tool update-shell",
        "path-update": "uv:tool dir --bin",
        "fcc-missing": "fcc-server:--version",
    }.get(failure)
    if forbidden is not None:
        assert not any(forbidden in call for call in posix_harness.calls())


def test_install_sh_dry_run_never_executes_commands(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert posix_harness.calls() == []
    assert "Dry run complete. No changes were made." in result.stdout
    assert "Free Claude Code is installed and verified." not in result.stdout


def test_install_sh_rejects_broken_existing_client_without_replacing_it(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")

    result = posix_harness.run(fail_step="claude-verify")

    assert result.returncode != 0
    assert not any(call.startswith("download:") for call in posix_harness.calls())


def test_install_sh_rejects_unparseable_existing_uv(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv("not-a-version")

    result = posix_harness.run()

    assert result.returncode != 0
    assert not any("astral.sh" in call for call in posix_harness.calls())


def test_install_sh_voice_flags_only_change_fcc_spec(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--voice-all", "--torch-backend", "cu130")

    assert result.returncode == 0, result.stderr
    assert any(
        "--torch-backend cu130 free-claude-code[voice,voice_local] @ "
        "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
        in call
        for call in posix_harness.calls()
    )


def test_install_sh_rejects_invalid_options_before_mutation(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--torch-backend", "cu130")

    assert result.returncode != 0
    assert posix_harness.calls() == []


@pytest.mark.parametrize("command_name", FCC_COMMANDS)
def test_install_sh_rejects_running_fcc_before_mutation(
    posix_harness: PosixHarness,
    command_name: str,
) -> None:
    posix_harness.env["FCC_RUNNING_COMMAND"] = command_name

    result = posix_harness.run()

    assert result.returncode != 0
    assert posix_harness.calls() == []
    assert f"{command_name} (PID 4242)" in result.stderr


def test_install_sh_rechecks_for_fcc_process_before_tool_replacement(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv("0.11.16")
    posix_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server"
    posix_harness.env["FCC_RUNNING_PHASE"] = "late"

    result = posix_harness.run()

    assert result.returncode != 0
    assert "fcc-server (PID 4242)" in result.stderr
    assert not any(call.startswith("uv:tool install") for call in posix_harness.calls())


def test_install_sh_ignores_similarly_named_process(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server-helper"

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("command_name", "process_args"),
    (
        ("free-claude-code", "/home/user/.local/bin/free-claude-code"),
        ("fcc-server", "/usr/bin/python3 /home/user/.local/bin/fcc-server"),
    ),
)
def test_install_sh_process_fallback_reads_full_command_line(
    posix_harness: PosixHarness,
    command_name: str,
    process_args: str,
) -> None:
    posix_harness.use_process_list_fallback(f"4242 {process_args}")

    result = posix_harness.run()

    assert result.returncode != 0
    assert posix_harness.calls() == []
    assert f"{command_name} (PID 4242)" in result.stderr


def _powershells() -> tuple[str, ...]:
    candidates = (shutil.which("pwsh"), shutil.which("powershell"))
    return tuple(dict.fromkeys(path for path in candidates if path is not None))


def _create_windows_shortcut(
    powershell: str,
    shortcut_path: Path,
    target_path: Path,
) -> None:
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "FCC_TEST_SHORTCUT": str(shortcut_path),
        "FCC_TEST_TARGET": str(target_path),
    }
    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-Command",
            (
                "$shell = New-Object -ComObject WScript.Shell; "
                "$shortcut = $shell.CreateShortcut($env:FCC_TEST_SHORTCUT); "
                "$shortcut.TargetPath = $env:FCC_TEST_TARGET; "
                "$shortcut.Save()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _batch_client(name: str) -> str:
    help_output = (
        "echo   --extension, -e ^<path^>  Load an extension\n"
        "echo   --models ^<patterns^>     Scope models"
        if name == "pi"
        else "rem no product help"
    )
    return f"""@echo off
echo {name}:%*>>"%CALL_LOG%"
if "%FAIL_STEP%"=="{name}-verify" exit /b 51
if "%1"=="--version" echo {name} 1.0.0
if "%1"=="--help" (
{help_output}
)
exit /b 0
"""


def _batch_npm() -> str:
    return r"""@echo off
echo npm:%*>>"%CALL_LOG%"
if "%1"=="prefix" if "%2"=="-g" echo %FAKE_NPM_PREFIX%& exit /b 0
if "%1"=="config" if "%2"=="get" if "%3"=="prefix" echo %FAKE_NPM_PREFIX%& exit /b 0
exit /b 71
"""


def _batch_uv(version: str) -> str:
    return rf"""@echo off
echo uv:%*>>"%CALL_LOG%"
if "%1"=="--version" goto version
if "%1"=="tool" if "%2"=="install" goto install
if "%1"=="tool" if "%2"=="update-shell" goto update_shell
if "%1"=="tool" if "%2"=="dir" if "%3"=="--bin" goto tool_bin
exit /b 59
:version
if "%FCC_RUNNING_PHASE%"=="late" type nul > "%FCC_PROCESS_MARKER%"
if "%FAIL_STEP%"=="uv-verify" exit /b 52
echo uv {version}
exit /b 0
:install
if "%FAIL_STEP%"=="fcc-install" exit /b 53
if not exist "%FAKE_TOOL_BIN%" mkdir "%FAKE_TOOL_BIN%"
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-server.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-desktop.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-claude.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-pi.cmd" >nul
if not "%FAIL_STEP%"=="fcc-missing" copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-codex.cmd" >nul
exit /b 0
:update_shell
if "%FAIL_STEP%"=="path-update" exit /b 54
exit /b 0
:tool_bin
echo %FAKE_TOOL_BIN%
exit /b 0
"""


@dataclass
class PowerShellHarness:
    root: Path
    bin_dir: Path
    fixtures: Path
    tool_bin: Path
    log: Path
    env: dict[str, str]
    powershell: str
    wrapper: Path

    def add_client(self, name: str) -> None:
        _write_executable(self.bin_dir / f"{name}.cmd", _batch_client(name))

    def add_unrelated_pi(self) -> None:
        _write_executable(self.bin_dir / "pi.cmd", _batch_client("unrelated-pi"))

    def add_npm_prefix(self, prefix: Path) -> None:
        prefix.mkdir(parents=True)
        self.env["FAKE_NPM_PREFIX"] = str(prefix)
        _write_executable(self.bin_dir / "npm.cmd", _batch_npm())

    def add_uv(self, version: str) -> None:
        _write_executable(self.bin_dir / "uv.cmd", _batch_uv(version))

    def run(self, *args: str, fail_step: str = "") -> subprocess.CompletedProcess[str]:
        env = self.env | {"FAIL_STEP": fail_step}
        return subprocess.run(
            [
                self.powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.wrapper),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.fixture(
    params=_powershells() or (None,),
    ids=lambda path: Path(path).name if path is not None else "unavailable",
)
def powershell_harness(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> PowerShellHarness:
    powershell = request.param
    if powershell is None or os.name != "nt":
        pytest.skip("PowerShell installer scenarios run on Windows hosts")

    bin_dir = tmp_path / "bin"
    fixtures = tmp_path / "fixtures"
    tool_bin = tmp_path / "tool-bin"
    home = tmp_path / "home"
    local_app_data = tmp_path / "local-app-data"
    app_data = tmp_path / "app-data"
    log = tmp_path / "calls.log"
    for path in (bin_dir, fixtures, tool_bin, home, local_app_data, app_data):
        path.mkdir(parents=True)

    (fixtures / "claude-command.cmd").write_text(
        _batch_client("claude"), encoding="utf-8"
    )
    (fixtures / "codex-command.cmd").write_text(
        _batch_client("codex"), encoding="utf-8"
    )
    (fixtures / "pi-command.cmd").write_text(_batch_client("pi"), encoding="utf-8")
    (fixtures / "uv-command.cmd").write_text(_batch_uv("0.11.28"), encoding="utf-8")
    (fixtures / "fcc-command.cmd").write_text(
        """@echo off
for %%I in ("%~f0") do set "FCC_NAME=%%~nI"
echo %FCC_NAME%:%*>>"%CALL_LOG%"
if "%FAIL_STEP%"=="fcc-verify" exit /b 55
if "%FCC_NAME%"=="fcc-server" if "%1"=="--version" echo free-claude-code 3.5.18
exit /b 0
""",
        encoding="utf-8",
    )
    (fixtures / "claude-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "claude-install") { exit 61 }
$bin = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "claude-command.cmd") (Join-Path $bin "claude.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "claude-install"
""",
        encoding="utf-8",
    )
    (fixtures / "codex-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "codex-install") { exit 62 }
$bin = Join-Path $env:LOCALAPPDATA "Programs\OpenAI\Codex\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "codex-command.cmd") (Join-Path $bin "codex.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "codex-install:$env:CODEX_NON_INTERACTIVE"
""",
        encoding="utf-8",
    )
    (fixtures / "pi-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "pi-install") { exit 64 }
$bin = if ($env:FAKE_NPM_PREFIX) { $env:FAKE_NPM_PREFIX } else { Join-Path $env:APPDATA "npm" }
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "pi-command.cmd") (Join-Path $bin "pi.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "pi-install"
""",
        encoding="utf-8",
    )
    (fixtures / "uv-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "uv-install") { exit 63 }
$bin = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "uv-command.cmd") (Join-Path $bin "uv.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "uv-install"
""",
        encoding="utf-8",
    )

    wrapper = tmp_path / "run-installer.ps1"
    wrapper.write_text(
        """Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Invoke-RestMethod {
    [CmdletBinding()]
    param([string] $Uri, [string] $OutFile)

    Add-Content -LiteralPath $env:CALL_LOG -Value "download:$Uri"
    if (
        ($env:FAIL_STEP -eq "claude-download" -and $Uri.Contains("claude.ai")) -or
        ($env:FAIL_STEP -eq "codex-download" -and $Uri.Contains("chatgpt.com")) -or
        ($env:FAIL_STEP -eq "pi-download" -and $Uri.Contains("pi.dev")) -or
        ($env:FAIL_STEP -eq "uv-download" -and $Uri.Contains("astral.sh"))
    ) {
        throw "simulated download failure"
    }
    if ($Uri.Contains("claude.ai")) {
        $source = Join-Path $env:FAKE_FIXTURES "claude-installer.ps1"
    }
    elseif ($Uri.Contains("chatgpt.com")) {
        $source = Join-Path $env:FAKE_FIXTURES "codex-installer.ps1"
    }
    elseif ($Uri.Contains("pi.dev")) {
        $source = Join-Path $env:FAKE_FIXTURES "pi-installer.ps1"
    }
    elseif ($Uri.Contains("astral.sh")) {
        $source = Join-Path $env:FAKE_FIXTURES "uv-installer.ps1"
    }
    else {
        throw "unexpected installer URL: $Uri"
    }
    Copy-Item -LiteralPath $source -Destination $OutFile -Force
}
function Get-Process {
    [CmdletBinding()]
    param([string[]] $Name)

    if ([string]::IsNullOrWhiteSpace($env:FCC_RUNNING_COMMAND)) {
        return
    }
    if (
        $env:FCC_RUNNING_PHASE -eq "late" -and
        -not (Test-Path -LiteralPath $env:FCC_PROCESS_MARKER)
    ) {
        return
    }
    foreach ($requestedName in $Name) {
        if ($requestedName -eq $env:FCC_RUNNING_COMMAND) {
            [pscustomobject] @{ Id = 4242; ProcessName = $requestedName }
        }
    }
}
$installer = [scriptblock]::Create([IO.File]::ReadAllText($env:FCC_INSTALLER))
& $installer @args
""",
        encoding="utf-8",
    )

    system_root = os.environ["SYSTEMROOT"]
    env = os.environ.copy()
    env.update(
        {
            "PATH": os.pathsep.join(
                [str(bin_dir), str(Path(system_root) / "System32"), system_root]
            ),
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "USERPROFILE": str(home),
            "LOCALAPPDATA": str(local_app_data),
            "APPDATA": str(app_data),
            "CALL_LOG": str(log),
            "FAKE_FIXTURES": str(fixtures),
            "FAKE_TOOL_BIN": str(tool_bin),
            "FCC_INSTALLER": str(_repo_root() / "scripts" / "install.ps1"),
            "FCC_PROCESS_MARKER": str(tmp_path / "fcc-process-ready"),
            "FCC_RUNNING_COMMAND": "",
            "FCC_RUNNING_PHASE": "early",
            "FAIL_STEP": "",
        }
    )
    return PowerShellHarness(
        tmp_path, bin_dir, fixtures, tool_bin, log, env, powershell, wrapper
    )


def test_install_ps1_fresh_install_is_verified(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Free Claude Code is installed and verified." in result.stdout
    calls = powershell_harness.calls()
    assert calls.index("claude-install") < calls.index("claude:--version")
    assert calls.index("codex-install:1") < calls.index("codex:--version")
    assert calls.index("pi-install") < calls.index("pi:--version")
    assert calls.index("uv-install") < calls.index("uv:--version")
    assert any(
        call.startswith(
            "uv:tool install --force --refresh-package free-claude-code "
            '--python 3.14.0 "free-claude-code @ '
            'https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"'
        )
        for call in calls
    )
    assert not any(call.startswith("git:") for call in calls)
    assert calls[-3:] == [
        "uv:tool update-shell",
        "uv:tool dir --bin",
        "fcc-server:--version",
    ]
    home = Path(powershell_harness.env["USERPROFILE"])
    app_data = Path(powershell_harness.env["APPDATA"])
    assert (home / "Desktop" / "Free Claude Code.lnk").is_file()
    assert (
        app_data
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Free Claude Code.lnk"
    ).is_file()


def test_install_ps1_preserves_unowned_desktop_shortcut(
    powershell_harness: PowerShellHarness,
) -> None:
    desktop_shortcut = (
        Path(powershell_harness.env["USERPROFILE"]) / "Desktop" / "Free Claude Code.lnk"
    )
    unrelated_target = powershell_harness.root / "unrelated.cmd"
    unrelated_target.write_text("@echo off\n", encoding="utf-8")
    _create_windows_shortcut(
        powershell_harness.powershell,
        desktop_shortcut,
        unrelated_target,
    )
    original_shortcut = desktop_shortcut.read_bytes()

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "not managed by Free Claude Code" in result.stdout
    assert desktop_shortcut.read_bytes() == original_shortcut


@pytest.mark.parametrize("uv_version", ("0.11.16", "0.11.16+build.1"))
def test_install_ps1_preserves_valid_existing_tools(
    powershell_harness: PowerShellHarness,
    uv_version: str,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv(uv_version)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("download:") for call in powershell_harness.calls())
    assert "leaving it unchanged" in result.stdout


def test_install_ps1_replaces_unrelated_pi_command(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_unrelated_pi()
    powershell_harness.add_uv("0.11.16")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "is not Pi Coding Agent; installing Pi" in result.stdout
    assert "pi-install" in powershell_harness.calls()


def test_install_ps1_discovers_custom_pi_npm_prefix(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_npm_prefix(powershell_harness.root / "custom-npm")
    powershell_harness.add_uv("0.11.16")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    calls = powershell_harness.calls()
    assert "npm:prefix -g" in calls
    assert "pi:--help" in calls
    assert "pi:--version" in calls


def test_install_ps1_replaces_obsolete_uv(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv("0.5.9")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "uv 0.5.9 does not satisfy stable >=0.11.16" in result.stdout
    assert "uv-install" in powershell_harness.calls()


@pytest.mark.parametrize("version", ("0.11.16-alpha.1", "0.12.0-rc.1"))
def test_install_ps1_replaces_prerelease_uv(
    powershell_harness: PowerShellHarness,
    version: str,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv(version)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert f"uv {version} does not satisfy stable >=0.11.16" in result.stdout
    assert "uv-install" in powershell_harness.calls()


@pytest.mark.parametrize(
    "failure",
    [
        "claude-download",
        "claude-install",
        "claude-verify",
        "codex-download",
        "codex-install",
        "codex-verify",
        "pi-download",
        "pi-install",
        "pi-verify",
        "uv-download",
        "uv-install",
        "uv-verify",
        "fcc-install",
        "path-update",
        "fcc-missing",
        "fcc-verify",
    ],
)
def test_install_ps1_stops_without_success_on_each_failure(
    powershell_harness: PowerShellHarness,
    failure: str,
) -> None:
    result = powershell_harness.run(fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    forbidden = {
        "claude-download": "claude-install",
        "claude-install": "claude:--version",
        "claude-verify": "chatgpt.com",
        "codex-download": "codex-install",
        "codex-install": "codex:--version",
        "codex-verify": "pi.dev",
        "pi-download": "pi-install",
        "pi-install": "pi:--version",
        "pi-verify": "astral.sh",
        "uv-download": "uv-install",
        "uv-install": "uv:--version",
        "uv-verify": "uv:tool install",
        "fcc-install": "uv:tool update-shell",
        "path-update": "uv:tool dir --bin",
        "fcc-missing": "fcc-server:--version",
    }.get(failure)
    if forbidden is not None:
        assert not any(forbidden in call for call in powershell_harness.calls())


def test_install_ps1_dry_run_never_executes_commands(
    powershell_harness: PowerShellHarness,
) -> None:
    result = subprocess.run(
        [
            powershell_harness.powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_repo_root() / "scripts" / "install.ps1"),
            "-DryRun",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=powershell_harness.env,
    )

    assert result.returncode == 0, result.stderr
    assert powershell_harness.calls() == []
    assert "Dry run complete. No changes were made." in result.stdout
    assert "Free Claude Code is installed and verified." not in result.stdout


def test_install_ps1_rejects_broken_existing_client_without_replacing_it(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")

    result = powershell_harness.run(fail_step="claude-verify")

    assert result.returncode != 0
    assert not any(call.startswith("download:") for call in powershell_harness.calls())


def test_install_ps1_rejects_unparseable_existing_uv(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv("not-a-version")

    result = powershell_harness.run()

    assert result.returncode != 0
    assert not any("astral.sh" in call for call in powershell_harness.calls())


def test_install_ps1_voice_flags_only_change_fcc_spec(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run("-VoiceAll", "-TorchBackend", "cu130")

    assert result.returncode == 0, result.stderr
    assert any(
        '--torch-backend cu130 "free-claude-code[voice,voice_local] @ '
        'https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"'
        in call
        for call in powershell_harness.calls()
    )


@pytest.mark.parametrize("command_name", FCC_COMMANDS)
def test_install_ps1_rejects_running_fcc_before_mutation(
    powershell_harness: PowerShellHarness,
    command_name: str,
) -> None:
    powershell_harness.env["FCC_RUNNING_COMMAND"] = command_name

    result = powershell_harness.run()

    assert result.returncode != 0
    assert powershell_harness.calls() == []
    assert f"{command_name} (PID 4242)" in result.stderr


def test_install_ps1_rechecks_for_fcc_process_before_tool_replacement(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv("0.11.16")
    powershell_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server"
    powershell_harness.env["FCC_RUNNING_PHASE"] = "late"

    result = powershell_harness.run()

    assert result.returncode != 0
    assert "fcc-server (PID 4242)" in result.stderr
    assert not any(
        call.startswith("uv:tool install") for call in powershell_harness.calls()
    )


def test_install_ps1_ignores_similarly_named_process(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server-helper"

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr


def test_installers_use_native_clients_and_single_python_selection() -> None:
    shell = (_repo_root() / "scripts" / "install.sh").read_text(encoding="utf-8")
    powershell = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")

    for text in (shell, powershell):
        for command_name in FCC_COMMANDS:
            assert command_name in text
        assert "@anthropic-ai/claude-code" not in text
        assert "@openai/codex" not in text
        assert "@earendil-works/pi-coding-agent" not in text
        assert "git+" not in text
        assert "git --version" not in text
        assert (
            "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
            in text
        )
        assert "python install" not in text
        assert "--refresh-package" in text
        assert "tool update-shell" in text
        assert "--python" in text

    assert "https://pi.dev/install.sh" in shell
    assert "https://pi.dev/install.ps1" in powershell


def test_readme_install_section_has_no_manual_git_prerequisite() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    install_section = readme.split("### 1. Install Or Update", 1)[1].split(
        "### 2. Start FCC", 1
    )[0]

    assert "Install Git" not in install_section
    assert "official native installers" not in install_section


@pytest.mark.parametrize("powershell", _powershells())
def test_install_ps1_falls_back_when_pshome_executable_is_unavailable(
    tmp_path: Path,
    powershell: str,
) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(text, "function Get-PowerShellExecutable")
    fallback = tmp_path / "fallback" / "powershell.exe"
    script = tmp_path / "test-powershell-resolution.ps1"
    script.write_text(
        f"""Set-StrictMode -Version Latest
function Get-ApplicationCommand {{
    param([string] $Name)
    return [pscustomobject] @{{ Source = {str(fallback)!r} }}
}}
function Get-PowerShellExecutable {{
{body}
}}
$resolved = Get-PowerShellExecutable -PowerShellHome {str(tmp_path / "missing")!r}
if ($resolved -ne {str(fallback)!r}) {{
    throw "Unexpected fallback: $resolved"
}}
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
