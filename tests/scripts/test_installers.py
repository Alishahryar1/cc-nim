import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


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
    return f"""#!/bin/sh
echo "{name}:$*" >> "$CALL_LOG"
if [ "$FAIL_STEP" = "{name}-verify" ]; then
    exit 31
fi
if [ "${{1:-}}" = "--version" ]; then
    echo "{name} 1.0.0"
fi
"""


def _posix_uv_command(version: str) -> str:
    return f"""#!/bin/sh
echo "uv:$*" >> "$CALL_LOG"
if [ "${{1:-}}" = "--version" ]; then
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
    cp "$FAKE_FIXTURES/fcc-command.sh" "$FAKE_TOOL_BIN/fcc-claude"
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

    def add_uv(self, version: str) -> None:
        _write_executable(self.bin_dir / "uv", _posix_uv_command(version))

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
    *claude.ai*:claude-download|*chatgpt.com*:codex-download|*astral.sh*:uv-download)
        exit 41
        ;;
esac
case "$url" in
    *claude.ai*) source="$FAKE_FIXTURES/claude-installer.sh" ;;
    *chatgpt.com*) source="$FAKE_FIXTURES/codex-installer.sh" ;;
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

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(home),
            "CALL_LOG": str(log),
            "FAKE_FIXTURES": str(fixtures),
            "FAKE_TOOL_BIN": str(tool_bin),
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


def test_install_sh_preserves_valid_existing_tools(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_uv("0.11.7")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("download:") for call in posix_harness.calls())
    assert "leaving it unchanged" in result.stdout


def test_install_sh_replaces_obsolete_uv(posix_harness: PosixHarness) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_uv("0.5.9")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "uv 0.5.9 is below 0.11.0" in result.stdout
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
        "codex-verify": "astral.sh",
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


def _powershells() -> tuple[str, ...]:
    candidates = (shutil.which("pwsh"), shutil.which("powershell"))
    return tuple(dict.fromkeys(path for path in candidates if path is not None))


def _batch_client(name: str) -> str:
    return f"""@echo off
echo {name}:%*>>"%CALL_LOG%"
if "%FAIL_STEP%"=="{name}-verify" exit /b 51
if "%1"=="--version" echo {name} 1.0.0
exit /b 0
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
if "%FAIL_STEP%"=="uv-verify" exit /b 52
echo uv {version}
exit /b 0
:install
if "%FAIL_STEP%"=="fcc-install" exit /b 53
if not exist "%FAKE_TOOL_BIN%" mkdir "%FAKE_TOOL_BIN%"
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-server.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%FAKE_TOOL_BIN%\fcc-claude.cmd" >nul
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
    log = tmp_path / "calls.log"
    for path in (bin_dir, fixtures, tool_bin, home, local_app_data):
        path.mkdir(parents=True)

    (fixtures / "claude-command.cmd").write_text(
        _batch_client("claude"), encoding="utf-8"
    )
    (fixtures / "codex-command.cmd").write_text(
        _batch_client("codex"), encoding="utf-8"
    )
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
    elseif ($Uri.Contains("astral.sh")) {
        $source = Join-Path $env:FAKE_FIXTURES "uv-installer.ps1"
    }
    else {
        throw "unexpected installer URL: $Uri"
    }
    Copy-Item -LiteralPath $source -Destination $OutFile -Force
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
            "CALL_LOG": str(log),
            "FAKE_FIXTURES": str(fixtures),
            "FAKE_TOOL_BIN": str(tool_bin),
            "FCC_INSTALLER": str(_repo_root() / "scripts" / "install.ps1"),
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


def test_install_ps1_preserves_valid_existing_tools(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_uv("0.11.7")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("download:") for call in powershell_harness.calls())
    assert "leaving it unchanged" in result.stdout


def test_install_ps1_replaces_obsolete_uv(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_uv("0.5.9")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "uv 0.5.9 is below 0.11.0" in result.stdout
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
        "codex-verify": "astral.sh",
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


def test_installers_use_native_clients_and_single_python_selection() -> None:
    shell = (_repo_root() / "scripts" / "install.sh").read_text(encoding="utf-8")
    powershell = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")

    for text in (shell, powershell):
        assert "@anthropic-ai/claude-code" not in text
        assert "@openai/codex" not in text
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


def test_readme_install_section_has_no_manual_git_prerequisite() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    install_section = readme.split("### 1. Install Or Update", 1)[1].split(
        "### 2. Start The Server", 1
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
