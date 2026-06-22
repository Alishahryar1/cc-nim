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


def test_install_sh_installs_claude_only_when_missing() -> None:
    text = _script_text("install.sh")
    body = _braced_body(text, "install_claude_if_missing()")
    main = text[text.index('parse_args "$@"') :]

    assert "Installs Claude Code and Codex if missing" in text
    assert "if command -v claude >/dev/null 2>&1; then" in body
    assert "Claude Code already found on PATH; skipping install." in body
    assert "require_command npm" in body
    assert "run npm install -g @anthropic-ai/claude-code" in body
    assert body.index("command -v claude") < body.index("run npm install")
    assert body.index("return 0") < body.index("run npm install")
    assert 'step "Installing Claude Code if missing"\ninstall_claude_if_missing' in main
    assert "npm install -g @anthropic-ai/claude-code" not in main


def test_install_sh_installs_codex_only_when_missing() -> None:
    text = _script_text("install.sh")
    body = _braced_body(text, "install_codex_if_missing()")
    main = text[text.index('parse_args "$@"') :]

    assert "if command -v codex >/dev/null 2>&1; then" in body
    assert "Codex already found on PATH; skipping install." in body
    assert "require_command npm" in body
    assert "run npm install -g @openai/codex" in body
    assert body.index("command -v codex") < body.index("run npm install")
    assert body.index("return 0") < body.index("run npm install")
    assert 'step "Installing Codex if missing"\ninstall_codex_if_missing' in main
    assert "npm install -g @openai/codex" not in main
    assert "fcc-claude" in text
    assert "fcc-codex" in text


def test_install_sh_installs_missing_uv_without_self_update() -> None:
    text = _script_text("install.sh")
    body = _braced_body(text, "install_or_update_uv()")

    assert "if command -v uv >/dev/null 2>&1; then" in body
    assert "update_existing_uv" in body
    assert "run uv self update" not in body

    update_index = body.index("update_existing_uv")
    validate_existing_index = body.index("validate_uv_version", update_index)
    installer_index = body.index("run_uv_installer")
    validate_installed_index = body.index("validate_uv_version", installer_index)
    verification_index = body.index('if [ "$dry_run" -eq 0 ] && ! command -v uv')

    assert update_index < validate_existing_index < installer_index
    assert installer_index < verification_index < validate_installed_index


def test_install_sh_updates_uv_with_detected_source() -> None:
    text = _script_text("install.sh")
    update_body = _braced_body(text, "update_existing_uv()")

    assert "uv self update --dry-run" in text
    assert update_body.count("run uv self update") == 1
    assert update_body.index("uv_self_update_supported") < update_body.index(
        "run uv self update"
    )

    assert "brew list --versions uv" in text
    assert "run brew upgrade uv" in update_body
    assert "pipx list" in text
    assert "run pipx upgrade uv" in update_body
    assert "VIRTUAL_ENV" in text
    assert "run python -m pip install --upgrade uv" in update_body
    assert "uv_version_satisfies_minimum" in update_body
    assert "install source was not detected" in update_body


def test_install_sh_validates_minimum_uv_version() -> None:
    text = _script_text("install.sh")
    validate_body = _braced_body(text, "validate_uv_version()")

    assert 'MIN_UV_VERSION="0.11.0"' in text
    assert "uv self version --short" in text
    assert "version_ge" in validate_body
    assert "uv $MIN_UV_VERSION or newer is required" in validate_body


def test_install_ps1_installs_claude_only_when_missing() -> None:
    text = _script_text("install.ps1")
    body = _braced_body(text, "function Install-ClaudeIfMissing")

    assert "Installs Claude Code and Codex if missing" in text
    assert "if (Get-Command claude -ErrorAction SilentlyContinue)" in body
    assert "Claude Code already found on PATH; skipping install." in body
    assert 'Assert-CommandAvailable "npm"' in body
    assert (
        'Invoke-InstallCommand -FilePath "npm" '
        '-Arguments @("install", "-g", "@anthropic-ai/claude-code")'
    ) in body
    assert body.index("Get-Command claude") < body.index("Invoke-InstallCommand")
    assert body.index("return") < body.index("Invoke-InstallCommand")
    assert (
        'Write-Step "Installing Claude Code if missing"\nInstall-ClaudeIfMissing'
        in text
    )


def test_install_ps1_installs_codex_only_when_missing() -> None:
    text = _script_text("install.ps1")
    body = _braced_body(text, "function Install-CodexIfMissing")

    assert "if (Get-Command codex -ErrorAction SilentlyContinue)" in body
    assert "Codex already found on PATH; skipping install." in body
    assert 'Assert-CommandAvailable "npm"' in body
    assert (
        'Invoke-InstallCommand -FilePath "npm" '
        '-Arguments @("install", "-g", "@openai/codex")'
    ) in body
    assert body.index("Get-Command codex") < body.index("Invoke-InstallCommand")
    assert body.index("return") < body.index("Invoke-InstallCommand")
    assert 'Write-Step "Installing Codex if missing"\nInstall-CodexIfMissing' in text
    assert "fcc-claude" in text
    assert "fcc-codex" in text


def test_install_ps1_installs_missing_uv_without_self_update() -> None:
    text = _script_text("install.ps1")
    body = _braced_body(text, "function Install-OrUpdateUv")
    self_update = 'Invoke-InstallCommand -FilePath "uv" -Arguments @("self", "update")'

    assert "if (Get-Command uv -ErrorAction SilentlyContinue)" in body
    assert "Update-ExistingUv" in body
    assert self_update not in body

    update_index = body.index("Update-ExistingUv")
    validate_existing_index = body.index("Assert-MinUvVersion", update_index)
    installer_index = body.index("Invoke-UvInstaller")
    verification_index = body.index("if ((-not $DryRun)")
    validate_installed_index = body.index("Assert-MinUvVersion", installer_index)

    assert update_index < validate_existing_index < installer_index
    assert installer_index < verification_index < validate_installed_index


def test_install_ps1_updates_uv_with_detected_source() -> None:
    text = _script_text("install.ps1")
    update_body = _braced_body(text, "function Update-ExistingUv")
    self_update = 'Invoke-InstallCommand -FilePath "uv" -Arguments @("self", "update")'

    assert '"self", "update", "--dry-run"' in text
    assert update_body.count(self_update) == 1
    assert update_body.index("Test-UvSelfUpdateSupported") < update_body.index(
        self_update
    )

    assert (
        'Invoke-InstallCommand -FilePath "scoop" -Arguments @("update", "uv")'
        in update_body
    )
    assert '"winget"' in update_body
    assert '"astral-sh.uv"' in update_body
    assert '"--accept-package-agreements"' in update_body
    assert (
        'Invoke-InstallCommand -FilePath "pipx" -Arguments @("upgrade", "uv")'
        in update_body
    )
    assert (
        'Invoke-InstallCommand -FilePath "python" -Arguments @("-m", "pip", "install", "--upgrade", "uv")'
        in update_body
    )
    assert "Test-UvVersionSatisfiesMinimum" in update_body
    assert "install source was not detected" in update_body


def test_install_ps1_validates_minimum_uv_version() -> None:
    text = _script_text("install.ps1")
    validate_body = _braced_body(text, "function Assert-MinUvVersion")
    get_version_body = _braced_body(text, "function Get-InstalledUvVersion")

    assert '$MinUvVersion = "0.11.0"' in text
    assert '"self", "version", "--short"' in text
    assert "Convert-UvVersionOutput $selfVersionProbe.Output" in get_version_body
    assert "Convert-UvVersionOutput $versionProbe.Output" in get_version_body
    assert ".Output.Trim()" not in get_version_body
    assert "[version]" in text
    assert "uv $MinUvVersion or newer is required" in validate_body


def test_install_ps1_parses_uv_version_probe_outputs(tmp_path: Path) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is not available")

    text = _script_text("install.ps1")
    convert_body = _braced_body(text, "function Convert-UvVersionOutput")
    get_version_body = _braced_body(text, "function Get-InstalledUvVersion")
    compare_body = _braced_body(text, "function Test-UvVersionAtLeast")
    script = f"""
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Convert-UvVersionOutput {{
{convert_body}
}}
function Get-InstalledUvVersion {{
{get_version_body}
}}
function Test-UvVersionAtLeast {{
{compare_body}
}}

$script:Mode = "self-long"
function Invoke-ProbeCommand {{
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $joined = $Arguments -join " "
    if ($script:Mode -eq "self-long" -and $joined -eq "self version --short") {{
        return [pscustomobject] @{{ ExitCode = 0; Output = "0.11.7 (9d177269e 2026-06-05 x86_64-pc-windows-msvc)" }}
    }}
    if ($script:Mode -eq "fallback-long" -and $joined -eq "self version --short") {{
        return [pscustomobject] @{{ ExitCode = 1; Output = "" }}
    }}
    if ($script:Mode -eq "fallback-long" -and $joined -eq "--version") {{
        return [pscustomobject] @{{ ExitCode = 0; Output = "uv 0.11.7 (9d177269e 2026-06-05 x86_64-pc-windows-msvc)" }}
    }}
    if ($script:Mode -eq "bad") {{
        return [pscustomobject] @{{ ExitCode = 0; Output = "not a uv version" }}
    }}
    throw "Unexpected probe: $joined"
}}

if ((Convert-UvVersionOutput "0.11.7") -ne "0.11.7") {{ throw "clean version parse failed" }}
if ((Convert-UvVersionOutput "uv 0.11.7 (9d177269e 2026)") -ne "0.11.7") {{ throw "uv --version parse failed" }}
if ((Convert-UvVersionOutput "0.11.7 (9d177269e 2026)") -ne "0.11.7") {{ throw "self version parse failed" }}
if ((Convert-UvVersionOutput "not a uv version") -ne "") {{ throw "bad output should not parse" }}

$script:Mode = "self-long"
if ((Get-InstalledUvVersion) -ne "0.11.7") {{ throw "self version normalization failed" }}
$script:Mode = "fallback-long"
if ((Get-InstalledUvVersion) -ne "0.11.7") {{ throw "fallback version normalization failed" }}
if (-not (Test-UvVersionAtLeast -Version "0.11.7 (9d177269e 2026)" -Minimum "0.11.0")) {{ throw "version comparison failed" }}

$script:Mode = "bad"
try {{
    Get-InstalledUvVersion | Out-Null
    throw "bad version did not fail"
}}
catch {{
    if ($_.Exception.Message -ne "Unable to determine uv version.") {{
        throw
    }}
}}
"""
    script_path = tmp_path / "test-install-uv-version.ps1"
    script_path.write_text(script, encoding="utf-8")

    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
