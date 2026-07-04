param(
    [switch] $VoiceNim,
    [switch] $VoiceLocal,
    [switch] $VoiceAll,
    [string] $TorchBackend = "",
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoGitUrl = "git+https://github.com/Alishahryar1/free-claude-code.git"
$PythonVersion = "3.14.0"
$MinUvVersion = "0.11.0"
$UvInstallUrl = "https://astral.sh/uv/install.ps1"

function Show-Usage {
    @"
Usage: install.ps1 [options]

Installs Claude Code and Codex if missing, installs or updates uv, Python 3.14.0, and Free Claude Code.

Options:
  -VoiceNim              Install NVIDIA NIM voice transcription support.
  -VoiceLocal            Install local Whisper voice transcription support.
  -VoiceAll              Install all voice transcription backends.
  -TorchBackend VALUE    Use a uv PyTorch backend, such as cu130. Requires local voice.
  -DryRun                Print commands without running them.
  -Help                  Show this help text.
"@
}

function Write-Step {
    param([string] $Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Format-Argument {
    param([string] $Value)

    if ($Value -match '^[A-Za-z0-9_./:@%+=,\[\]-]+$') {
        return $Value
    }

    return "'" + ($Value -replace "'", "''") + "'"
}

function Invoke-InstallCommand {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $parts = @($FilePath) + $Arguments
    $commandText = ($parts | ForEach-Object { Format-Argument ([string] $_) }) -join " "
    Write-Host "+ $commandText"

    if (-not $DryRun) {
        & $FilePath @Arguments
    }
}

function Invoke-UvInstaller {
    Write-Host "+ irm $UvInstallUrl | iex"

    if (-not $DryRun) {
        Invoke-RestMethod $UvInstallUrl | Invoke-Expression
    }
}

function Add-PathEntry {
    param([string] $PathEntry)

    if ([string]::IsNullOrWhiteSpace($PathEntry)) {
        return
    }

    $separator = [IO.Path]::PathSeparator
    $entries = @()
    if (-not [string]::IsNullOrEmpty($env:Path)) {
        $entries = $env:Path -split [regex]::Escape([string] $separator)
    }

    if ($entries -notcontains $PathEntry) {
        $env:Path = "$PathEntry$separator$env:Path"
    }
}

function Add-UvToPath {
    Add-PathEntry (Join-Path $HOME ".local\bin")
    Add-PathEntry (Join-Path $HOME ".cargo\bin")
}

function Assert-CommandAvailable {
    param([string] $Name)

    if ((-not $DryRun) -and (-not (Get-Command $Name -ErrorAction SilentlyContinue))) {
        throw "$Name is required. Install it first, then rerun this installer."
    }
}

function Invoke-ProbeCommand {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        $output = & $FilePath @Arguments 2>$null
        return [pscustomobject] @{
            ExitCode = $LASTEXITCODE
            Output = ($output | Out-String)
        }
    }
    catch {
        return [pscustomobject] @{
            ExitCode = 1
            Output = ""
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Convert-UvVersionOutput {
    param([string] $Output)

    if ([string]::IsNullOrWhiteSpace($Output)) {
        return ""
    }

    if ($Output -match '(?m)(?:^|\s)(?:uv\s+)?(?<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?)\b') {
        return $Matches["version"]
    }

    return ""
}

function Get-InstalledUvVersion {
    $version = ""

    $selfVersionProbe = Invoke-ProbeCommand -FilePath "uv" -Arguments @("self", "version", "--short")
    if ($selfVersionProbe.ExitCode -eq 0) {
        $version = Convert-UvVersionOutput $selfVersionProbe.Output
    }

    if ([string]::IsNullOrWhiteSpace($version)) {
        $versionProbe = Invoke-ProbeCommand -FilePath "uv" -Arguments @("--version")
        if ($versionProbe.ExitCode -eq 0) {
            $version = Convert-UvVersionOutput $versionProbe.Output
        }
    }

    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "Unable to determine uv version."
    }

    return $version
}

function Test-UvVersionAtLeast {
    param(
        [string] $Version,
        [string] $Minimum
    )

    $normalizedVersion = Convert-UvVersionOutput $Version
    $normalizedMinimum = Convert-UvVersionOutput $Minimum
    if ([string]::IsNullOrWhiteSpace($normalizedVersion) -or [string]::IsNullOrWhiteSpace($normalizedMinimum)) {
        throw "Unable to compare uv versions."
    }

    $normalizedVersion = $normalizedVersion -replace '[-+].*$', ''
    $normalizedMinimum = $normalizedMinimum -replace '[-+].*$', ''
    return ([version] $normalizedVersion) -ge ([version] $normalizedMinimum)
}

function Test-UvVersionSatisfiesMinimum {
    $version = Get-InstalledUvVersion
    return Test-UvVersionAtLeast -Version $version -Minimum $MinUvVersion
}

function Assert-MinUvVersion {
    if ($DryRun) {
        return
    }

    $version = Get-InstalledUvVersion
    if (-not (Test-UvVersionAtLeast -Version $version -Minimum $MinUvVersion)) {
        throw "uv $MinUvVersion or newer is required; found uv $version. Upgrade uv with its installer or package manager, then rerun this installer."
    }
}

function Test-UvSelfUpdateSupported {
    $probe = Invoke-ProbeCommand -FilePath "uv" -Arguments @("self", "update", "--dry-run")
    return $probe.ExitCode -eq 0
}

function Test-UvInstalledByScoop {
    if (-not (Get-Command scoop -ErrorAction SilentlyContinue)) {
        return $false
    }

    $probe = Invoke-ProbeCommand -FilePath "scoop" -Arguments @("list", "uv")
    return ($probe.ExitCode -eq 0) -and ($probe.Output -match '(^|\s)uv(\s|$)')
}

function Test-UvInstalledByWinget {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        return $false
    }

    $probe = Invoke-ProbeCommand -FilePath "winget" -Arguments @("list", "--id", "astral-sh.uv", "-e")
    return ($probe.ExitCode -eq 0) -and ($probe.Output -match 'astral-sh\.uv')
}

function Test-UvInstalledByPipx {
    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        return $false
    }

    $probe = Invoke-ProbeCommand -FilePath "pipx" -Arguments @("list")
    return ($probe.ExitCode -eq 0) -and ($probe.Output -match '(?m)\bpackage uv\b')
}

function Test-UvInstalledInActiveVirtualenv {
    if ([string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
        return $false
    }

    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        return $false
    }

    $uvPath = [IO.Path]::GetFullPath($uvCommand.Source)
    $venvPath = ([IO.Path]::GetFullPath($env:VIRTUAL_ENV)).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $nativePrefix = "$venvPath$([IO.Path]::DirectorySeparatorChar)"
    $alternatePrefix = "$venvPath$([IO.Path]::AltDirectorySeparatorChar)"

    return $uvPath.StartsWith($nativePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        $uvPath.StartsWith($alternatePrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Update-ExistingUv {
    if (Test-UvSelfUpdateSupported) {
        Invoke-InstallCommand -FilePath "uv" -Arguments @("self", "update")
        return
    }

    if (Test-UvInstalledByScoop) {
        Invoke-InstallCommand -FilePath "scoop" -Arguments @("update", "uv")
        return
    }

    if (Test-UvInstalledByWinget) {
        Invoke-InstallCommand -FilePath "winget" -Arguments @(
            "upgrade",
            "--id",
            "astral-sh.uv",
            "-e",
            "--accept-package-agreements",
            "--accept-source-agreements"
        )
        return
    }

    if (Test-UvInstalledByPipx) {
        Invoke-InstallCommand -FilePath "pipx" -Arguments @("upgrade", "uv")
        return
    }

    if (Test-UvInstalledInActiveVirtualenv) {
        Invoke-InstallCommand -FilePath "python" -Arguments @("-m", "pip", "install", "--upgrade", "uv")
        return
    }

    if (Test-UvVersionSatisfiesMinimum) {
        Write-Host "uv is already installed and satisfies >=$MinUvVersion; skipping automatic uv update because the install source was not detected."
        return
    }

    $version = "unknown"
    try {
        $version = Get-InstalledUvVersion
    }
    catch {
        $version = "unknown"
    }
    throw "uv $MinUvVersion or newer is required; found uv $version. The existing uv install source was not detected. Upgrade uv manually with the package manager that installed it, then rerun this installer."
}

function Install-ClaudeIfMissing {
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        Write-Host "Claude Code already found on PATH; skipping install."
        return
    }

    Assert-CommandAvailable "npm"
    Invoke-InstallCommand -FilePath "npm" -Arguments @("install", "-g", "@anthropic-ai/claude-code")
}

function Install-CodexIfMissing {
    if (Get-Command codex -ErrorAction SilentlyContinue) {
        Write-Host "Codex already found on PATH; skipping install."
        return
    }

    Assert-CommandAvailable "npm"
    Invoke-InstallCommand -FilePath "npm" -Arguments @("install", "-g", "@openai/codex")
}

function Install-OrUpdateUv {
    Add-UvToPath

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Update-ExistingUv
        Assert-MinUvVersion
        return
    }

    Invoke-UvInstaller
    Add-UvToPath

    if ((-not $DryRun) -and (-not (Get-Command uv -ErrorAction SilentlyContinue))) {
        throw "uv was installed, but it is not available on PATH. Open a new terminal or add uv's bin directory to PATH."
    }

    Assert-MinUvVersion
}

function Get-PackageSpec {
    $includeNim = $VoiceNim
    $includeLocal = $VoiceLocal

    if ($VoiceAll) {
        $includeNim = $true
        $includeLocal = $true
    }

    if ((-not [string]::IsNullOrWhiteSpace($TorchBackend)) -and (-not $includeLocal)) {
        throw "-TorchBackend requires -VoiceLocal or -VoiceAll."
    }

    if ($includeNim -and $includeLocal) {
        return "free-claude-code[voice,voice_local] @ $RepoGitUrl"
    }

    if ($includeNim) {
        return "free-claude-code[voice] @ $RepoGitUrl"
    }

    if ($includeLocal) {
        return "free-claude-code[voice_local] @ $RepoGitUrl"
    }

    return $RepoGitUrl
}

function Install-FreeClaudeCode {
    $packageSpec = Get-PackageSpec
    $toolArgs = @("tool", "install", "--force")

    if (-not [string]::IsNullOrWhiteSpace($TorchBackend)) {
        $toolArgs += @("--torch-backend", $TorchBackend)
    }

    $toolArgs += $packageSpec
    Invoke-InstallCommand -FilePath "uv" -Arguments $toolArgs
}

if ($Help) {
    Show-Usage
    return
}

if ($RemainingArgs.Count -gt 0) {
    Show-Usage
    throw "Unknown option: $($RemainingArgs -join ' ')"
}

if ((-not [string]::IsNullOrWhiteSpace($TorchBackend)) -and (-not ($VoiceLocal -or $VoiceAll))) {
    throw "-TorchBackend requires -VoiceLocal or -VoiceAll."
}

Write-Step "Installing Claude Code if missing"
Install-ClaudeIfMissing

Write-Step "Installing Codex if missing"
Install-CodexIfMissing

Write-Step "Installing uv if missing, updating if present"
Install-OrUpdateUv

Write-Step "Installing Python $PythonVersion"
Invoke-InstallCommand -FilePath "uv" -Arguments @("python", "install", $PythonVersion)

Write-Step "Installing or updating Free Claude Code"
Install-FreeClaudeCode

Write-Host ""
Write-Host "Free Claude Code is installed. Start the proxy with: fcc-server"
Write-Host "Run Claude Code with: fcc-claude"
Write-Host "Run Codex with: fcc-codex"
