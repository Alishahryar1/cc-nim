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
$ProgressPreference = "SilentlyContinue"

$RepoArchiveUrl = "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
$PythonVersion = "3.14.0"
$MinUvVersion = "0.11.0"
$ClaudeInstallUrl = "https://claude.ai/install.ps1"
$CodexInstallUrl = "https://chatgpt.com/codex/install.ps1"
$UvInstallUrl = "https://astral.sh/uv/install.ps1"

function Show-Usage {
    @"
Usage: install.ps1 [options]

Installs Claude Code and Codex if missing, ensures a compatible uv, and installs or updates Free Claude Code.

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

    if ($Value -match '^[A-Za-z0-9_./:@%+=,\[\]\\-]+$') {
        return $Value
    }

    return "'" + ($Value -replace "'", "''") + "'"
}

function Format-Command {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $parts = @($FilePath) + $Arguments
    return ($parts | ForEach-Object { Format-Argument ([string] $_) }) -join " "
}

function Invoke-NativeCommand {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    if ($DryRun) {
        return
    }

    $global:LASTEXITCODE = 0
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }
}

function Invoke-NativeCapture {
    param(
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    $commandText = Format-Command -FilePath $FilePath -Arguments $Arguments
    Write-Host "+ $commandText"
    $global:LASTEXITCODE = 0
    $output = & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $commandText"
    }

    return ($output | Out-String).Trim()
}

function Get-ApplicationCommand {
    param([string] $Name)

    $commands = @(Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
    if ($commands.Count -eq 0) {
        return $null
    }

    return $commands[0]
}

function Get-PowerShellExecutable {
    param([string] $PowerShellHome = $PSHOME)

    $executableName = if ($PSVersionTable.PSEdition -eq "Core") {
        "pwsh.exe"
    }
    else {
        "powershell.exe"
    }
    $bundledExecutable = Join-Path $PowerShellHome $executableName
    if (Test-Path -LiteralPath $bundledExecutable -PathType Leaf) {
        return $bundledExecutable
    }

    $pathCommand = Get-ApplicationCommand ([IO.Path]::GetFileNameWithoutExtension($executableName))
    if ($pathCommand) {
        return $pathCommand.Source
    }

    throw "Unable to locate a PowerShell executable for the downloaded installer."
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

function Add-KnownBinDirectories {
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        Add-PathEntry (Join-Path $env:USERPROFILE ".local\bin")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        Add-PathEntry (Join-Path $env:LOCALAPPDATA "Programs\OpenAI\Codex\bin")
    }
}

function Invoke-DownloadedPowerShellInstaller {
    param(
        [string] $Url,
        [string] $Name,
        [switch] $NonInteractive
    )

    if ($DryRun) {
        Write-Host "+ irm $Url -OutFile <temporary-script>"
        $prefix = if ($NonInteractive) { "CODEX_NON_INTERACTIVE=1 " } else { "" }
        Write-Host "+ ${prefix}powershell -NoProfile -ExecutionPolicy Bypass -File <temporary-script>"
        return
    }

    $temporaryScript = Join-Path ([IO.Path]::GetTempPath()) ("fcc-install-" + [guid]::NewGuid().ToString("N") + ".ps1")
    try {
        Write-Host "+ irm $Url -OutFile $(Format-Argument $temporaryScript)"
        Invoke-RestMethod -Uri $Url -OutFile $temporaryScript -ErrorAction Stop
        if ((-not (Test-Path -LiteralPath $temporaryScript)) -or ((Get-Item -LiteralPath $temporaryScript).Length -eq 0)) {
            throw "The downloaded $Name installer was empty."
        }

        $powerShellPath = Get-PowerShellExecutable

        $hadNonInteractive = Test-Path Env:CODEX_NON_INTERACTIVE
        $previousNonInteractive = $env:CODEX_NON_INTERACTIVE
        try {
            if ($NonInteractive) {
                $env:CODEX_NON_INTERACTIVE = "1"
            }
            Invoke-NativeCommand -FilePath $powerShellPath -Arguments @(
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                $temporaryScript
            )
        }
        finally {
            if ($hadNonInteractive) {
                $env:CODEX_NON_INTERACTIVE = $previousNonInteractive
            }
            else {
                Remove-Item Env:CODEX_NON_INTERACTIVE -ErrorAction SilentlyContinue
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryScript -Force -ErrorAction SilentlyContinue
    }
}

function Confirm-Application {
    param(
        [string] $CommandName,
        [string] $DisplayName
    )

    if ($DryRun) {
        Write-Host "+ $CommandName --version"
        return
    }

    $command = Get-ApplicationCommand $CommandName
    if (-not $command) {
        throw "$DisplayName was installed, but '$CommandName' is not available on PATH."
    }
    Invoke-NativeCommand -FilePath $command.Source -Arguments @("--version")
}

function Ensure-ClaudeCode {
    if (Get-ApplicationCommand "claude") {
        Write-Host "Claude Code already found on PATH; verifying it."
    }
    else {
        Invoke-DownloadedPowerShellInstaller -Url $ClaudeInstallUrl -Name "Claude Code"
        Add-KnownBinDirectories
    }

    Confirm-Application -CommandName "claude" -DisplayName "Claude Code"
}

function Ensure-Codex {
    if (Get-ApplicationCommand "codex") {
        Write-Host "Codex already found on PATH; verifying it."
    }
    else {
        Invoke-DownloadedPowerShellInstaller -Url $CodexInstallUrl -Name "Codex" -NonInteractive
        Add-KnownBinDirectories
    }

    Confirm-Application -CommandName "codex" -DisplayName "Codex"
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

function Get-UvVersion {
    param([string] $UvPath)

    $output = Invoke-NativeCapture -FilePath $UvPath -Arguments @("--version")
    $version = Convert-UvVersionOutput $output
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "uv is present, but 'uv --version' did not return a valid version."
    }

    return $version
}

function Test-UvVersionAtLeast {
    param(
        [string] $Version,
        [string] $Minimum
    )

    $normalizedVersion = (Convert-UvVersionOutput $Version) -replace '[-+].*$', ''
    $normalizedMinimum = (Convert-UvVersionOutput $Minimum) -replace '[-+].*$', ''
    if ([string]::IsNullOrWhiteSpace($normalizedVersion) -or [string]::IsNullOrWhiteSpace($normalizedMinimum)) {
        throw "Unable to compare uv versions."
    }

    return ([version] $normalizedVersion) -ge ([version] $normalizedMinimum)
}

function Confirm-Uv {
    if ($DryRun) {
        Write-Host "+ uv --version"
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv was installed, but it is not available on PATH."
    }

    $version = Get-UvVersion $uvCommand.Source
    if (-not (Test-UvVersionAtLeast -Version $version -Minimum $MinUvVersion)) {
        throw "uv $MinUvVersion or newer is required; found uv $version after installation."
    }
    Write-Host "Verified uv $version."
}

function Ensure-Uv {
    if ($DryRun) {
        if (Get-ApplicationCommand "uv") {
            Write-Host "+ uv --version"
            Write-Host "A compatible existing uv will be left unchanged; an obsolete one will be replaced by the standalone installer."
        }
        else {
            Write-Host "uv is not installed; the current standalone uv would be installed."
            Invoke-DownloadedPowerShellInstaller -Url $UvInstallUrl -Name "uv"
            Confirm-Uv
        }
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if ($uvCommand) {
        $version = Get-UvVersion $uvCommand.Source
        if (Test-UvVersionAtLeast -Version $version -Minimum $MinUvVersion) {
            Write-Host "uv $version already satisfies >=$MinUvVersion; leaving it unchanged."
            return
        }
        Write-Host "uv $version is below $MinUvVersion; installing the current standalone uv."
    }
    else {
        Write-Host "uv is not installed; installing the current standalone uv."
    }

    Invoke-DownloadedPowerShellInstaller -Url $UvInstallUrl -Name "uv"
    Add-KnownBinDirectories
    Confirm-Uv
}

function Get-PackageSpec {
    $includeNim = $VoiceNim
    $includeLocal = $VoiceLocal

    if ($VoiceAll) {
        $includeNim = $true
        $includeLocal = $true
    }

    if ($includeNim -and $includeLocal) {
        return "free-claude-code[voice,voice_local] @ $RepoArchiveUrl"
    }
    if ($includeNim) {
        return "free-claude-code[voice] @ $RepoArchiveUrl"
    }
    if ($includeLocal) {
        return "free-claude-code[voice_local] @ $RepoArchiveUrl"
    }
    return "free-claude-code @ $RepoArchiveUrl"
}

function Install-FreeClaudeCode {
    $packageSpec = Get-PackageSpec
    $arguments = @(
        "tool",
        "install",
        "--force",
        "--refresh-package",
        "free-claude-code",
        "--python",
        $PythonVersion
    )
    if (-not [string]::IsNullOrWhiteSpace($TorchBackend)) {
        $arguments += @("--torch-backend", $TorchBackend)
    }
    $arguments += $packageSpec

    $uvPath = "uv"
    if (-not $DryRun) {
        $uvCommand = Get-ApplicationCommand "uv"
        if (-not $uvCommand) {
            throw "uv is not available for the Free Claude Code installation."
        }
        $uvPath = $uvCommand.Source
    }
    Invoke-NativeCommand -FilePath $uvPath -Arguments $arguments
}

function Configure-AndConfirmFreeClaudeCode {
    if ($DryRun) {
        Write-Host "+ uv tool update-shell"
        Write-Host "+ uv tool dir --bin"
        Write-Host "+ verify fcc-server, fcc-claude, and fcc-codex in the uv tool bin directory"
        Write-Host "+ fcc-server --version"
        return
    }

    $uvCommand = Get-ApplicationCommand "uv"
    if (-not $uvCommand) {
        throw "uv is not available for PATH configuration."
    }
    Invoke-NativeCommand -FilePath $uvCommand.Source -Arguments @("tool", "update-shell")
    $toolBin = Invoke-NativeCapture -FilePath $uvCommand.Source -Arguments @("tool", "dir", "--bin")
    if ([string]::IsNullOrWhiteSpace($toolBin)) {
        throw "uv returned an empty tool bin directory."
    }

    Add-PathEntry $toolBin
    $toolBinPath = ([IO.Path]::GetFullPath($toolBin)).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $installedCommands = @{}
    foreach ($commandName in @("fcc-server", "fcc-claude", "fcc-codex")) {
        $command = Get-ApplicationCommand $commandName
        if (-not $command) {
            throw "Free Claude Code installation did not create '$commandName'."
        }
        $commandDirectory = ([IO.Path]::GetFullPath((Split-Path -Parent $command.Source))).TrimEnd(
            [IO.Path]::DirectorySeparatorChar,
            [IO.Path]::AltDirectorySeparatorChar
        )
        if (-not $commandDirectory.Equals($toolBinPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "'$commandName' resolved outside the uv tool bin directory: $($command.Source)"
        }
        $installedCommands[$commandName] = $command.Source
    }

    Invoke-NativeCommand -FilePath $installedCommands["fcc-server"] -Arguments @("--version")
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

Add-KnownBinDirectories

Write-Step "Ensuring Claude Code is installed"
Ensure-ClaudeCode

Write-Step "Ensuring Codex is installed"
Ensure-Codex

Write-Step "Ensuring uv $MinUvVersion or newer is installed"
Ensure-Uv

Write-Step "Installing or updating Free Claude Code"
Install-FreeClaudeCode

Write-Step "Configuring PATH and verifying Free Claude Code"
Configure-AndConfirmFreeClaudeCode

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No changes were made."
}
else {
    Write-Host "Free Claude Code is installed and verified. Start the proxy with: fcc-server"
    Write-Host "Run Claude Code with: fcc-claude"
    Write-Host "Run Codex with: fcc-codex"
}
