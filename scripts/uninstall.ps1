param(
    [switch] $DryRun,
    [switch] $Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]] $RemainingArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PackageName = "free-claude-code"
$FccHomeDirname = ".fcc"
$FccCommands = @(
    "fcc-server",
    "fcc-claude",
    "fcc-codex",
    "fcc-init",
    "free-claude-code"
)

function Show-Usage {
    @"
Usage: uninstall.ps1 [options]

Removes the Free Claude Code uv tool and deletes ~/.fcc/.
Does not remove uv, Claude Code, Codex, or the uv-managed Python runtime.

Options:
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

function Test-MissingUvToolError {
    param([string] $Output)

    $normalized = $Output.ToLowerInvariant()
    return (
        $normalized.Contains("not installed") -or
        $normalized.Contains("no tool") -or
        $normalized.Contains("nothing to uninstall")
    )
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

function Assert-NoFccProcessesRunning {
    $running = @()

    foreach ($commandName in $FccCommands) {
        $processes = @(Get-Process -Name $commandName -ErrorAction SilentlyContinue)
        if ($processes.Count -gt 0) {
            $running += $commandName
        }
    }

    if ($running.Count -gt 0) {
        throw "Free Claude Code is still running ($($running -join ', ')). Stop those processes, then rerun uninstall."
    }
}

function Uninstall-FreeClaudeCode {
    Add-UvToPath

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "uv not found on PATH; skipping uv tool uninstall."
        return
    }

    Write-Host "+ uv tool uninstall $PackageName"
    if (-not $DryRun) {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = (& uv tool uninstall $PackageName 2>&1 | Out-String).Trim()
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 0) {
                return
            }
            if (Test-MissingUvToolError -Output $output) {
                Write-Host "Free Claude Code uv tool not installed or already removed; skipping uv tool uninstall."
                return
            }
            if (-not [string]::IsNullOrWhiteSpace($output)) {
                [Console]::Error.WriteLine($output)
            }
            throw "uv tool uninstall $PackageName failed with exit code $exitCode; aborting before deleting ~/.fcc."
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
    }
}

function Purge-FccHome {
    $fccHome = Join-Path $HOME $FccHomeDirname
    if (-not (Test-Path -LiteralPath $fccHome)) {
        Write-Host "No FCC config directory at $fccHome; skipping purge."
        return
    }

    $commandText = @(
        "Remove-Item",
        "-LiteralPath",
        (Format-Argument $fccHome),
        "-Recurse",
        "-Force"
    ) -join " "
    Write-Host "+ $commandText"

    if (-not $DryRun) {
        Remove-Item -LiteralPath $fccHome -Recurse -Force
    }
}

if ($Help) {
    Show-Usage
    return
}

if ($RemainingArgs.Count -gt 0) {
    Show-Usage
    throw "Unknown option: $($RemainingArgs -join ' ')"
}

Write-Step "Checking for running Free Claude Code processes"
Assert-NoFccProcessesRunning

Write-Step "Removing Free Claude Code uv tool"
Uninstall-FreeClaudeCode

Write-Step "Purging FCC config and data from ~/.fcc"
Purge-FccHome

Write-Host ""
Write-Host "Free Claude Code has been removed."
Write-Host "uv, Claude Code, Codex, and the uv-managed Python runtime were left installed."
