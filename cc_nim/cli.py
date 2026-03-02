#!/usr/bin/env python3
"""cc-nim CLI: Configuration and server management."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import uvicorn
import typer
from dotenv import load_dotenv

from config.settings import get_settings

# Embedded .env template (guaranteed available)
ENV_TEMPLATE = """# NVIDIA NIM Config
NVIDIA_NIM_API_KEY=""

# OpenRouter Config
OPENROUTER_API_KEY=""

# LM Studio Config (local provider, no API key required)
LM_STUDIO_BASE_URL="http://localhost:1234/v1"

# All Claude model requests are mapped to this model
# Format: provider_type/model/name
# Valid providers: "nvidia_nim" | "open_router" | "lmstudio"
MODEL="nvidia_nim/stepfun-ai/step-3.5-flash"

# Provider Config
PROVIDER_RATE_LIMIT=40
PROVIDER_RATE_WINDOW=60
PROVIDER_MAX_CONCURRENCY=5

# HTTP client timeouts (seconds) for provider API requests
HTTP_READ_TIMEOUT=300
HTTP_WRITE_TIMEOUT=10
HTTP_CONNECT_TIMEOUT=2
HTTP_TRUST_ENV=false

# Messaging Platform: "telegram" | "discord"
MESSAGING_PLATFORM="discord"
MESSAGING_RATE_LIMIT=1
MESSAGING_RATE_WINDOW=1

# Voice Note Transcription
VOICE_NOTE_ENABLED=false
WHISPER_MODEL="base"
WHISPER_DEVICE="cpu"
HF_TOKEN=""

# Telegram Config
TELEGRAM_BOT_TOKEN=""
ALLOWED_TELEGRAM_USER_ID=""

# Discord Config
DISCORD_BOT_TOKEN=""
ALLOWED_DISCORD_CHANNELS=""

# Agent Config
CLAUDE_WORKSPACE="./agent_workspace"
ALLOWED_DIR=""
FAST_PREFIX_DETECTION=true
ENABLE_NETWORK_PROBE_MOCK=true
ENABLE_TITLE_GENERATION_SKIP=true
ENABLE_SUGGESTION_MODE_SKIP=true
ENABLE_FILEPATH_EXTRACTION_MOCK=true
"""


def _get_config_path() -> Path:
    """Resolve config file with priority: CCPROXY_CONFIG > ~/.ccenv > .env."""
    override = os.environ.get("CCPROXY_CONFIG")
    if override:
        return Path(override).expanduser()
    home_config = Path.home() / ".ccenv"
    if home_config.exists():
        return home_config
    fallback = Path.cwd() / ".env"
    if fallback.exists():
        return fallback
    return home_config  # default location (may not exist yet)


def _get_pid_path() -> Path:
    """Return PID file path: ~/.cc-nim/cc-nim.pid."""
    pid_dir = Path.home() / ".cc-nim"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir / "cc-nim.pid"


def init_command() -> int:
    """Initialize config file at the resolved path."""
    config_path = _get_config_path()
    if config_path.exists():
        print(
            f"Config already exists at {config_path}, refusing to overwrite.",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(ENV_TEMPLATE)
    print(f"Created config at {config_path}")
    print(f"Edit {config_path} with your API keys and settings.")
    return 0


def start_command() -> int:
    """Start the cc-nim server."""
    config_path = _get_config_path()
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}.", file=sys.stderr)
        print("Run 'cc-nim init' to create it first.", file=sys.stderr)
        raise typer.Exit(1)
    load_dotenv(dotenv_path=config_path, override=True)
    get_settings.cache_clear()
    settings = get_settings()
    print(f"Starting cc-nim on {settings.host}:{settings.port}...", file=sys.stderr)
    print(f"Using config: {config_path}", file=sys.stderr)
    from cc_nim.server import app as fastapi_app

    # Write PID file
    pid_path = _get_pid_path()
    # Check for existing process
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 0)
                print("cc-nim is already running.", file=sys.stderr)
                raise typer.Exit(1)
            except ProcessLookupError:
                pid_path.unlink(missing_ok=True)
            except OSError:
                pid_path.unlink(missing_ok=True)
        except (ValueError, OSError):
            pid_path.unlink(missing_ok=True)
    pid_path.write_text(str(os.getpid()))

    try:
        uvicorn.run(
            fastapi_app,
            host=settings.host,
            port=settings.port,
            log_level="info",
            timeout_graceful_shutdown=5,
        )
    finally:
        # Clean up PID file on exit
        with contextlib.suppress(Exception):
            pid_path.unlink(missing_ok=True)
    return 0


def stop_command() -> int:
    """Stop the cc-nim server (via brew service if applicable, else PID)."""
    brew_stopped = False
    # Try to stop a brew-managed service first
    try:
        result = subprocess.run(
            ["brew", "services", "stop", "cc-proxy"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Stopped brew service cc-proxy.", file=sys.stderr)
            brew_stopped = True
    except FileNotFoundError:
        # Brew not installed; skip
        pass
    except Exception:
        # Any other error, ignore and proceed to PID
        pass

    pid_path = _get_pid_path()
    if pid_path.exists():
        if brew_stopped:
            # Brew service was stopped; just clean up PID file without killing
            with contextlib.suppress(Exception):
                pid_path.unlink(missing_ok=True)
            return 0
        try:
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to process {pid}.", file=sys.stderr)
            except ProcessLookupError:
                print(f"Process {pid} already dead.", file=sys.stderr)
        except (ValueError, OSError) as e:
            print(f"Failed to read/kill PID: {e}", file=sys.stderr)
            raise typer.Exit(1)
        finally:
            with contextlib.suppress(Exception):
                pid_path.unlink(missing_ok=True)
        return 0
    else:
        if brew_stopped:
            # Brew stopped the service; no PID file is okay
            return 0
        print("PID file not found. Is the server running?", file=sys.stderr)
        print(f"Expected PID file: {pid_path}", file=sys.stderr)
        raise typer.Exit(1)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("Usage: cc-nim <init|start|stop>", file=sys.stderr)
        return 1
    cmd = argv[0]
    if cmd == "init":
        return init_command()
    elif cmd == "start":
        return start_command()
    elif cmd == "stop":
        return stop_command()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Available commands: init, start, stop", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
