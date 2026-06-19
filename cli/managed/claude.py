"""Managed Claude Code task command, environment, and stdout parsing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cli.claude_env import CLAUDE_CODE_AUTO_COMPACT_WINDOW, claude_auth_token


@dataclass(frozen=True, slots=True)
class ManagedClaudeTaskRequest:
    """One prompt execution request for a managed Claude Code subprocess."""

    prompt: str
    session_id: str | None = None
    fork_session: bool = False


@dataclass(frozen=True, slots=True)
class ManagedClaudeInvocation:
    """Concrete subprocess invocation assembled for a managed Claude task."""

    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManagedClaudeConfig:
    """Configuration for a managed Claude Code subprocess."""

    workspace_path: str
    api_url: str
    allowed_dirs: list[str] = field(default_factory=list)
    plans_directory: str | None = None
    claude_bin: str = "claude"
    auth_token: str = ""


@dataclass(slots=True)
class ManagedClaudeParseState:
    """Mutable stdout parser state for one managed Claude Code task run."""

    log_raw_cli_diagnostics: bool = False
    session_id_extracted: bool = False


def build_managed_claude_invocation(
    *,
    config: ManagedClaudeConfig,
    request: ManagedClaudeTaskRequest,
    base_env: Mapping[str, str],
) -> ManagedClaudeInvocation:
    """Build a Claude Code stream-json subprocess invocation."""

    cmd = build_managed_claude_command(
        claude_bin=config.claude_bin,
        prompt=request.prompt,
        session_id=request.session_id,
        fork_session=request.fork_session,
        allowed_dirs=config.allowed_dirs,
        plans_directory=config.plans_directory,
    )
    resume_session_id = (
        request.session_id
        if request.session_id and not request.session_id.startswith("pending_")
        else None
    )
    return ManagedClaudeInvocation(
        argv=tuple(cmd),
        env=build_managed_claude_env(
            api_url=config.api_url,
            auth_token=config.auth_token,
            base_env=base_env,
        ),
        cwd=config.workspace_path,
        trace_metadata={
            "client_cli_id": "claude",
            "resume_session_id": resume_session_id,
            "fork_session": request.fork_session,
            "prompt": request.prompt,
            "cwd": config.workspace_path,
            "claude_binary": config.claude_bin,
            "cli_argv": cmd,
        },
    )


def build_managed_claude_env(
    *,
    api_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a Claude Code task environment that targets the local proxy."""

    env = dict(base_env)
    env["ANTHROPIC_API_URL"] = api_url
    env["ANTHROPIC_BASE_URL"] = api_url[:-3] if api_url.endswith("/v1") else api_url
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = CLAUDE_CODE_AUTO_COMPACT_WINDOW
    env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_AUTH_TOKEN"] = claude_auth_token(auth_token)
    env["TERM"] = "dumb"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build_managed_claude_command(
    *,
    claude_bin: str,
    prompt: str,
    session_id: str | None,
    fork_session: bool,
    allowed_dirs: list[str],
    plans_directory: str | None,
) -> list[str]:
    """Return the Claude Code stream-json command for a managed task."""

    if session_id and not session_id.startswith("pending_"):
        cmd = [
            claude_bin,
            "--resume",
            session_id,
        ]
        if fork_session:
            cmd.append("--fork-session")
        cmd += [
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]
    else:
        cmd = [
            claude_bin,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]

    for directory in allowed_dirs:
        cmd.extend(["--add-dir", directory])

    if plans_directory is not None:
        cmd.extend(["--settings", json.dumps({"plansDirectory": plans_directory})])

    return cmd


def parse_managed_claude_stdout_line(
    line: str, state: ManagedClaudeParseState
) -> Iterable[Any]:
    """Parse one Claude Code stream-json stdout line."""

    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        if state.log_raw_cli_diagnostics:
            logger.debug("Non-JSON output: {}", line)
        else:
            logger.debug("Non-JSON CLI line: char_len={}", len(line))
        yield {"type": "raw", "content": line}
        return

    if not state.session_id_extracted:
        extracted_id = extract_managed_claude_session_id(event)
        if extracted_id:
            state.session_id_extracted = True
            logger.info("Extracted session ID: {}", extracted_id)
            yield {"type": "session_info", "session_id": extracted_id}

    yield event


def extract_managed_claude_session_id(event: Any) -> str | None:
    """Extract a Claude Code session ID from supported stream-json event shapes."""

    if not isinstance(event, dict):
        return None

    if session_id := _string_value(event.get("session_id")):
        return session_id
    if session_id := _string_value(event.get("sessionId")):
        return session_id

    for key in ("init", "system", "result", "metadata"):
        nested = event.get(key)
        if not isinstance(nested, dict):
            continue
        if session_id := _string_value(nested.get("session_id")):
            return session_id
        if session_id := _string_value(nested.get("sessionId")):
            return session_id

    conversation = event.get("conversation")
    if isinstance(conversation, dict):
        return _string_value(conversation.get("id"))

    return None


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None
