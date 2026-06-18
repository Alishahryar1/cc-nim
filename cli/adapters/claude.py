"""Claude Code client CLI adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from loguru import logger

from .base import CliInvocation, CliParseState, CliTaskRequest

_AUTO_COMPACT_WINDOW = "190000"
_NO_AUTH_SENTINEL = "fcc-no-auth"


class ClaudeCliAdapter:
    """Client CLI adapter for Claude Code."""

    id = "claude"
    display_name = "Claude Code"
    default_binary = "claude"
    install_hint = "Install Claude Code with: npm install -g @anthropic-ai/claude-code"
    trace_stage = "claude_cli"
    process_launch_event = "claude_cli.process.launch"
    trace_source = "claude_cli"

    def build_task_invocation(
        self,
        *,
        config: Any,
        request: CliTaskRequest,
        base_env: Mapping[str, str],
    ) -> CliInvocation:
        """Build a Claude Code stream-json subprocess invocation."""

        env = self._task_env(
            api_url=config.api_url,
            auth_token=config.auth_token,
            base_env=base_env,
        )
        cmd = self._task_command(
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
        return CliInvocation(
            argv=tuple(cmd),
            env=env,
            cwd=config.workspace_path,
            trace_metadata={
                "client_cli_id": self.id,
                "resume_session_id": resume_session_id,
                "fork_session": request.fork_session,
                "prompt": request.prompt,
                "cwd": config.workspace_path,
                "claude_binary": config.claude_bin,
                "cli_argv": cmd,
            },
        )

    def parse_stdout_line(self, line: str, state: CliParseState) -> Iterable[Any]:
        """Parse one Claude Code JSONL line into existing parser-ready events."""

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
            extracted_id = self.extract_session_id(event)
            if extracted_id:
                state.session_id_extracted = True
                logger.info(f"Extracted session ID: {extracted_id}")
                yield {"type": "session_info", "session_id": extracted_id}

        yield event

    def extract_session_id(self, event: Any) -> str | None:
        """Extract a Claude Code session ID from supported event shapes."""

        if not isinstance(event, dict):
            return None

        if session_id := _string_value(event.get("session_id")):
            return session_id
        if session_id := _string_value(event.get("sessionId")):
            return session_id

        for key in ["init", "system", "result", "metadata"]:
            nested = event.get(key)
            if not isinstance(nested, dict):
                continue
            if session_id := _string_value(nested.get("session_id")):
                return session_id
            if session_id := _string_value(nested.get("sessionId")):
                return session_id

        conv = event.get("conversation")
        if isinstance(conv, dict):
            return _string_value(conv.get("id"))

        return None

    def get_launcher_binary_name(self, settings: Any) -> str:
        """Return the configured Claude Code binary name."""

        configured = getattr(settings, "claude_cli_bin", "")
        return configured or self.default_binary

    def build_launcher_command(
        self,
        *,
        binary_path: str,
        argv: Iterable[str],
        settings: Any,
        proxy_root_url: str,
    ) -> list[str]:
        """Return the Claude wrapper command without changing user arguments."""

        return [binary_path, *argv]

    def build_launcher_env(
        self,
        *,
        proxy_root_url: str,
        auth_token: str,
        base_env: Mapping[str, str],
    ) -> dict[str, str]:
        """Return a Claude Code environment that targets the local proxy."""

        env = {
            key: value
            for key, value in base_env.items()
            if not key.startswith("ANTHROPIC_")
        }
        env["ANTHROPIC_BASE_URL"] = proxy_root_url
        env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = _AUTO_COMPACT_WINDOW
        env["ANTHROPIC_AUTH_TOKEN"] = _claude_auth_token(auth_token)
        return env

    def _task_env(
        self,
        *,
        api_url: str,
        auth_token: str,
        base_env: Mapping[str, str],
    ) -> dict[str, str]:
        env = dict(base_env)
        env["ANTHROPIC_API_URL"] = api_url
        if api_url.endswith("/v1"):
            env["ANTHROPIC_BASE_URL"] = api_url[:-3]
        else:
            env["ANTHROPIC_BASE_URL"] = api_url
        env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = _AUTO_COMPACT_WINDOW
        env.pop("ANTHROPIC_API_KEY", None)
        env["ANTHROPIC_AUTH_TOKEN"] = _claude_auth_token(auth_token)

        env["TERM"] = "dumb"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _task_command(
        self,
        *,
        claude_bin: str,
        prompt: str,
        session_id: str | None,
        fork_session: bool,
        allowed_dirs: list[str],
        plans_directory: str | None,
    ) -> list[str]:
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
            settings_json = json.dumps({"plansDirectory": plans_directory})
            cmd.extend(["--settings", settings_json])

        return cmd


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _claude_auth_token(auth_token: str) -> str:
    return auth_token.strip() or _NO_AUTH_SENTINEL


CLAUDE_CLI_ADAPTER = ClaudeCliAdapter()
