"""Managed Claude Code subprocess session."""

import asyncio
import os
from collections.abc import AsyncGenerator

from loguru import logger

from cli.process_registry import kill_pid_tree_best_effort, register_pid, unregister_pid
from core.trace import trace_event

from .claude import (
    ManagedClaudeConfig,
    ManagedClaudeParseState,
    ManagedClaudeTaskRequest,
    build_managed_claude_invocation,
    parse_managed_claude_stdout_line,
)

# Cap stderr capture so a runaway child cannot exhaust memory; pipe is still drained.
_MAX_STDERR_CAPTURE_BYTES = 256 * 1024


class ManagedClaudeSession:
    """Manages a single persistent Claude Code subprocess."""

    def __init__(
        self,
        workspace_path: str,
        api_url: str,
        allowed_dirs: list[str] | None = None,
        plans_directory: str | None = None,
        claude_bin: str = "claude",
        auth_token: str = "",
        *,
        log_raw_cli_diagnostics: bool = False,
    ):
        self.config = ManagedClaudeConfig(
            workspace_path=os.path.normpath(os.path.abspath(workspace_path)),
            api_url=api_url,
            allowed_dirs=[os.path.normpath(d) for d in (allowed_dirs or [])],
            plans_directory=plans_directory,
            claude_bin=claude_bin,
            auth_token=auth_token,
        )
        self.workspace = self.config.workspace_path
        self.api_url = self.config.api_url
        self.allowed_dirs = self.config.allowed_dirs
        self.plans_directory = self.config.plans_directory
        self.claude_bin = self.config.claude_bin
        self.auth_token = self.config.auth_token
        self._log_raw_cli_diagnostics = log_raw_cli_diagnostics
        self.process: asyncio.subprocess.Process | None = None
        self.current_session_id: str | None = None
        self._is_busy = False
        self._cli_lock = asyncio.Lock()

    @staticmethod
    async def _drain_stderr_bounded(
        process: asyncio.subprocess.Process,
        *,
        max_bytes: int = _MAX_STDERR_CAPTURE_BYTES,
    ) -> bytes:
        """Read stderr concurrently with stdout to avoid subprocess pipe deadlocks.

        Retains at most ``max_bytes`` for logging; any excess is discarded, but
        the pipe is read until EOF so a noisy child cannot fill the buffer and
        block forever.
        """
        if not process.stderr:
            return b""
        parts: list[bytes] = []
        received = 0
        while True:
            chunk = await process.stderr.read(65_536)
            if not chunk:
                break
            if received < max_bytes:
                take = min(len(chunk), max_bytes - received)
                if take:
                    parts.append(chunk[:take])
                    received += take
            # If already at cap, keep reading and discarding until EOF.
        return b"".join(parts)

    @property
    def is_busy(self) -> bool:
        """Check if a task is currently running."""
        return self._is_busy

    async def start_task(
        self, prompt: str, session_id: str | None = None, fork_session: bool = False
    ) -> AsyncGenerator[dict]:
        """
        Start a new task or continue an existing session.

        Args:
            prompt: The user's message/prompt
            session_id: Optional session ID to resume

        Yields:
            Event dictionaries from the CLI
        """
        async with self._cli_lock:
            self._is_busy = True
            invocation = build_managed_claude_invocation(
                config=self.config,
                request=ManagedClaudeTaskRequest(
                    prompt=prompt,
                    session_id=session_id,
                    fork_session=fork_session,
                ),
                base_env=os.environ,
            )

            trace_event(
                stage="claude_cli",
                event="claude_cli.process.launch",
                source="claude_cli",
                **invocation.trace_metadata,
            )

            try:
                self.process = await asyncio.create_subprocess_exec(
                    *invocation.argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=invocation.cwd,
                    env=invocation.env,
                )
                if self.process and self.process.pid:
                    register_pid(self.process.pid)

                if not self.process or not self.process.stdout:
                    yield {"type": "exit", "code": 1}
                    return

                parse_state = ManagedClaudeParseState(
                    log_raw_cli_diagnostics=self._log_raw_cli_diagnostics
                )
                buffer = bytearray()
                stderr_task: asyncio.Task[bytes] | None = None
                if self.process.stderr:
                    stderr_task = asyncio.create_task(
                        self._drain_stderr_bounded(self.process)
                    )

                try:
                    while True:
                        chunk = await self.process.stdout.read(65536)
                        if not chunk:
                            if buffer:
                                line_str = buffer.decode(
                                    "utf-8", errors="replace"
                                ).strip()
                                if line_str:
                                    async for event in self._handle_line_gen(
                                        line_str, parse_state
                                    ):
                                        yield event
                            break

                        buffer.extend(chunk)

                        while True:
                            newline_pos = buffer.find(b"\n")
                            if newline_pos == -1:
                                break

                            line = buffer[:newline_pos]
                            buffer = buffer[newline_pos + 1 :]

                            line_str = line.decode("utf-8", errors="replace").strip()
                            if line_str:
                                async for event in self._handle_line_gen(
                                    line_str, parse_state
                                ):
                                    yield event
                except asyncio.CancelledError:
                    # Cancelling the handler task should not leave a Claude CLI
                    # subprocess running in the background.
                    await asyncio.shield(self.stop())
                    raise
                finally:
                    stderr_bytes = b""
                    if stderr_task is not None:
                        stderr_bytes = await stderr_task

                stderr_text = None
                if stderr_bytes:
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                    if stderr_text:
                        if self._log_raw_cli_diagnostics:
                            logger.error("Claude CLI stderr: {}", stderr_text)
                        else:
                            logger.error(
                                "Claude CLI stderr: bytes={} text_chars={}",
                                len(stderr_bytes),
                                len(stderr_text),
                            )
                        logger.info("CLI_SESSION: Yielding error event from stderr")
                        yield {"type": "error", "error": {"message": stderr_text}}

                return_code = await self.process.wait()
                logger.info(
                    f"Claude CLI exited with code {return_code}, stderr_present={bool(stderr_text)}"
                )
                if return_code != 0 and not stderr_text:
                    logger.warning(
                        f"CLI_SESSION: Process exited with code {return_code} but no stderr captured"
                    )
                yield {
                    "type": "exit",
                    "code": return_code,
                    "stderr": stderr_text,
                }
            finally:
                self._is_busy = False
                if self.process and self.process.pid:
                    unregister_pid(self.process.pid)

    async def _handle_line_gen(
        self, line_str: str, parse_state: ManagedClaudeParseState
    ) -> AsyncGenerator[dict]:
        """Process a single line and yield events."""
        for event in parse_managed_claude_stdout_line(line_str, parse_state):
            if isinstance(event, dict) and event.get("type") == "session_info":
                session_id = event.get("session_id")
                if isinstance(session_id, str):
                    self.current_session_id = session_id
            yield event

    async def stop(self):
        """Stop the CLI process."""
        if self.process and self.process.returncode is None:
            try:
                logger.info(f"Stopping Claude CLI process {self.process.pid}")
                kill_pid_tree_best_effort(self.process.pid)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                if self.process and self.process.pid:
                    unregister_pid(self.process.pid)
                return True
            except Exception as e:
                if self._log_raw_cli_diagnostics:
                    logger.error(
                        "Error stopping process: {}: {}",
                        type(e).__name__,
                        e,
                    )
                else:
                    logger.error(
                        "Error stopping process: exc_type={}",
                        type(e).__name__,
                    )
                return False
        return False
