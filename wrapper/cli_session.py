import asyncio
import os
import json
import logging
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


class CLISession:
    """Manages a single persistent Claude Code CLI subprocess."""

    def __init__(
        self,
        workspace_path: str,
        api_url: str,
        allowed_dirs: Optional[list[str]] = None,
    ):
        self.workspace = workspace_path
        self.api_url = api_url
        self.allowed_dirs = allowed_dirs or []
        self.process: Optional[asyncio.subprocess.Process] = None

    async def start_task(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Runs a single prompt and yields JSON events from the CLI."""
        env = os.environ.copy()

        # FIX for 404:
        # ANTHROPIC_API_URL usually wants the full path to the API endpoint or version root
        env["ANTHROPIC_API_URL"] = self.api_url

        # ANTHROPIC_BASE_URL usually wants the server root, and the client appends /v1/messages
        # If self.api_url is "http://localhost:8082/v1", base should be "http://localhost:8082"
        if self.api_url.endswith("/v1"):
            env["ANTHROPIC_BASE_URL"] = self.api_url[:-3]
        else:
            env["ANTHROPIC_BASE_URL"] = self.api_url

        # Ensure we don't try to use interactive TTY features
        env["TERM"] = "dumb"
        # Ensure path is normalized for Windows to avoid \a (bell) character issues
        normalized_workspace = os.path.normpath(self.workspace)

        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]

        if self.allowed_dirs:
            for d in self.allowed_dirs:
                cmd.extend(["--add-dir", os.path.normpath(d)])

        logger.info(
            f"Launching Claude CLI in {normalized_workspace} with API {self.api_url}"
        )

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=normalized_workspace,
            env=env,
        )

        # Read stdout line by line
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break

            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                event = json.loads(line_str)
                yield event
            except json.JSONDecodeError:
                # Log non-JSON lines for debugging but don't crash
                logger.debug(f"Non-JSON output: {line_str}")
                yield {"type": "raw", "content": line_str}

        # Capture remaining stderr if the process crashed
        stderr_output = await self.process.stderr.read()
        if stderr_output:
            logger.error(
                f"Claude CLI Stderr: {stderr_output.decode('utf-8', errors='replace')}"
            )

        return_code = await self.process.wait()
        logger.info(f"Claude CLI exited with code {return_code}")
        yield {"type": "exit", "code": return_code}
