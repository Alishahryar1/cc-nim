"""Read/write contract smoke for the latest installed Codex app-server."""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from free_claude_code.application.work import CodexThreadSettings
from free_claude_code.runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerProcessPlan,
)


def _codex_version(binary: str) -> str:
    completed = subprocess.run(
        (binary, "--version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def _client(binary: str, version: str) -> CodexAppServerClient:
    async def process_plan() -> CodexAppServerProcessPlan:
        return CodexAppServerProcessPlan(
            command=(binary, "app-server"),
            env=dict(os.environ),
            binary_path=binary,
            version=version,
        )

    return CodexAppServerClient(process_plan, client_version="contract-smoke")


async def _thread_is_listed(client: CodexAppServerClient, thread_id: str) -> bool:
    cursor: str | None = None
    while True:
        page = await client.list_threads_page(cursor=cursor, limit=100)
        if any(record.get("id") == thread_id for record in page.records):
            return True
        cursor = page.next_cursor
        if cursor is None:
            return False


async def _run() -> None:
    binary = shutil.which("codex")
    if binary is None:
        raise RuntimeError(
            "Install Codex before running its app-server contract smoke."
        )
    version = await asyncio.to_thread(_codex_version, binary)
    thread_id: str | None = None
    model: str | None = None

    with tempfile.TemporaryDirectory(prefix="fcc-codex-contract-") as directory:
        cwd = str(Path(directory).resolve())
        first = _client(binary, version)
        try:
            initialization = await first.initialize()
            if not initialization.connection_id:
                raise RuntimeError("Codex initialize returned no connection identity.")
            handle = await first.start_thread(CodexThreadSettings(cwd=cwd))
            thread_id = handle.thread_id
            response_model = handle.response.get("model")
            if not isinstance(response_model, str) or not response_model:
                raise RuntimeError("Codex thread/start returned no concrete model.")
            model = response_model
            await first.materialize_thread(thread_id)
            page = await first.list_turns_page(
                thread_id=thread_id,
                cursor=None,
                limit=1,
            )
            if page.records:
                raise RuntimeError("A newly created Codex thread was not empty.")
        finally:
            await first.close()

        second = _client(binary, version)
        try:
            if thread_id is None or model is None:
                raise RuntimeError("Codex thread creation did not complete.")
            await second.resume_thread(
                thread_id,
                CodexThreadSettings(cwd=cwd, model=model),
            )
            page = await second.list_turns_page(
                thread_id=thread_id,
                cursor=None,
                limit=1,
            )
            if page.records:
                raise RuntimeError("The resumed Codex thread was not empty.")
            await second.delete_thread(thread_id)
            if await _thread_is_listed(second, thread_id):
                raise RuntimeError("Codex still listed the deleted thread.")
            thread_id = None
        finally:
            if thread_id is not None:
                await second.delete_thread(thread_id)
            await second.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
