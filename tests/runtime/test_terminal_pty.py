import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from free_claude_code.application.terminal import (
    TerminalOutputEvent,
    TerminalService,
    TerminalStateEvent,
    TerminalStatus,
)
from free_claude_code.runtime.terminal_pty import create_terminal_process_factory


async def _read_until(
    events,
    output: bytearray,
    marker: bytes,
    *,
    timeout: float = 10,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while marker not in output:
        remaining = deadline - asyncio.get_running_loop().time()
        try:
            event = await asyncio.wait_for(anext(events), timeout=remaining)
        except TimeoutError as exc:
            raise AssertionError(
                f"PTY output did not contain {marker!r}: {bytes(output[-4000:])!r}"
            ) from exc
        if isinstance(event, TerminalOutputEvent):
            output.extend(event.data)


def _shell_command(*arguments: str) -> bytes:
    if os.name == "nt":
        quoted = " ".join(f"'{value.replace("'", "''")}'" for value in arguments)
        return f"& {quoted}\r".encode()
    return f"{shlex.join(arguments)}\r".encode()


@pytest.mark.asyncio
async def test_platform_pty_runs_input_and_stops_process_tree(tmp_path: Path) -> None:
    child = tmp_path / "interactive_child.py"
    child.write_text(
        "import sys\n"
        "import time\n"
        "print(f'TTY:{sys.stdin.isatty()}:{sys.stdout.isatty()}', flush=True)\n"
        "print('\\x1b[31mANSI-UNICODE-✓\\x1b[0m', flush=True)\n"
        "line = sys.stdin.readline().rstrip('\\r\\n')\n"
        "print(f'ECHO:{line}', flush=True)\n"
        "try:\n"
        "    while True:\n"
        "        time.sleep(0.1)\n"
        "except KeyboardInterrupt:\n"
        "    print('INTERRUPTED', flush=True)\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    service = TerminalService(
        create_terminal_process_factory(),
        home=tmp_path,
        env=environment,
    )
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id)
    events = attachment.__aiter__()

    try:
        await service.resize(session.id, rows=31, columns=101)
        output = bytearray(attachment.initial.output)
        await service.write(session.id, _shell_command(sys.executable, str(child)))
        await _read_until(events, output, b"TTY:True:True")
        await _read_until(events, output, "ANSI-UNICODE-✓".encode())

        await service.write(session.id, "héllo\r".encode())
        await _read_until(events, output, "ECHO:héllo".encode())

        await service.write(session.id, b"\x03")
        await _read_until(events, output, b"INTERRUPTED")
        assert (await service.get_session(session.id)).status is TerminalStatus.RUNNING

        stopped = await asyncio.wait_for(service.stop_session(session.id), timeout=10)
        assert stopped.status is TerminalStatus.EXITED
        assert stopped.error is None
    finally:
        await attachment.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_platform_stop_terminates_nested_child(tmp_path: Path) -> None:
    marker = tmp_path / "orphaned-child.txt"
    nested = tmp_path / "nested_child.py"
    nested.write_text(
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        "time.sleep(1.5)\n"
        "pathlib.Path(sys.argv[1]).write_text('orphaned', encoding='utf-8')\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent_child.py"
    parent.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "print('NESTED_READY', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    service = TerminalService(create_terminal_process_factory(), home=tmp_path)
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id)
    events = attachment.__aiter__()

    try:
        await service.write(
            session.id,
            _shell_command(
                sys.executable,
                str(parent),
                str(nested),
                str(marker),
            ),
        )
        output = bytearray(attachment.initial.output)
        await _read_until(events, output, b"NESTED_READY")
        stopped = await asyncio.wait_for(service.stop_session(session.id), timeout=10)
        assert stopped.status is TerminalStatus.EXITED
        assert stopped.error is None
        await asyncio.sleep(2)
        assert not marker.exists()
    finally:
        await attachment.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_platform_shell_exit_settles_session(tmp_path: Path) -> None:
    service = TerminalService(create_terminal_process_factory(), home=tmp_path)
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id)
    events = attachment.__aiter__()

    try:
        await service.write(session.id, b"exit\r")
        while True:
            event = await asyncio.wait_for(anext(events), timeout=10)
            if (
                isinstance(event, TerminalStateEvent)
                and event.session.status is TerminalStatus.EXITED
            ):
                assert event.session.error is None
                break
    finally:
        await attachment.aclose()
        await service.close()
