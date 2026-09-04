import asyncio
import os
import shlex
import sys
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from free_claude_code.application.terminal import (
    TerminalAttachmentEvent,
    TerminalOutputEvent,
    TerminalService,
    TerminalStateEvent,
    TerminalStatus,
)
from free_claude_code.runtime.terminal_zellij import ZellijTerminalEngineHost

if os.name == "nt":
    from free_claude_code.runtime.terminal_pty_windows import (
        WindowsProcessContainment as PlatformProcessContainment,
    )
    from free_claude_code.runtime.terminal_pty_windows import (
        WindowsTerminalClientFactory as PlatformTerminalClientFactory,
    )
else:
    from free_claude_code.runtime.terminal_pty_posix import (
        PosixProcessContainment as PlatformProcessContainment,
    )
    from free_claude_code.runtime.terminal_pty_posix import (
        PosixTerminalClientFactory as PlatformTerminalClientFactory,
    )


def _engine(tmp_path: Path) -> ZellijTerminalEngineHost:
    configured = os.environ.get("FCC_TEST_ZELLIJ_BIN")
    if not configured:
        pytest.skip("FCC_TEST_ZELLIJ_BIN is required for real-engine tests")
    binary = Path(configured).resolve()
    if not binary.is_file():
        pytest.fail(f"FCC_TEST_ZELLIJ_BIN does not exist: {binary}")
    root = tmp_path / "engine"
    return ZellijTerminalEngineHost(
        binary=binary,
        config=root / "config.kdl",
        sockets=root / "sockets",
        data=root / "data",
        lock_path=root / "terminal.lock",
        client_factory=PlatformTerminalClientFactory(),
        containment_factory=PlatformProcessContainment,
    )


async def _read_until(
    events: AsyncIterator[TerminalAttachmentEvent],
    output: bytearray,
    marker: bytes,
    *,
    timeout: float = 15,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while marker not in output:
        remaining = deadline - asyncio.get_running_loop().time()
        try:
            event = await asyncio.wait_for(anext(events), timeout=remaining)
        except TimeoutError as exc:
            raise AssertionError(
                f"Terminal output did not contain {marker!r}: {bytes(output[-4000:])!r}"
            ) from exc
        if isinstance(event, TerminalOutputEvent):
            output.extend(event.data)


def _shell_command(*arguments: str) -> str:
    if os.name == "nt":
        quoted = " ".join(f"'{value.replace("'", "''")}'" for value in arguments)
        return f"& {quoted}\r"
    return f"{shlex.join(arguments)}\r"


def _environment() -> Mapping[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    return environment


@pytest.mark.asyncio
async def test_platform_engine_runs_input_and_stops_process_tree(
    tmp_path: Path,
) -> None:
    child = tmp_path / "interactive_child.py"
    child.write_text(
        "import sys\n"
        "import time\n"
        "print(f'TTY:{sys.stdin.isatty()}:{sys.stdout.isatty()}', flush=True)\n"
        "print('\\x1b[31mANSI-UNICODE-✓\\x1b[0m', flush=True)\n"
        "line = sys.stdin.readline().rstrip('\\r\\n')\n"
        "print(f'ECHO:{line}', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    service = TerminalService(
        _engine(tmp_path),
        home=tmp_path,
        env=_environment(),
    )
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id, rows=31, columns=101)
    events = attachment.__aiter__()

    try:
        output = bytearray(attachment.initial.output)
        await attachment.write(_shell_command(sys.executable, str(child)))
        await _read_until(events, output, b"TTY:True:True")
        await _read_until(events, output, "ANSI-UNICODE-✓".encode())

        await attachment.write("héllo\r")
        await _read_until(events, output, "ECHO:héllo".encode())

        stopped = await asyncio.wait_for(service.stop_session(session.id), timeout=15)
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
    service = TerminalService(_engine(tmp_path), home=tmp_path, env=_environment())
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id, rows=24, columns=80)
    events = attachment.__aiter__()

    try:
        await attachment.write(
            _shell_command(
                sys.executable,
                str(parent),
                str(nested),
                str(marker),
            )
        )
        output = bytearray(attachment.initial.output)
        await _read_until(events, output, b"NESTED_READY")
        stopped = await asyncio.wait_for(service.stop_session(session.id), timeout=15)
        assert stopped.status is TerminalStatus.EXITED
        assert stopped.error is None
        await asyncio.sleep(2)
        assert not marker.exists()
    finally:
        await attachment.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_platform_shell_exit_settles_session(tmp_path: Path) -> None:
    service = TerminalService(_engine(tmp_path), home=tmp_path, env=_environment())
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id, rows=24, columns=80)
    events = attachment.__aiter__()

    try:
        await attachment.write("exit\r")
        while True:
            event = await asyncio.wait_for(anext(events), timeout=15)
            if (
                isinstance(event, TerminalStateEvent)
                and event.session.status is TerminalStatus.EXITED
            ):
                assert event.session.error is None
                break
    finally:
        await attachment.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_platform_detach_does_not_stop_terminal(tmp_path: Path) -> None:
    service = TerminalService(_engine(tmp_path), home=tmp_path, env=_environment())
    await service.start()
    session = await service.create_session()
    first = await service.attach(session.id, rows=24, columns=80)
    await first.aclose()

    try:
        assert (await service.get_session(session.id)).status is TerminalStatus.RUNNING
        second = await service.attach(session.id, rows=30, columns=100)
        try:
            assert second.initial.output
        finally:
            await second.aclose()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_platform_two_clients_share_output_and_transfer_control(
    tmp_path: Path,
) -> None:
    observer_side_effect = tmp_path / "observer-side-effect.txt"
    service = TerminalService(_engine(tmp_path), home=tmp_path, env=_environment())
    await service.start()
    session = await service.create_session()
    controller = await service.attach(session.id, rows=28, columns=96)
    observer = await service.attach(session.id, rows=35, columns=110)
    controller_events = controller.__aiter__()
    observer_events = observer.__aiter__()

    try:
        controller_output = bytearray(controller.initial.output)
        observer_output = bytearray(observer.initial.output)
        await observer.write(
            _shell_command(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(observer_side_effect)!r}).touch()",
            )
        )
        await asyncio.sleep(0.5)
        assert not observer_side_effect.exists()

        await controller.write(
            _shell_command(sys.executable, "-c", "print('CONTROLLER_OUTPUT')")
        )
        await _read_until(controller_events, controller_output, b"CONTROLLER_OUTPUT")
        await _read_until(observer_events, observer_output, b"CONTROLLER_OUTPUT")

        await observer.claim()
        await observer.write(
            _shell_command(sys.executable, "-c", "print('OBSERVER_PROMOTED')")
        )
        await _read_until(controller_events, controller_output, b"OBSERVER_PROMOTED")
        await _read_until(observer_events, observer_output, b"OBSERVER_PROMOTED")
        promoted = await service.get_session(session.id)
        assert (promoted.rows, promoted.columns) == (35, 110)
    finally:
        await controller.aclose()
        await observer.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_platform_terminal_keeps_running_without_attached_clients(
    tmp_path: Path,
) -> None:
    delayed = tmp_path / "delayed_output.py"
    delayed.write_text(
        "import time\n"
        "print('DETACHED_STARTED', flush=True)\n"
        "time.sleep(0.5)\n"
        "print('DETACHED_OUTPUT', flush=True)\n",
        encoding="utf-8",
    )
    service = TerminalService(_engine(tmp_path), home=tmp_path, env=_environment())
    await service.start()
    session = await service.create_session()
    first = await service.attach(session.id, rows=24, columns=80)
    first_events = first.__aiter__()
    first_output = bytearray(first.initial.output)
    await first.write(_shell_command(sys.executable, str(delayed)))
    await _read_until(first_events, first_output, b"DETACHED_STARTED")
    await first.aclose()

    try:
        await asyncio.sleep(1)
        second = await service.attach(session.id, rows=30, columns=100)
        events = second.__aiter__()
        output = bytearray(second.initial.output)
        try:
            await _read_until(events, output, b"DETACHED_OUTPUT")
        finally:
            await second.aclose()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_platform_ctrl_c_returns_to_the_owned_shell(tmp_path: Path) -> None:
    blocking = tmp_path / "blocking_child.py"
    blocking.write_text(
        "import time\nprint('INTERRUPT_READY', flush=True)\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    service = TerminalService(_engine(tmp_path), home=tmp_path, env=_environment())
    await service.start()
    session = await service.create_session()
    attachment = await service.attach(session.id, rows=24, columns=80)
    events = attachment.__aiter__()
    output = bytearray(attachment.initial.output)

    try:
        await attachment.write(_shell_command(sys.executable, str(blocking)))
        await _read_until(events, output, b"INTERRUPT_READY")
        await attachment.write("\x03")
        await asyncio.sleep(0.3)
        await attachment.write(
            _shell_command(sys.executable, "-c", "print('AFTER_INTERRUPT')")
        )
        await _read_until(events, output, b"AFTER_INTERRUPT")
    finally:
        await attachment.aclose()
        await service.close()
