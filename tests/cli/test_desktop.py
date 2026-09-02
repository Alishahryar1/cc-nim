"""Tray-less FCC desktop host lifecycle and singleton contracts."""

import signal
import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from free_claude_code.cli import desktop
from free_claude_code.cli.tls_proxy import FrontStartError
from free_claude_code.config.settings import Settings
from free_claude_code.core.interprocess_lock import InterprocessFileLock


def _settings(
    *,
    tls_proxy_enabled: bool = True,
    tls_proxy_port: int = 8443,
) -> Settings:
    return Settings.model_construct(
        host="0.0.0.0",
        port=8082,
        tls_proxy_enabled=tls_proxy_enabled,
        tls_proxy_port=tls_proxy_port,
        desktop_gateway_prefix="claude-desktop",
    )


@pytest.fixture
def fake_supervisor(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    supervisor = MagicMock()
    supervisor.status = "Running"
    stopped = threading.Event()

    def run(*, open_admin_browser: bool | None = None) -> None:
        stopped.wait()

    supervisor.run.side_effect = run
    supervisor.request_stop.side_effect = stopped.set
    supervisor.schedule_run.return_value = True
    monkeypatch.setattr(desktop, "ServerSupervisor", MagicMock(return_value=supervisor))
    return supervisor


@pytest.fixture
def fake_front(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    front = MagicMock()
    monkeypatch.setattr(desktop, "CaddyTlsProxy", MagicMock(return_value=front))
    return front


@pytest.fixture
def acquired_lock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    lock = MagicMock()
    lock.acquire.return_value = True
    monkeypatch.setattr(desktop, "InterprocessFileLock", MagicMock(return_value=lock))
    return lock


@pytest.fixture
def ready_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = _settings()
    monkeypatch.setattr(desktop, "load_server_settings", lambda: settings)
    return settings


def test_launch_runs_front_server_and_stops_both_in_order(
    fake_supervisor: MagicMock,
    fake_front: MagicMock,
    acquired_lock: MagicMock,
    ready_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    order: list[str] = []
    fake_front.start.side_effect = lambda: order.append("front-start")
    fake_front.stop.side_effect = lambda: order.append("front-stop")
    fake_supervisor.schedule_run.side_effect = lambda: (
        order.append("server-schedule") or True
    )

    printed = threading.Event()

    def fake_print(*_args: object, **_kwargs: object) -> None:
        order.append("print-url")
        printed.set()

    import builtins

    monkeypatch.setattr(builtins, "print", fake_print)
    monkeypatch.setattr(desktop, "_wait_for_signal", lambda: order.append("wait"))
    # signal.signal is main-thread-only; this scenario runs the host on a
    # worker thread so the registration itself is patched out (the dedicated
    # signal test covers real registration on the main thread).
    monkeypatch.setattr(desktop.signal, "signal", lambda *_: None)

    stopper = threading.Thread(target=desktop.launch_desktop)
    stopper.start()
    # The gateway URL is printed before the host blocks on signals.
    assert printed.wait(timeout=5)

    stopper.join(timeout=5)

    assert order == [
        "front-start",
        "server-schedule",
        "print-url",
        "wait",
        "front-stop",
    ]
    fake_supervisor.request_stop.assert_called_once()
    fake_front.stop.assert_called_once()
    acquired_lock.release.assert_called_once()


def test_launch_prints_desktop_gateway_url(
    fake_supervisor: MagicMock,
    fake_front: MagicMock,
    acquired_lock: MagicMock,
    ready_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(desktop, "_wait_for_signal", lambda: None)

    desktop.launch_desktop()

    out = capsys.readouterr().out
    assert "Claude Desktop gateway: https://localhost:8443/claude-desktop" in out


def test_front_start_failure_aborts_before_server(
    fake_supervisor: MagicMock,
    fake_front: MagicMock,
    acquired_lock: MagicMock,
    ready_settings: Settings,
) -> None:
    fake_front.start.side_effect = FrontStartError("port busy")

    with pytest.raises(FrontStartError, match="port busy"):
        desktop.launch_desktop()

    fake_supervisor.schedule_run.assert_not_called()
    fake_front.stop.assert_called_once()
    acquired_lock.release.assert_called_once()


def test_front_start_failure_skipped_when_tls_disabled(
    fake_supervisor: MagicMock,
    fake_front: MagicMock,
    acquired_lock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tls_proxy_enabled=False)
    monkeypatch.setattr(desktop, "load_server_settings", lambda: settings)
    monkeypatch.setattr(desktop, "_wait_for_signal", lambda: None)

    desktop.launch_desktop()

    fake_front.start.assert_not_called()
    fake_front.stop.assert_called_once()
    fake_supervisor.request_stop.assert_called_once()


def test_second_desktop_launch_opens_existing_admin_without_new_server(
    fake_supervisor: MagicMock,
    fake_front: MagicMock,
    ready_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = False
    monkeypatch.setattr(
        desktop, "InterprocessFileLock", MagicMock(return_value=instance_lock)
    )
    open_admin = MagicMock()
    monkeypatch.setattr(desktop, "open_admin_when_ready", open_admin)

    desktop.launch_desktop()

    open_admin.assert_called_once_with(ready_settings)
    fake_supervisor.schedule_run.assert_not_called()
    fake_front.start.assert_not_called()
    instance_lock.release.assert_not_called()


def test_launch_registers_sigint_and_sigterm_handlers(
    fake_supervisor: MagicMock,
    fake_front: MagicMock,
    acquired_lock: MagicMock,
    ready_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[int, Callable[..., None]] = {}

    def fake_signal(signalnum: int, handler: Callable[..., None]) -> None:
        registered[signalnum] = handler

    monkeypatch.setattr(desktop.signal, "signal", fake_signal)
    monkeypatch.setattr(desktop, "_wait_for_signal", lambda: None)

    desktop.launch_desktop()

    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
    # Handlers set the module stop event.
    desktop._stop_event.clear()
    registered[signal.SIGINT]()
    assert desktop._shutdown_requested() is True


def test_lock_release_runs_even_when_server_scheduling_fails(
    fake_front: MagicMock,
    acquired_lock: MagicMock,
    ready_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = MagicMock()
    supervisor.schedule_run.return_value = False
    monkeypatch.setattr(desktop, "ServerSupervisor", MagicMock(return_value=supervisor))
    monkeypatch.setattr(desktop, "_wait_for_signal", lambda: None)

    with pytest.raises(RuntimeError, match="could not be scheduled"):
        desktop.launch_desktop()

    fake_front.stop.assert_called_once()
    acquired_lock.release.assert_called_once()


def test_desktop_instance_lock_is_exclusive_and_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "desktop.lock"
    first = InterprocessFileLock(lock_path)
    second = InterprocessFileLock(lock_path)

    assert first.acquire() is True
    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    first.release()
    assert second.acquire() is True
    second.release()


def test_tray_module_removed() -> None:
    import importlib.util

    assert importlib.util.find_spec("free_claude_code.cli.desktop_tray") is None
