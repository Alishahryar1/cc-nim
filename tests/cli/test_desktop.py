"""Desktop shell lifecycle and singleton contracts."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from free_claude_code.cli.commands import ServerStatus, ServerSupervisor
from free_claude_code.cli.desktop import DesktopController, DesktopInstanceLock
from free_claude_code.config.settings import Settings


def _settings() -> Settings:
    return Settings.model_construct(host="0.0.0.0", port=8082)


def test_desktop_instance_lock_is_exclusive_and_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "desktop.lock"
    first = DesktopInstanceLock(lock_path)
    second = DesktopInstanceLock(lock_path)

    assert first.acquire() is True
    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    first.release()
    assert second.acquire() is True
    second.release()


def test_supervisor_accepts_restart_during_scheduled_startup() -> None:
    supervisor = ServerSupervisor(console_logging=False)
    settings = _settings()

    with (
        patch(
            "free_claude_code.cli.commands.load_server_settings",
            return_value=settings,
        ),
        patch.object(supervisor, "_run_once", return_value=False) as run_once,
        patch("free_claude_code.cli.commands.kill_all_best_effort"),
    ):
        assert supervisor.schedule_run() is True
        assert supervisor.status is ServerStatus.STARTING
        assert supervisor.request_restart() is True
        supervisor.run(open_admin_browser=False)

    run_once.assert_called_once_with(
        settings,
        open_admin_browser=False,
        restart_generation=1,
    )
    assert supervisor.status is ServerStatus.STOPPED


def test_desktop_controller_owns_server_thread_and_graceful_quit() -> None:
    opened = threading.Event()

    class FakeSupervisor:
        def __init__(self) -> None:
            self.status = ServerStatus.STARTING
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.run_arguments: list[bool | None] = []
            self.schedule_count = 0
            self.restart_count = 0
            self.stop_count = 0

        def schedule_run(self) -> bool:
            self.schedule_count += 1
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            self.run_arguments.append(open_admin_browser)
            self.status = ServerStatus.RUNNING
            self.started.set()
            assert self.stopped.wait(2)
            self.status = ServerStatus.STOPPED

        def request_restart(self) -> bool:
            self.restart_count += 1
            return True

        def request_stop(self) -> None:
            self.stop_count += 1
            self.status = ServerStatus.STOPPING
            self.stopped.set()

    class FakeTray:
        def __init__(self, controller: DesktopController) -> None:
            self.controller = controller
            self.run_thread_id: int | None = None
            self.stop_count = 0

        def run(self) -> None:
            self.run_thread_id = threading.get_ident()
            assert supervisor.started.wait(2)
            self.controller.open_admin()
            self.controller.restart_server()
            self.controller.quit()

        def stop(self) -> None:
            self.stop_count += 1

    supervisor = FakeSupervisor()
    tray: FakeTray | None = None

    def make_tray(controller: DesktopController) -> FakeTray:
        nonlocal tray
        tray = FakeTray(controller)
        return tray

    main_thread_id = threading.get_ident()
    controller = DesktopController(supervisor, make_tray, opened.set)
    controller.run()

    assert tray is not None
    assert tray.run_thread_id == main_thread_id
    assert supervisor.run_arguments == [False]
    assert supervisor.schedule_count == 1
    assert supervisor.restart_count == 1
    assert supervisor.stop_count >= 1
    assert tray.stop_count >= 1
    assert opened.is_set()


def test_restart_during_server_startup_is_accepted_without_waiting() -> None:
    class StartupSupervisor:
        def __init__(self) -> None:
            self.status = ServerStatus.STARTING
            self.run_called = threading.Event()
            self.allow_run = threading.Event()
            self.worker_started = threading.Event()
            self.release_worker = threading.Event()
            self.run_scheduled = False
            self.restart_count = 0
            self.accepted_restart_count = 0

        def schedule_run(self) -> bool:
            self.run_scheduled = True
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            assert open_admin_browser is False
            self.run_called.set()
            assert self.allow_run.wait(2)
            self.run_scheduled = False
            self.worker_started.set()
            assert self.release_worker.wait(2)
            self.status = ServerStatus.STOPPED

        def request_restart(self) -> bool:
            self.restart_count += 1
            if self.run_scheduled:
                self.accepted_restart_count += 1
                return True
            return False

        def request_stop(self) -> None:
            self.release_worker.set()

    class WaitingTray:
        def __init__(self, _controller: DesktopController) -> None:
            self.started = threading.Event()
            self.stopped = threading.Event()

        def run(self) -> None:
            self.started.set()
            assert self.stopped.wait(2)

        def stop(self) -> None:
            self.stopped.set()

    supervisor = StartupSupervisor()
    tray: WaitingTray | None = None

    def make_tray(controller: DesktopController) -> WaitingTray:
        nonlocal tray
        tray = WaitingTray(controller)
        return tray

    controller = DesktopController(supervisor, make_tray, MagicMock())
    controller_thread = threading.Thread(target=controller.run)
    controller_thread.start()
    assert tray is not None
    assert tray.started.wait(2)
    assert supervisor.run_called.wait(2)

    restart_thread = threading.Thread(target=controller.restart_server)
    restart_thread.start()
    restart_thread.join(0.5)
    restart_blocked = restart_thread.is_alive()

    supervisor.allow_run.set()
    assert supervisor.worker_started.wait(2)
    controller.quit()
    supervisor.release_worker.set()
    restart_thread.join(2)
    controller_thread.join(2)

    assert restart_blocked is False
    assert supervisor.restart_count == 1
    assert supervisor.accepted_restart_count == 1
    assert not restart_thread.is_alive()
    assert not controller_thread.is_alive()


def test_second_desktop_launch_opens_existing_admin_without_new_server() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = False

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "DesktopInstanceLock", return_value=instance_lock),
        patch.object(desktop, "open_admin_when_ready", return_value=True) as open_admin,
        patch.object(desktop, "ServerSupervisor") as supervisor,
    ):
        desktop.launch_desktop(MagicMock())

    open_admin.assert_called_once_with(settings)
    supervisor.assert_not_called()
    instance_lock.release.assert_not_called()


def test_desktop_attaches_to_terminal_server_instead_of_binding_twice() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = True

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "DesktopInstanceLock", return_value=instance_lock),
        patch.object(desktop, "preflight_proxy", return_value=None),
        patch.object(desktop, "open_admin_when_ready", return_value=True) as open_admin,
        patch.object(desktop, "ServerSupervisor") as supervisor,
    ):
        desktop.launch_desktop(MagicMock())

    open_admin.assert_called_once_with(settings)
    supervisor.assert_not_called()
    instance_lock.release.assert_called_once_with()


def test_fresh_desktop_launch_disables_console_and_automatic_browser() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = True
    supervisor = MagicMock()
    controller = MagicMock()

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "DesktopInstanceLock", return_value=instance_lock),
        patch.object(desktop, "preflight_proxy", return_value="connection refused"),
        patch.object(desktop, "ServerSupervisor", return_value=supervisor) as owner,
        patch.object(desktop, "DesktopController", return_value=controller) as shell,
    ):
        tray_factory = MagicMock()
        desktop.launch_desktop(tray_factory)

    owner.assert_called_once_with(console_logging=False)
    assert shell.call_args.args[:2] == (supervisor, tray_factory)
    controller.run.assert_called_once_with()
    instance_lock.release.assert_called_once_with()
