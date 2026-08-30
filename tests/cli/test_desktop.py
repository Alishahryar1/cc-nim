"""Desktop shell lifecycle and singleton contracts."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from free_claude_code.cli import commands
from free_claude_code.cli.commands import ServerStatus, ServerSupervisor
from free_claude_code.cli.desktop import DesktopController
from free_claude_code.config.settings import Settings
from free_claude_code.core.interprocess_lock import InterprocessFileLock


def _settings() -> Settings:
    return Settings.model_construct(host="0.0.0.0", port=8082)


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

        def desktop_gateway_url(self) -> str | None:
            return None

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
    assert supervisor.run_arguments == [None]
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
            assert open_admin_browser is None
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

        def desktop_gateway_url(self) -> str | None:
            return None

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
        patch.object(desktop, "InterprocessFileLock", return_value=instance_lock),
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
        patch.object(desktop, "InterprocessFileLock", return_value=instance_lock),
        patch.object(desktop, "preflight_proxy", return_value=None),
        patch.object(desktop, "open_admin_when_ready", return_value=True) as open_admin,
        patch.object(desktop, "ServerSupervisor") as supervisor,
    ):
        desktop.launch_desktop(MagicMock())

    open_admin.assert_called_once_with(settings)
    supervisor.assert_not_called()
    instance_lock.release.assert_called_once_with()


def test_fresh_desktop_launch_uses_console_free_supervisor() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = True
    supervisor = MagicMock()
    controller = MagicMock()

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "InterprocessFileLock", return_value=instance_lock),
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


def test_desktop_controller_exposes_live_gateway_url_from_server() -> None:
    """The desktop integration reads the gateway URL the server published.

    Regression guard for the Greptile "published desktop gateway URL is
    never handed to the Claude Desktop integration" finding: the
    controller surfaces ``ServerOwner.desktop_gateway_url()`` so the
    config merge consumes the live TLS-prefixed endpoint rather than
    re-deriving it.
    """

    class PublishingSupervisor:
        status = ServerStatus.RUNNING
        gateway: str | None = None

        def schedule_run(self) -> bool:
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            return None

        def request_restart(self) -> bool:
            return True

        def request_stop(self) -> None:
            return None

        def desktop_gateway_url(self) -> str | None:
            return self.gateway

    supervisor = PublishingSupervisor()
    controller = DesktopController(supervisor, MagicMock(), MagicMock())

    assert controller.desktop_gateway_url() is None
    supervisor.gateway = "https://localhost:18443/claude-desktop"
    assert controller.desktop_gateway_url() == "https://localhost:18443/claude-desktop"


def test_tray_status_consumes_published_gateway_url() -> None:
    """The desktop tray surface reads the supervisor-published gateway URL.

    Regression guard for the Greptile "desktop gateway URL is unconsumed"
    finding: the tray's status notification is a production consumer of
    ``DesktopController.desktop_gateway_url()``, reporting the TLS-prefixed
    HTTPS endpoint when a front verifies and the plain-HTTP fallback
    otherwise — and only the process state before any URL is published.
    """
    from free_claude_code.cli.desktop_tray import status_notification

    class PublishingSupervisor:
        status = ServerStatus.RUNNING
        gateway: str | None = None

        def schedule_run(self) -> bool:
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            return None

        def request_restart(self) -> bool:
            return True

        def request_stop(self) -> None:
            return None

        def desktop_gateway_url(self) -> str | None:
            return self.gateway

    supervisor = PublishingSupervisor()
    controller = DesktopController(supervisor, MagicMock(), MagicMock())

    # Before the server publishes a URL, only the process state is shown.
    assert status_notification(controller) == "Server is Running."

    # HTTPS front verified: the TLS-prefixed endpoint is surfaced.
    supervisor.gateway = "https://localhost:18443/claude-desktop"
    assert (
        status_notification(controller)
        == "Server is Running. Gateway: https://localhost:18443/claude-desktop"
    )

    # Plain-HTTP fallback: the fallback endpoint is surfaced.
    supervisor.gateway = "http://127.0.0.1:8082/claude-desktop"
    assert (
        status_notification(controller)
        == "Server is Running. Gateway: http://127.0.0.1:8082/claude-desktop"
    )


def test_desktop_quit_stops_the_managed_tls_front() -> None:
    """The desktop quit path tears down the HTTPS front with the server.

    Regression guard for the Greptile "managed TLS proxy never enters the
    desktop lifecycle" finding: the desktop shell drives the same
    ``ServerSupervisor`` generation lifecycle, so quitting the tray must
    stop the generation's ``CaddyTlsProxy`` after the server exits.
    """
    from free_claude_code.cli import commands, desktop

    settings = _settings()
    events: list[str] = []
    server_running = threading.Event()

    class FakeTlsProxy:
        def __init__(self, proxy_settings: Settings) -> None:
            assert proxy_settings is settings
            events.append("construct")

        def start(self) -> bool:
            events.append("start")
            return True

        def stop(self) -> None:
            events.append("stop")

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        def run(self):
            events.append("server-run")
            server_running.set()
            while not self.should_exit:
                time.sleep(0.05)

    def fake_config(app, **kwargs):
        return SimpleNamespace(app=app, kwargs=kwargs)

    class QuittingTray:
        def __init__(self, controller: DesktopController) -> None:
            self._controller = controller

        def run(self) -> None:
            assert server_running.wait(2)
            self._controller.quit()

        def stop(self) -> None:
            return None

    supervisor = ServerSupervisor(console_logging=False)
    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "get_settings", return_value=settings),
        patch.object(commands, "get_settings", return_value=settings),
        patch.object(commands.uvicorn, "Config", side_effect=fake_config),
        patch.object(commands.uvicorn, "Server", side_effect=FakeServer),
        patch.object(
            commands,
            "build_asgi_app",
            return_value=SimpleNamespace(runtime=SimpleNamespace(is_closed=False)),
        ),
        patch.object(commands, "CaddyTlsProxy", side_effect=FakeTlsProxy),
        patch.object(commands, "GATEWAY_HEALTH_UPGRADE_SECONDS", 0.5),
        patch.object(commands, "probe_fcc_front", return_value=True),
        patch.object(commands, "schedule_open_admin_browser"),
        patch.object(commands, "kill_all_best_effort"),
    ):
        DesktopController(supervisor, QuittingTray, lambda: None).run()

    assert events == ["construct", "start", "server-run", "stop"]


def test_verified_https_readiness_repersists_desktop_config() -> None:
    """Readiness upgrade rewrites the persisted Claude Desktop config.

    Regression guard for the Greptile "TLS endpoint stays unconfigured"
    finding: the pre-lifecycle merge records the plain-HTTP fallback while
    the front is still starting. Once the readiness task verifies the
    front, the config file must be re-merged with the verified root —
    Claude Desktop reads the config file, not this process's memory.
    """

    supervisor = ServerSupervisor(console_logging=False)
    settings = Settings.model_construct(
        host="0.0.0.0", port=8082, tls_proxy_enabled=True, tls_proxy_port=8444
    )

    with (
        patch.object(
            commands,
            "configure_claude_desktop_config",
            return_value=True,
        ) as remerge,
    ):
        supervisor._publish_verified_https_gateway_url(
            settings, "https://localhost:8444"
        )

    remerge.assert_called_once_with(
        settings=settings, gateway_base_url="https://localhost:8444/claude-desktop"
    )


def test_verified_https_repersist_failure_does_not_raise() -> None:
    """A config re-merge failure downgrades to a warning, not a crash.

    The re-merge runs inside the live serving window on the readiness
    thread; a filesystem failure there must not take the generation down.
    """

    supervisor = ServerSupervisor(console_logging=False)
    settings = Settings.model_construct(
        host="0.0.0.0", port=8082, tls_proxy_enabled=True, tls_proxy_port=8444
    )

    with (
        patch.object(
            commands,
            "configure_claude_desktop_config",
            side_effect=OSError("disk full"),
        ),
    ):
        supervisor._publish_verified_https_gateway_url(
            settings, "https://localhost:8444"
        )

    assert supervisor.desktop_gateway_url() == "https://localhost:8444/claude-desktop"
