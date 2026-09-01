"""Linux desktop shell: tray probe, console fallback, and entrypoint routing."""

import contextlib
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

try:
    import pystray  # noqa: F401
except Exception:
    # Headless session: pystray's Linux backend connects to the display at
    # import time, which would break collection of this whole module before
    # any console-fallback behavior can be exercised. Stub the package so
    # the tray adapter imports; every test below replaces the tray symbols
    # it actually uses.
    sys.modules["pystray"] = MagicMock()

from free_claude_code.cli.commands import ServerStatus
from free_claude_code.cli.desktop import DesktopController
from free_claude_code.cli.desktop_assets import app_icon_bytes
from free_claude_code.cli.desktop_console import ConsoleDesktopTray, launch
from free_claude_code.cli.desktop_entrypoint import launch as entrypoint_launch
from free_claude_code.cli.desktop_tray import tray_is_available


def test_tray_is_available_true_when_icon_constructs():
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("pystray.Icon", MagicMock()),
    ):
        available, reason = tray_is_available()

    assert available is True
    assert reason == ""


def test_tray_probe_releases_backend_display_connection():
    """A probe icon's X11 display is closed eagerly, not by ``__del__``.

    Regression guard for the unraisable-exception flake: the X11 backend
    opens a display connection in ``Icon.__init__`` and closes it only in a
    garbage-collected destructor, which can run during an unrelated test
    and fail on the already-closed socket (turning into a suite-wide
    ``PytestUnraisableExceptionWarning`` failure under ``filterwarnings =
    error``). The probe must close the connection itself and leave the
    destructor a no-op stand-in.
    """

    class XorgLikeIcon:
        """Backend icon holding an X11-style display connection."""

        def __init__(self) -> None:
            self._display = MagicMock()

        def __del__(self) -> None:
            self._display.close()

    icon = XorgLikeIcon()
    display = icon._display
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("pystray.Icon", MagicMock(return_value=icon)),
    ):
        available, _ = tray_is_available()

    assert available is True
    display.close.assert_called_once()
    # The stand-in left behind keeps the destructor harmless.
    assert icon._display is not display
    icon.__del__()  # The destructor runs against the no-op stand-in.


def test_tray_is_available_reports_backend_import_error():
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch(
            "pystray.Icon",
            MagicMock(side_effect=ImportError("gir bindings missing")),
        ),
    ):
        available, reason = tray_is_available()

    assert available is False
    assert "appindicator" in reason


def test_tray_is_available_reports_other_construction_errors_verbatim():
    error = RuntimeError("no display")
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("pystray.Icon", MagicMock(side_effect=error)),
    ):
        available, reason = tray_is_available()

    assert available is False
    assert reason == "no display"


def test_tray_is_available_reports_artwork_load_failure():
    with patch(
        "free_claude_code.cli.desktop_tray._create_icon",
        MagicMock(side_effect=OSError("missing asset")),
    ):
        available, reason = tray_is_available()

    assert available is False
    assert "artwork" in reason


def test_console_tray_run_blocks_until_stop():
    tray = ConsoleDesktopTray(MagicMock(spec=DesktopController))
    finished = threading.Event()

    def run_tray():
        tray.run()
        finished.set()

    worker = threading.Thread(target=run_tray)
    worker.start()
    assert not finished.wait(timeout=0.2)
    tray.stop()
    assert finished.wait(timeout=5)
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_console_fallback_uses_null_tray_with_reason_notice(capsys):
    with (
        patch("free_claude_code.cli.desktop_tray.tray_is_available") as probe,
        patch("free_claude_code.cli.desktop_console.launch_desktop") as start,
    ):
        probe.return_value = (False, "backend missing")
        launch()

    factory = start.call_args.args[0]
    assert factory is ConsoleDesktopTray
    output = capsys.readouterr().out
    assert "console mode" in output
    assert "backend missing" in output


class _RaisingModule:
    """Stands in for ``desktop_tray`` when its pystray import would raise."""

    def __getattr__(self, name: str):
        raise RuntimeError("no display")


def test_launch_falls_back_to_console_when_native_adapter_import_raises(capsys):
    with (
        patch.dict(
            sys.modules, {"free_claude_code.cli.desktop_tray": _RaisingModule()}
        ),
        patch("free_claude_code.cli.desktop_console.launch_desktop") as start,
    ):
        launch()

    assert start.call_args.args[0] is ConsoleDesktopTray
    output = capsys.readouterr().out
    assert "console mode" in output
    assert "failed to load" in output
    assert "no display" in output


def test_console_launch_uses_native_tray_when_probe_succeeds():
    with (
        patch("free_claude_code.cli.desktop_tray.tray_is_available") as probe,
        patch("free_claude_code.cli.desktop_console.launch_desktop") as start,
    ):
        probe.return_value = (True, "")
        launch()

    from free_claude_code.cli.desktop_tray import PystrayDesktopTray

    assert start.call_args.args[0] is PystrayDesktopTray


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_entrypoint_routes_supported_native_platforms_to_tray(platform):
    with (
        patch.object(sys, "platform", platform),
        patch("free_claude_code.cli.desktop_tray.launch") as tray_launch,
    ):
        entrypoint_launch([])

    tray_launch.assert_called_once()


def test_entrypoint_defers_desktop_routing_merge_to_the_lifecycle():
    """No config merge happens at the entrypoint.

    Regression guard for the Greptile "desktop startup writes a cleartext
    credential endpoint" finding: the entrypoint runs before any lifecycle
    has started or verified a TLS front, so a merge here would write the
    reusable gateway credential pointing at the plain-HTTP fallback. The
    lifecycle layers merge only behind a verified HTTPS front.
    """

    events: list[str] = []

    with (
        patch.object(sys, "platform", "linux"),
        patch(
            "free_claude_code.cli.desktop_console.launch_desktop",
            side_effect=lambda *_: events.append("lifecycle"),
        ),
    ):
        entrypoint_launch([])

    assert events == ["lifecycle"]


def test_entrypoint_routes_linux_through_probe_and_console_mode(capsys):
    with (
        patch.object(sys, "platform", "linux"),
        patch("free_claude_code.cli.desktop_tray.tray_is_available") as probe,
        patch("free_claude_code.cli.desktop_console.launch_desktop") as start,
    ):
        probe.return_value = (False, "backend missing")
        entrypoint_launch([])

    assert start.call_args.args[0] is ConsoleDesktopTray


def test_entrypoint_still_rejects_unsupported_platforms(capsys):
    with (
        patch.object(sys, "platform", "freebsd"),
        pytest.raises(SystemExit) as exit_info,
    ):
        entrypoint_launch([])

    assert exit_info.value.code == 1
    assert "Linux" in capsys.readouterr().err


def test_console_tray_consumes_published_gateway_url() -> None:
    """The tray-less console surface reports the live published gateway URL.

    Regression guard for the Greptile "Claude Desktop gateway is never
    configured / console overclaims routing" finding: the console tray is a
    production consumer of ``DesktopController.desktop_gateway_url()``,
    printing the endpoint the active generation actually published — the
    TLS-prefixed HTTPS URL when a front verifies, the plain-HTTP fallback
    otherwise — and re-printing only when the value changes.
    """
    from free_claude_code.cli.desktop_console import console_gateway_line

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

    # Before the server publishes a URL, the console prints nothing.
    assert console_gateway_line(controller) is None

    # Plain-HTTP fallback published: the fallback endpoint is surfaced.
    supervisor.gateway = "http://127.0.0.1:8082/claude-desktop"
    assert (
        console_gateway_line(controller)
        == "Claude Desktop gateway: http://127.0.0.1:8082/claude-desktop"
    )

    # HTTPS front verified: the TLS-prefixed endpoint is surfaced.
    supervisor.gateway = "https://localhost:18443/claude-desktop"
    assert (
        console_gateway_line(controller)
        == "Claude Desktop gateway: https://localhost:18443/claude-desktop"
    )


def test_console_tray_prints_gateway_url_changes_while_running(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The console tray prints each new gateway URL exactly once while live."""

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
    tray = ConsoleDesktopTray(controller)

    def wait_for_printed(expected: str) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if expected in capsys.readouterr().out:
                return
            time.sleep(0.05)
        raise AssertionError(f"{expected} was never printed")

    worker = threading.Thread(target=tray.run, daemon=True)
    worker.start()
    try:
        supervisor.gateway = "http://127.0.0.1:8082/claude-desktop"
        wait_for_printed("http://127.0.0.1:8082/claude-desktop")
        supervisor.gateway = "https://localhost:18443/claude-desktop"
        wait_for_printed("https://localhost:18443/claude-desktop")
    finally:
        tray.stop()
        worker.join(timeout=5)

    assert not worker.is_alive()


def test_console_tray_lifecycle_drives_controller_to_clean_stop():
    events: list[str] = []
    server_started = threading.Event()

    class FakeSupervisor:
        status = ServerStatus.STOPPED

        def schedule_run(self) -> bool:
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            events.append("server-run")
            server_started.set()

        def request_restart(self) -> bool:
            return True

        def request_stop(self) -> None:
            events.append("server-stop")

        def desktop_gateway_url(self) -> str | None:
            return None

    controller = DesktopController(
        FakeSupervisor(),
        lambda controller: ConsoleDesktopTray(controller),
        lambda: None,
    )

    worker = threading.Thread(target=controller.run, daemon=True)
    worker.start()
    assert server_started.wait(timeout=5)
    controller.quit()
    worker.join(timeout=5)

    assert not worker.is_alive()
    # quit() stops the server, then controller.run()'s finally chain repeats
    # the idempotent stop; the null tray must survive both.
    assert events[0] == "server-run"
    assert events.count("server-stop") >= 1


@contextlib.contextmanager
def _headless_entrypoint():
    """Import ``desktop_entrypoint`` as if the tray backend were broken.

    Drops the cached entrypoint and tray modules, then makes the tray
    module raise on import the way pystray does on a headless Linux
    session without an X11/appindicator backend.
    """
    saved_entrypoint = sys.modules.pop("free_claude_code.cli.desktop_entrypoint", None)
    saved_tray = sys.modules.pop("free_claude_code.cli.desktop_tray", None)
    try:
        with patch.dict(
            sys.modules,
            {"free_claude_code.cli.desktop_tray": _RaisingModule()},
        ):
            import free_claude_code.cli.desktop_entrypoint as fresh_entrypoint

            yield fresh_entrypoint
    finally:
        sys.modules.pop("free_claude_code.cli.desktop_entrypoint", None)
        if saved_entrypoint is not None:
            sys.modules["free_claude_code.cli.desktop_entrypoint"] = saved_entrypoint
        if saved_tray is not None:
            sys.modules["free_claude_code.cli.desktop_tray"] = saved_tray


def test_headless_entrypoint_import_survives_broken_tray_backend():
    with _headless_entrypoint() as fresh_entrypoint:
        assert callable(fresh_entrypoint.launch)


def test_headless_entrypoint_still_exports_icon(tmp_path):
    destination = tmp_path / "icons" / "app-icon.png"

    with _headless_entrypoint() as fresh_entrypoint:
        fresh_entrypoint.launch(["--export-icon", str(destination)])

    assert destination.read_bytes() == app_icon_bytes(".png")


def test_headless_entrypoint_routes_linux_to_console_mode(capsys):
    with (
        _headless_entrypoint() as fresh_entrypoint,
        patch.object(sys, "platform", "linux"),
        patch("free_claude_code.cli.desktop_console.launch_desktop") as start,
    ):
        fresh_entrypoint.launch([])

    assert start.call_args.args[0] is ConsoleDesktopTray
    assert "failed to load" in capsys.readouterr().out


def test_tray_module_imports_without_display():
    """pystray resolves an X11 display at import; the module must not."""

    import importlib

    module = sys.modules["free_claude_code.cli.desktop_tray"]
    # A ``None`` entry makes ``import pystray`` fail as if it were absent.
    try:
        with patch.dict(sys.modules, {"pystray": None}):
            reloaded = importlib.reload(module)
    finally:
        importlib.reload(module)
    assert reloaded.tray_is_available.__module__ == module.__name__


def test_merge_verified_gateway_passes_verified_url_explicitly():
    from free_claude_code.cli.desktop import _merge_verified_gateway

    settings = MagicMock(name="settings")
    verified_url = "https://localhost:8443/claude-desktop"
    with (
        patch(
            "free_claude_code.cli.desktop.verified_https_gateway_url",
            return_value=verified_url,
        ),
        patch(
            "free_claude_code.cli.desktop.configure_claude_desktop_config"
        ) as configure,
    ):
        _merge_verified_gateway(settings)

    configure.assert_called_once_with(settings=settings, gateway_base_url=verified_url)


def test_merge_verified_gateway_skips_when_front_unverified():
    from free_claude_code.cli.desktop import _merge_verified_gateway

    with (
        patch(
            "free_claude_code.cli.desktop.verified_https_gateway_url",
            return_value=None,
        ),
        patch(
            "free_claude_code.cli.desktop.configure_claude_desktop_config"
        ) as configure,
    ):
        _merge_verified_gateway(MagicMock(name="settings"))

    configure.assert_not_called()


def test_merge_verified_gateway_swallows_failures():
    from free_claude_code.cli.desktop import _merge_verified_gateway

    with (
        patch(
            "free_claude_code.cli.desktop.verified_https_gateway_url",
            return_value="https://localhost:8443/claude-desktop",
        ),
        patch(
            "free_claude_code.cli.desktop.configure_claude_desktop_config",
            side_effect=OSError("disk full"),
        ),
    ):
        _merge_verified_gateway(MagicMock(name="settings"))  # must not raise


def test_launch_desktop_merges_claude_desktop_config():
    from free_claude_code.cli import desktop as desktop_module

    merged: list[object] = []

    def fake_configure(path=None, settings=None, gateway_base_url=None):
        merged.append(gateway_base_url)
        return True

    supervisor = MagicMock()
    supervisor.schedule_run.return_value = True
    supervisor.status = ServerStatus.STOPPED
    verified_url = "https://localhost:8443/claude-desktop"

    with (
        patch.object(desktop_module, "load_server_settings") as load_settings,
        patch.object(desktop_module.InterprocessFileLock, "acquire", return_value=True),
        patch.object(desktop_module, "preflight_proxy", return_value=object()),
        patch.object(desktop_module, "CaddyTlsProxy") as tls_proxy_cls,
        patch.object(
            desktop_module, "verified_https_gateway_url", return_value=verified_url
        ),
        patch.object(desktop_module, "configure_claude_desktop_config", fake_configure),
        patch.object(desktop_module, "DesktopController", MagicMock()) as controller,
    ):
        load_settings.return_value = MagicMock(name="settings")
        tls_proxy_cls.return_value.start.return_value = True
        controller.return_value.run.side_effect = RuntimeError("stop loop")

        with contextlib.suppress(RuntimeError):
            desktop_module.launch_desktop(MagicMock())

    assert merged == [verified_url]
    tls_proxy_cls.return_value.start.assert_called_once()
    tls_proxy_cls.return_value.stop.assert_called_once()


def test_tray_launch_item_spawns_claude_desktop_without_notification():
    from free_claude_code.cli.desktop_tray import PystrayDesktopTray

    controller = MagicMock(spec=DesktopController)
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("pystray.Icon", MagicMock()),
        patch(
            "free_claude_code.cli.desktop_tray.ensure_configured_and_launch"
        ) as spawn,
    ):
        tray = PystrayDesktopTray(controller)
        tray._launch_claude_desktop(tray._icon, MagicMock(name="item"))

    spawn.assert_called_once()


def test_tray_launch_item_notifies_when_binary_missing():
    from free_claude_code.cli.desktop_tray import PystrayDesktopTray

    controller = MagicMock(spec=DesktopController)
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("pystray.Icon", MagicMock()),
        patch(
            "free_claude_code.cli.desktop_tray.ensure_configured_and_launch",
            side_effect=FileNotFoundError("claude-desktop"),
        ),
    ):
        tray = PystrayDesktopTray(controller)
        tray._launch_claude_desktop(tray._icon, MagicMock(name="item"))

    tray._icon.notify.assert_called_once()


def test_tray_launch_item_notifies_when_no_https_front():
    # Regression guard for the Greptile tray-bypass finding: when the
    # TLS-gated launch refuses (no verified HTTPS front), the tray must
    # surface the refusal as a notification rather than crash the callback.
    from free_claude_code.cli.desktop_tray import PystrayDesktopTray

    controller = MagicMock(spec=DesktopController)
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("pystray.Icon", MagicMock()),
        patch(
            "free_claude_code.cli.desktop_tray.ensure_configured_and_launch",
            side_effect=RuntimeError("no verified FCC HTTPS front"),
        ),
    ):
        tray = PystrayDesktopTray(controller)
        tray._launch_claude_desktop(tray._icon, MagicMock(name="item"))

    tray._icon.notify.assert_called_once()
    notified = tray._icon.notify.call_args.args[0]
    assert "no verified FCC HTTPS front" in notified
