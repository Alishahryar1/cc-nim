"""Linux desktop shell: tray probe, console fallback, and entrypoint routing."""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.cli.commands import ServerStatus
from free_claude_code.cli.desktop import DesktopController
from free_claude_code.cli.desktop_console import ConsoleDesktopTray, launch
from free_claude_code.cli.desktop_entrypoint import launch as entrypoint_launch
from free_claude_code.cli.desktop_tray import tray_is_available


def test_tray_is_available_true_when_icon_constructs():
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch("free_claude_code.cli.desktop_tray.Icon", MagicMock()),
    ):
        available, reason = tray_is_available()

    assert available is True
    assert reason == ""


def test_tray_is_available_reports_backend_import_error():
    with (
        patch("free_claude_code.cli.desktop_tray._create_icon"),
        patch(
            "free_claude_code.cli.desktop_tray.Icon",
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
        patch("free_claude_code.cli.desktop_tray.Icon", MagicMock(side_effect=error)),
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
