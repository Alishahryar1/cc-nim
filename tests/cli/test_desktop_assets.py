"""Packaged desktop icon and export contracts."""

import struct
from pathlib import Path

import pytest

from free_claude_code.cli import desktop_entrypoint
from free_claude_code.cli.desktop_assets import app_icon_bytes, export_app_icon


def test_packaged_icons_have_native_container_headers() -> None:
    png = app_icon_bytes(".png")
    ico = app_icon_bytes(".ico")
    icns = app_icon_bytes(".icns")

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (1024, 1024)
    assert ico[:4] == b"\x00\x00\x01\x00"
    assert struct.unpack("<H", ico[4:6])[0] >= 8
    assert icns.startswith(b"icns")
    assert struct.unpack(">I", icns[4:8])[0] == len(icns)


def test_export_app_icon_creates_parent_and_preserves_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "managed" / "AppIcon.ICNS"

    export_app_icon(destination)

    assert destination.read_bytes() == app_icon_bytes(".icns")


def test_export_app_icon_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported app icon format"):
        export_app_icon(tmp_path / "app-icon.jpg")


def test_desktop_entrypoint_exports_icon_without_launching_tray(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "app-icon.ico"
    monkeypatch.setattr(desktop_entrypoint.sys, "platform", "linux")

    desktop_entrypoint.launch(["--export-icon", str(destination)])

    assert destination.read_bytes() == app_icon_bytes(".ico")


def test_desktop_entrypoint_rejects_unknown_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        desktop_entrypoint.launch(["--unknown"])

    assert exc_info.value.code == 2
    assert "Usage: fcc-desktop" in capsys.readouterr().err
