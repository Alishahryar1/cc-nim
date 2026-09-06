"""Unit tests for shared installed-client launcher process helpers."""

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

from free_claude_code.cli.launchers import common

_PROXY_URL = "http://127.0.0.1:9191"
_SERVER_COMMAND = [
    sys.executable,
    "-m",
    "free_claude_code.cli.entrypoints",
    "serve",
]


def test_ensure_server_running_true_when_proxy_already_healthy() -> None:
    with (
        patch.object(common, "preflight_proxy", return_value=None),
        patch.object(common, "InterprocessFileLock") as lock_cls,
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is True

    lock_cls.assert_not_called()
    popen.assert_not_called()


def test_ensure_server_running_starts_server_and_waits_until_ready(
    tmp_path: Path,
) -> None:
    server_ready = False
    health_calls: list[str] = []

    def fake_preflight(_proxy_root_url: str) -> str | None:
        health_calls.append(_proxy_root_url)
        return None if server_ready else "connection refused"

    def fake_popen(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal server_ready
        server_ready = True
        return MagicMock()

    lock = MagicMock()
    lock.acquire.return_value = True

    with (
        patch.object(common, "preflight_proxy", side_effect=fake_preflight),
        patch.object(common, "InterprocessFileLock", return_value=lock) as lock_cls,
        patch.object(common, "config_dir_path", return_value=tmp_path),
        patch.object(
            common, "server_log_path", return_value=tmp_path / "logs" / "server.log"
        ),
        patch(
            "free_claude_code.cli.launchers.common.subprocess.Popen",
            side_effect=fake_popen,
        ) as popen,
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is True

    assert health_calls == [_PROXY_URL, _PROXY_URL, _PROXY_URL]
    popen.assert_called_once()
    assert popen.call_args.args[0] == _SERVER_COMMAND
    assert popen.call_args.kwargs["start_new_session"] is True
    lock.acquire.assert_called_once_with(wait=False)
    lock.release.assert_called_once()
    lock_path = lock_cls.call_args.args[0]
    assert lock_path == tmp_path / "server.startup.lock"
    assert (tmp_path / "logs" / "server.log").exists()


def test_ensure_server_running_skips_spawn_when_proxy_becomes_ready_after_lock(
    tmp_path: Path,
) -> None:
    health_calls: list[str] = []

    def fake_preflight(_proxy_root_url: str) -> str | None:
        health_calls.append(_proxy_root_url)
        return "connection refused" if len(health_calls) == 1 else None

    lock = MagicMock()
    lock.acquire.return_value = True

    with (
        patch.object(common, "preflight_proxy", side_effect=fake_preflight),
        patch.object(common, "InterprocessFileLock", return_value=lock),
        patch.object(common, "config_dir_path", return_value=tmp_path),
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is True

    assert len(health_calls) == 2
    popen.assert_not_called()
    lock.release.assert_called_once()


def test_ensure_server_running_waits_for_peer_when_startup_lock_is_held() -> None:
    health_results: Iterator[str | None] = iter(
        ["connection refused", "connection refused", None]
    )

    def fake_preflight(_proxy_root_url: str) -> str | None:
        return next(health_results)

    lock = MagicMock()
    lock.acquire.return_value = False

    with (
        patch.object(common, "preflight_proxy", side_effect=fake_preflight),
        patch.object(common, "InterprocessFileLock", return_value=lock),
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
        patch.object(common.time, "sleep"),
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is True

    assert lock.acquire.call_count >= 2
    lock.acquire.assert_called_with(wait=False)
    popen.assert_not_called()
    lock.release.assert_not_called()


def test_ensure_server_running_takes_over_when_starter_gives_up(
    tmp_path: Path,
) -> None:
    health_results: Iterator[str | None] = iter(
        [
            "connection refused",
            "connection refused",
            "connection refused",
            None,
        ]
    )

    def fake_preflight(_proxy_root_url: str) -> str | None:
        return next(health_results)

    def fake_popen(*_args: object, **_kwargs: object) -> MagicMock:
        return MagicMock()

    lock = MagicMock()
    # The starter is alive for the first poll, then gives up and releases.
    lock.acquire.side_effect = [False, True]

    with (
        patch.object(common, "preflight_proxy", side_effect=fake_preflight),
        patch.object(common, "InterprocessFileLock", return_value=lock),
        patch.object(common, "config_dir_path", return_value=tmp_path),
        patch.object(
            common,
            "server_log_path",
            return_value=tmp_path / "logs" / "server.log",
        ),
        patch(
            "free_claude_code.cli.launchers.common.subprocess.Popen",
            side_effect=fake_popen,
        ) as popen,
        patch.object(common.time, "sleep"),
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is True

    popen.assert_called_once()
    assert popen.call_args.args[0] == _SERVER_COMMAND
    lock.release.assert_called_once()


def test_ensure_server_running_returns_false_when_server_cannot_be_spawned(
    tmp_path: Path,
) -> None:
    lock = MagicMock()
    lock.acquire.return_value = True

    with (
        patch.object(common, "preflight_proxy", return_value="connection refused"),
        patch.object(common, "InterprocessFileLock", return_value=lock),
        patch.object(common, "config_dir_path", return_value=tmp_path),
        patch.object(
            common, "server_log_path", return_value=tmp_path / "logs" / "server.log"
        ),
        patch(
            "free_claude_code.cli.launchers.common.subprocess.Popen",
            side_effect=OSError("disk full"),
        ) as popen,
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is False

    popen.assert_called_once()
    lock.release.assert_called_once()


def test_ensure_server_running_fails_when_proxy_never_becomes_ready(
    tmp_path: Path,
) -> None:
    lock = MagicMock()
    lock.acquire.return_value = True

    with (
        patch.object(common, "preflight_proxy", return_value="connection refused"),
        patch.object(common, "InterprocessFileLock", return_value=lock),
        patch.object(common, "config_dir_path", return_value=tmp_path),
        patch.object(
            common, "server_log_path", return_value=tmp_path / "logs" / "server.log"
        ),
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
        patch.object(common, "_SERVER_START_TIMEOUT_SECONDS", 0.05),
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is False

    popen.assert_called_once()
    assert popen.call_args.args[0] == _SERVER_COMMAND
    lock.release.assert_called_once()


def test_ensure_server_running_loser_gives_up_when_never_ready() -> None:
    lock = MagicMock()
    lock.acquire.return_value = False

    with (
        patch.object(common, "preflight_proxy", return_value="connection refused"),
        patch.object(common, "InterprocessFileLock", return_value=lock),
        patch("free_claude_code.cli.launchers.common.subprocess.Popen") as popen,
        patch.object(common.time, "sleep"),
        patch.object(common, "_SERVER_START_TIMEOUT_SECONDS", 0.05),
    ):
        assert common._ensure_fcc_server_running(_PROXY_URL) is False

    popen.assert_not_called()
    lock.release.assert_not_called()
