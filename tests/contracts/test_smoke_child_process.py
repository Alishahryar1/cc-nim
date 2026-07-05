import subprocess
from pathlib import Path

from config.settings import Settings
from smoke.lib import child_process
from smoke.lib import server as smoke_server
from smoke.lib.child_process import (
    cmd_free_claude_code_serve,
    cmd_python_c,
    run_captured_text,
)
from smoke.lib.config import SmokeConfig


def test_free_claude_code_serve_command_uses_cli_entrypoint() -> None:
    assert cmd_free_claude_code_serve() == [
        child_process.python_exe(),
        "-c",
        "from cli.entrypoints import serve; serve()",
    ]


def test_start_server_disables_cli_admin_browser(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            captured["command"] = command
            captured.update(kwargs)

        def poll(self) -> int | None:
            return None

    config = SmokeConfig(
        root=tmp_path,
        results_dir=tmp_path / "results",
        live=True,
        interactive=False,
        targets=frozenset(),
        provider_matrix=frozenset(),
        timeout_s=1.0,
        prompt="",
        claude_bin="claude",
        worker_id="test",
        settings=Settings(),
    )

    monkeypatch.setattr(smoke_server, "find_free_port", lambda: 4567)
    monkeypatch.setattr(smoke_server.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        smoke_server, "_wait_for_health", lambda _server, *, timeout_s: None
    )
    monkeypatch.setattr(smoke_server, "_stop_process", lambda _process: None)

    with smoke_server.start_server(config):
        pass

    env_obj = captured["env"]
    assert isinstance(env_obj, dict)
    env = {str(key): value for key, value in env_obj.items()}
    assert env["FCC_OPEN_BROWSER"] == "0"
    assert env["HOST"] == "127.0.0.1"
    assert env["PORT"] == "4567"


def test_run_captured_text_uses_utf8_replacement(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls["command"] = command
        calls.update(kwargs)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(child_process.subprocess, "run", fake_run)

    result = run_captured_text(
        ("cmd", "arg"),
        cwd=tmp_path,
        env={"FCC_TEST": "1"},
        timeout=1.0,
    )

    assert result.stdout == "ok"
    assert calls["command"] == ["cmd", "arg"]
    assert calls["cwd"] == tmp_path
    assert calls["env"] == {"FCC_TEST": "1"}
    assert calls["capture_output"] is True
    assert calls["text"] is True
    assert calls["encoding"] == "utf-8"
    assert calls["errors"] == "replace"
    assert calls["timeout"] == 1.0
    assert calls["check"] is False


def test_run_captured_text_replaces_invalid_utf8_bytes(tmp_path: Path) -> None:
    result = run_captured_text(
        cmd_python_c(
            "import sys; "
            "sys.stdout.buffer.write(bytes([0x8f])); "
            "sys.stderr.buffer.write(bytes([0x8f]))"
        ),
        cwd=tmp_path,
        timeout=10.0,
    )

    assert result.returncode == 0
    assert result.stdout == "\ufffd"
    assert result.stderr == "\ufffd"
