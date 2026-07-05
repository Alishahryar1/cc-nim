import subprocess
from pathlib import Path

from smoke.lib import child_process
from smoke.lib.child_process import cmd_python_c, run_captured_text


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
