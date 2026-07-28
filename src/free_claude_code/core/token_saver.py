import re
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BANG_PATTERN = re.compile(r"^(!+|ERROR:)", re.IGNORECASE)

_TOKEN_SAVER_MODES = ("none", "low", "medium", "high", "max")

_KIB = 1024
_MODE_CAPS: dict[str, int] = {
    "low": 8 * _KIB,
    "medium": 4 * _KIB,
    "high": 2 * _KIB,
    "max": 1 * _KIB,
}

_SENTINEL = "\n...[token-saver]...\n"


def _cap(mode: str) -> int:
    return _MODE_CAPS.get(mode, 4096)


def _stringify(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        import json

        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _fit_capped(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    half = cap // 2
    if half == 0:
        return text
    candidate = text[:half] + _SENTINEL + text[-half:]
    if len(candidate) > len(text):
        return text
    return candidate


def _fit_max_lines(text: str, cap: int) -> str:
    lines = text.splitlines()
    if len(lines) <= 10:
        return _fit_capped(text, cap)
    keep = 3
    remaining = len(lines) - 2 * keep
    candidate = (
        "\n".join(lines[:keep])
        + f"\n...[token-saver: {remaining} lines elided]...\n"
        + "\n".join(lines[-keep:])
    )
    if len(candidate) > len(text):
        return text
    return candidate


def _fit_bang_guard(text: str, cap: int) -> str:
    stripped = text.strip()
    if _BANG_PATTERN.match(stripped):
        return text
    return _fit_capped(text, cap)


def _fit_todo_compact(text: str) -> str:
    lines = text.splitlines()
    kept = [line.strip() for line in lines if line.startswith("- [")]
    return "\n".join(kept) if kept else text


def _fit_bash_test_strip(text: str, cap: int) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    capture = False
    for line in lines:
        upper = line.upper()
        if any(
            k in upper
            for k in ("FAIL", "ERROR", "TRACEBACK", "EXCEPTION", "FAILED", "ERRORS")
        ):
            capture = True
            kept.append(line)
        elif capture:
            if line.strip() == "" or "====" in line or "----" in line:
                capture = False
            kept.append(line)
        elif "====" in line or "PASSED" in upper:
            pass
        else:
            kept.append(line)

    stripped = "\n".join(kept)
    if len(stripped) > cap:
        return _fit_capped(stripped, cap)
    return stripped


class TokenSaver:
    """RTK-inspired per-tool output saver.

    Applies strategy-selected compress/elide/truncate to tool results
    before they re-enter the conversation context.  Every fit function
    obeys the *never-worse* rule: compressed text is never longer than
    the original.
    """

    _instance: TokenSaver | None = None

    def __init__(self, mode: str = "medium") -> None:
        self._mode = mode if mode in _TOKEN_SAVER_MODES else "medium"

    @classmethod
    def singleton(cls, mode: str = "medium") -> TokenSaver:
        if cls._instance is None:
            cls._instance = cls(mode)
        elif mode in _TOKEN_SAVER_MODES and cls._instance._mode != mode:
            cls._instance._mode = mode
        return cls._instance

    def set_mode(self, mode: str) -> None:
        if mode in _TOKEN_SAVER_MODES:
            self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def save_result(
        self, content: Any, *, tool_name: str = "", is_error: bool = False
    ) -> str:
        if self._mode == "none":
            return _stringify(content)
        text = _stringify(content)
        original = text
        if is_error:
            text = _fit_bang_guard(text, _cap(self._mode))
        elif tool_name == "Bash":
            text = self._fit_bash(text)
        elif tool_name in ("Read", "Glob", "Grep"):
            text = _fit_capped(text, _cap(self._mode) * 2)
        elif tool_name == "Write":
            text = "[Write status preserved — user content elided]"
        elif tool_name in ("WebFetch", "WebSearch"):
            text = _fit_capped(text, _cap(self._mode) * 2)
        elif tool_name == "TodoWrite":
            text = self._fit_todo(text)
        elif tool_name == "Task":
            text = _fit_capped(text, _cap(self._mode) * 2)
        else:
            text = _fit_capped(text, _cap(self._mode) * 2)
        if len(text) > len(original):
            return original
        return text

    def fit(self, text: str) -> str:
        return self.save_result(text, tool_name="")

    def _fit_bash(self, text: str) -> str:
        text = _ANSI_ESCAPE.sub("", text)
        cap = _cap(self._mode)
        if self._mode == "max":
            return _fit_max_lines(text, cap)
        if self._mode == "high":
            return _fit_bash_test_strip(text, cap)
        return _fit_capped(text, cap)

    def _fit_todo(self, text: str) -> str:
        if self._mode in ("high", "max"):
            return _fit_todo_compact(text)
        return _fit_capped(text, _cap(self._mode) * 2)
