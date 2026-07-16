"""Loguru-based structured logging configuration.

Structured logs are written as JSON lines to a configurable path (default
``logs/server.log``). Stdlib logging is intercepted and funneled to loguru.
Context vars (request_id, node_id, chat_id) from contextualize() are
included at top level for easy grep/filter.
"""

import json
import logging
import re
import threading
from pathlib import Path

from loguru import logger

_configured = False
_current_level = "DEBUG"
_sink_id: int | None = None

# Loguru ``logger.bind()`` key used by structured TRACE payloads; ``core/trace.py``
# uses the identical string constant ``TRACE_PAYLOAD_BINDING``.
_TRACE_PAYLOAD_BINDING = "trace_payload"

# Context keys we promote to top-level JSON for traceability / grep
_CONTEXT_KEYS = (
    "request_id",
    "node_id",
    "chat_id",
    "claude_session_id",
    "http_method",
    "http_path",
)

_TELEGRAM_BOT_RE = re.compile(
    r"(https?://api\.telegram\.org/)bot([0-9]+:[A-Za-z0-9_-]+)(/?)",
    re.IGNORECASE,
)
# Authorization: Bearer <token> (HTTP client / proxy debug lines)
_AUTH_BEARER_RE = re.compile(
    r"(\bAuthorization\s*:\s*Bearer\s+)([^\s'\"]+)",
    re.IGNORECASE,
)


def _redact_sensitive_substrings(message: str) -> str:
    """Remove obvious API tokens and secrets before JSON log line emission."""
    text = _TELEGRAM_BOT_RE.sub(r"\1bot<redacted>\3", message)
    return _AUTH_BEARER_RE.sub(r"\1<redacted>", text)


def _serialize_with_context(record) -> str:
    """Format record as JSON with context vars at top level.
    Returns a format template; we inject _json into record for output.
    """
    extra = record.get("extra", {})
    out = {
        "time": str(record["time"]),
        "level": record["level"].name,
        "message": _redact_sensitive_substrings(str(record["message"])),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    trace_payload = extra.get(_TRACE_PAYLOAD_BINDING)
    for key in _CONTEXT_KEYS:
        if key in extra and extra[key] is not None:
            out[key] = extra[key]
    if isinstance(trace_payload, dict):
        for tk, tv in trace_payload.items():
            if tk in out:
                continue
            out[tk] = tv
        out["trace"] = True
    record["_json"] = json.dumps(out, default=str)
    return "{_json}\n"


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "active", False):
            # Avoid deadlock when nested stdlib records fire during a loguru emit.
            return
        self._local.active = True
        try:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame is not None and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
        finally:
            self._local.active = False


def _add_file_sink(log_file: str | Path, level: str) -> int:
    return logger.add(
        log_file,
        level=level,
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
        rotation="50 MB",
        enqueue=True,
    )


def configure_logging(
    log_file: str | Path,
    *,
    force: bool = False,
    verbose_third_party: bool = False,
    level: str = "DEBUG",
) -> None:
    """Configure loguru with JSON output to log_file and intercept stdlib logging.

    Idempotent: skips if already configured with the same level (e.g. hot reload).
    On level change, replaces only the file sink without truncating the log.
    Use force=True to reconfigure from scratch (e.g. in tests with a different log path).

    When ``verbose_third_party`` is false, noisy HTTP and Telegram loggers are capped
    at WARNING unless explicitly configured otherwise.
    """
    global _configured, _current_level, _sink_id

    if _configured and not force and level == _current_level:
        return

    if not _configured or force:
        # First-time configuration: full setup with stdlib interception
        _configured = True

        logger.remove()

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

        _sink_id = _add_file_sink(log_file, level)

        intercept = InterceptHandler()
        logging.root.handlers = [intercept]
        logging.root.setLevel(logging.DEBUG)

        third_party = (
            "httpx",
            "httpcore",
            "httpcore.http11",
            "httpcore.connection",
            "telegram",
            "telegram.ext",
        )
        for name in third_party:
            logging.getLogger(name).setLevel(
                logging.WARNING if not verbose_third_party else logging.NOTSET
            )
    else:
        # Level changed during supervised restart: replace only the file sink.
        # Don't truncate the log file and don't reconfigure stdlib interception.
        if _sink_id is not None:
            logger.remove(_sink_id)
        _sink_id = _add_file_sink(log_file, level)

    _current_level = level
