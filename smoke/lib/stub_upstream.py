"""Local OpenAI-compatible upstreams for deterministic provider-failure smokes.

Live provider quotas cannot be exhausted on demand, so failover scenarios point
a local provider id at one of these servers instead. They speak just enough of
the OpenAI chat-completions contract for FCC's provider adapters: a model list
for discovery warmup, and either a streaming completion or a chosen error.
"""

import json
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(slots=True)
class StubUpstream:
    """One running stub and the chat requests it received."""

    base_url: str
    model_id: str
    chat_requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat_request_count(self) -> int:
        return len(self.chat_requests)


def _completion_chunk(model_id: str, delta: dict[str, Any], finish: str | None) -> str:
    payload = {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _stream_body(model_id: str, text: str) -> bytes:
    return "".join(
        (
            _completion_chunk(model_id, {"role": "assistant", "content": text}, None),
            _completion_chunk(model_id, {}, "stop"),
            "data: [DONE]\n\n",
        )
    ).encode("utf-8")


def _handler(
    stub: StubUpstream,
    *,
    status: int,
    retry_after: str | None,
    reply: str,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Keep pytest output free of per-request access logs."""

        def do_GET(self) -> None:
            if self.path.rstrip("/").endswith("models"):
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": stub.model_id, "object": "model"}],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "unsupported stub route"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if not self.path.rstrip("/").endswith("chat/completions"):
                self._send_json(404, {"error": {"message": "unsupported stub route"}})
                return

            stub.chat_requests.append(json.loads(raw or b"{}"))
            if status != 200:
                self._send_json(
                    status,
                    {
                        "error": {
                            "message": "Rate limit reached for stub upstream.",
                            "type": "rate_limit_exceeded",
                            "code": "rate_limit_exceeded",
                        }
                    },
                    extra_headers=(
                        {} if retry_after is None else {"retry-after": retry_after}
                    ),
                )
                return

            body = _stream_body(stub.model_id, reply)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            code: int,
            payload: dict[str, Any],
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

    return Handler


@contextmanager
def stub_upstream(
    *,
    model_id: str,
    status: int = 200,
    retry_after: str | None = None,
    reply: str = "stub reply",
) -> Iterator[StubUpstream]:
    """Serve one OpenAI-compatible upstream on a free localhost port."""
    stub = StubUpstream(base_url="", model_id=model_id)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(stub, status=status, retry_after=retry_after, reply=reply),
    )
    stub.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def failing_primary_and_healthy_backup(
    *,
    retry_after: str,
    reply: str,
) -> Iterator[tuple[StubUpstream, StubUpstream]]:
    """Serve a quota-exhausted primary and a healthy backup upstream."""
    with ExitStack() as stack:
        primary = stack.enter_context(
            stub_upstream(
                model_id="stub-primary",
                status=429,
                retry_after=retry_after,
            )
        )
        backup = stack.enter_context(stub_upstream(model_id="stub-backup", reply=reply))
        yield primary, backup
