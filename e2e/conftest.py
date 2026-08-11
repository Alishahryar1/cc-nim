"""Isolated browser-test composition for the local Admin UI."""

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import uvicorn

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.chat import (
    ChatContextEstimate,
    ChatReasoning,
    ChatService,
    ChatSessionDetail,
    ChatSessionPage,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config import env_migrations, paths
from free_claude_code.config.env_migrations import recognized_env_keys
from free_claude_code.config.loader import clear_settings_cache, get_settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.chat_sqlite import SQLiteChatStore
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


class _ModelListingProvider(BaseProvider):
    def __init__(
        self,
        model_infos: frozenset[ProviderModelInfo] = frozenset(),
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__(
            ProviderConfig(
                api_key="browser-test",
                base_url="https://provider.invalid/v1",
                rate_limit=1_000,
                rate_window=1,
                max_concurrency=100,
                http_read_timeout=1.0,
                http_write_timeout=1.0,
                http_connect_timeout=1.0,
                proxy=None,
                log_raw_sse_events=False,
                log_api_error_tracebacks=False,
            )
        )
        self._model_infos = model_infos
        self._error = error
        self._slow_used = False
        self._message_attempts: dict[str, int] = {}

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._error is not None:
            raise self._error
        return self._model_infos

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, response_model
        summary = str(request.system).startswith("Summarize")
        user_content = str(request.messages[-1].content) if request.messages else ""
        attempt = self._message_attempts.get(user_content, 0) + 1
        self._message_attempts[user_content] = attempt
        slow = not self._slow_used and (
            "[slow]" in user_content
            or ("[slow-regenerate]" in user_content and attempt > 1)
            or (summary and "[slow-compaction]" in user_content)
        )
        fragmented = "[fragmented]" in user_content
        self._slow_used = self._slow_used or slow
        failed_regeneration = "[fail-regenerate]" in user_content and attempt > 1
        if summary:
            text = "Earlier details retained."
        elif failed_regeneration:
            text = "Partial replacement"
        else:
            text = "E2E answer"
        frames = [
            format_sse_event(
                "message_start",
                {"type": "message_start", "message": {"content": []}},
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "E2E thought"},
                },
            ),
            format_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]
        if not fragmented:
            frames.append(
                format_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        for frame in frames:
            yield frame
            await asyncio.sleep(0)
        if fragmented:
            for _index in range(2_000):
                yield format_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "abcd"},
                    },
                )
                await asyncio.sleep(0)
        if summary and "[fail-compaction]" in user_content:
            yield format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "summary provider failed",
                    },
                },
            )
            return
        if not summary and ("[fail-turn]" in user_content or failed_regeneration):
            yield format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "E2E provider failed",
                    },
                },
            )
            return
        if slow:
            await asyncio.Event().wait()
        yield format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 1}
        )
        yield format_sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        )
        yield format_sse_event("message_stop", {"type": "message_stop"})

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


@pytest.fixture
def admin_base_url(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[str]:
    """Serve one fully isolated Admin application on an OS-assigned port."""

    config_dir = tmp_path / ".fcc"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for key in recognized_env_keys():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FCC_ENV_FILE", raising=False)
    monkeypatch.setenv("MODEL", "open_router/e2e-default")
    monkeypatch.setenv("OPENROUTER_API_KEY", "e2e-openrouter-key")
    monkeypatch.setenv("GROQ_API_KEY", "e2e-groq-key")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "e2e-cloudflare-token")
    monkeypatch.setenv("MESSAGING_PLATFORM", "none")
    monkeypatch.setenv("VOICE_NOTE_ENABLED", "false")
    monkeypatch.setenv("FCC_OPEN_BROWSER", "false")
    monkeypatch.setenv("PROXY_AUTH_ENABLED", "false")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "e2e-proxy-token")
    for key, value in getattr(request, "param", {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(paths, "config_dir_path", lambda: config_dir)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    clear_settings_cache()

    provider_secret = "CREDENTIAL[unrecognized-format-987654321]"
    providers: dict[str, BaseProvider] = {
        "open_router": _ModelListingProvider(
            frozenset(
                {
                    ProviderModelInfo("vendor/model-a"),
                    ProviderModelInfo(
                        "vendor/model-b",
                        supports_thinking=True,
                        context_window_tokens=100_000,
                        max_output_tokens=20_000,
                    ),
                    ProviderModelInfo(
                        "vendor/small-context",
                        supports_thinking=True,
                        context_window_tokens=8_192,
                        max_output_tokens=4_096,
                    ),
                }
            )
        ),
        "groq": _ModelListingProvider(
            error=RuntimeError(f"Provider rejected credential {provider_secret}")
        ),
    }
    manager = ProviderRuntimeManager(
        get_settings(),
        runtime_factory=lambda snapshot: ProviderRuntime(snapshot, dict(providers)),
    )
    chat_store = SQLiteChatStore(
        config_dir / "chat" / "chat.db",
        config_dir / "chat" / "chat.lock",
    )
    chat = ChatService(manager, chat_store)
    original_estimate = chat.estimate
    original_get_detail = chat.get_detail
    original_get_turn_page = chat.get_turn_page
    original_list_sessions = chat.list_sessions
    original_update_session = chat.update_session
    original_begin_send = chat_store.begin_send
    delayed_estimate_seen = False

    async def estimate_with_delayed_first_result(
        session_id: str,
        *,
        draft: str,
    ) -> ChatContextEstimate:
        nonlocal delayed_estimate_seen
        delay_result = "[delay-first-estimate]" in draft and not delayed_estimate_seen
        delayed_estimate_seen = delayed_estimate_seen or delay_result
        try:
            return await original_estimate(session_id, draft=draft)
        finally:
            if delay_result:
                await asyncio.sleep(0.75)

    monkeypatch.setattr(chat, "estimate", estimate_with_delayed_first_result)

    async def get_detail_with_test_pagination(
        session_id: str,
    ) -> ChatSessionDetail:
        detail = await original_get_detail(session_id)
        if "[delay-detail]" in detail.session.title:
            await asyncio.sleep(0.75)
        if "[delay-older-page]" in detail.session.title and len(detail.turns) > 1:
            return replace(
                detail,
                turns=detail.turns[-1:],
                next_before=detail.turns[-1].sequence,
            )
        return detail

    async def get_turn_page_with_delayed_result(
        session_id: str,
        *,
        before_sequence: int | None,
        limit: int,
    ):
        result = await original_get_turn_page(
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )
        if before_sequence is not None:
            session = await chat.get_session(session_id)
            if "[delay-older-page]" in session.title:
                await asyncio.sleep(0.75)
        return result

    async def list_sessions_with_delayed_result(
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> ChatSessionPage:
        result = await original_list_sessions(query=query, cursor=cursor, limit=limit)
        if query == "race-old":
            await asyncio.sleep(0.75)
        elif cursor is not None:
            await asyncio.sleep(0.5)
        return result

    monkeypatch.setattr(chat, "get_detail", get_detail_with_test_pagination)
    monkeypatch.setattr(chat, "get_turn_page", get_turn_page_with_delayed_result)
    monkeypatch.setattr(chat, "list_sessions", list_sessions_with_delayed_result)

    async def update_session_with_delayed_result(
        session_id: str,
        *,
        expected_revision: int,
        title: str | None = None,
        model: str | None = None,
        reasoning: ChatReasoning | None = None,
    ):
        result = await original_update_session(
            session_id,
            expected_revision=expected_revision,
            title=title,
            model=model,
            reasoning=reasoning,
        )
        if title is not None and "[delay-title-save]" in title:
            await asyncio.sleep(0.75)
        return result

    monkeypatch.setattr(chat, "update_session", update_session_with_delayed_result)

    async def begin_send_with_delayed_ack(*args, **kwargs):
        result = await original_begin_send(*args, **kwargs)
        if kwargs.get("user_text") == "[delay-send-ack] keep one draft":
            await asyncio.sleep(0.75)
        return result

    monkeypatch.setattr(chat_store, "begin_send", begin_send_with_delayed_ack)
    runtime = ApplicationRuntime(manager, transcriber=None, chat_service=chat)
    app = RuntimeASGIApp(
        create_app(
            ApiServices(
                requests=manager,
                admin=runtime,
                tasks=runtime,
                chat=chat,
            )
        ),
        runtime,
    )

    async def local_provider_result(
        provider_id: str,
        base_url: str,
        path: str,
        *,
        api_key: str = "",
        proxy: str = "",
    ) -> dict[str, object]:
        return {
            "provider_id": provider_id,
            "status": "reachable",
            "label": "Reachable",
            "base_url": base_url,
        }

    monkeypatch.setattr(
        "free_claude_code.api.admin_routes._check_local_provider",
        local_provider_result,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="error",
            access_log=False,
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="fcc-admin-playwright",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 5.0
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("Admin browser-test server exited during startup")
            if time.monotonic() >= deadline:
                raise TimeoutError("Admin browser-test server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        listener.close()
        clear_settings_cache()
        if thread.is_alive():
            pytest.fail("Admin browser-test server did not stop")
