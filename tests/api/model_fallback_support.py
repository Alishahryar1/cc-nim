"""Controlled provider boundary for model-fallback API and product tests."""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import InferenceEvent, InferenceRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from tests.api.support import create_test_app
from tests.inference_support import text_event_stream


def execution_failure(message: str, *, retryable: bool = True) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message=message,
        retryable=retryable,
    )


def text_stream(text: str, *, model: str) -> list[InferenceEvent]:
    return text_event_stream(text, model=model)


class ControlledFallbackProvider:
    def __init__(
        self,
        *,
        failure: ExecutionFailure | None = None,
        chunks_before_failure: tuple[InferenceEvent, ...] = (),
        text: str | None = None,
        preflight_error: InvalidRequestError | None = None,
    ) -> None:
        self._failure = failure
        self._chunks_before_failure = chunks_before_failure
        self._text = text
        self._preflight_error = preflight_error
        self.preflight_models: list[str] = []
        self.stream_models: list[str] = []
        self.response_models: list[str] = []
        self.close_calls = 0

    def preflight_stream(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy,
    ) -> None:
        del request, reasoning
        self.preflight_models.append(provider_model)
        if self._preflight_error is not None:
            raise self._preflight_error

    async def stream_response(
        self,
        request: InferenceRequest,
        input_tokens: int = 0,
        *,
        provider_model: str,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[InferenceEvent]:
        del input_tokens, request_id, reasoning
        self.stream_models.append(provider_model)
        public_model = response_model or request.model
        self.response_models.append(public_model)
        try:
            for chunk in self._chunks_before_failure:
                yield chunk
            if self._failure is not None:
                raise self._failure
            assert self._text is not None
            for chunk in text_stream(self._text, model=public_model):
                yield chunk
        finally:
            self.close_calls += 1


@contextmanager
def fallback_client(
    primary: ControlledFallbackProvider,
    fallback: ControlledFallbackProvider,
) -> Iterator[TestClient]:
    app = create_test_app(
        Settings(
            model="nvidia_nim/primary-model",
            model_fallbacks=("groq/fallback-model",),
        )
    )

    def resolve(
        provider_id: str,
        *,
        lease: object,
    ) -> ControlledFallbackProvider:
        del lease
        return {"nvidia_nim": primary, "groq": fallback}[provider_id]

    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            side_effect=resolve,
        ),
        TestClient(app) as client,
    ):
        yield client


def messages_payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "nvidia_nim/primary-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": stream,
    }


def responses_payload() -> dict[str, object]:
    return {
        "model": "nvidia_nim/primary-model",
        "input": "Hello",
        "max_output_tokens": 32,
        "stream": True,
    }
