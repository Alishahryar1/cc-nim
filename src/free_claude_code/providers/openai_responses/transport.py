"""Responses request conversion and SDK backend composition."""

import uuid
from collections.abc import AsyncIterator, Callable
from typing import cast

import httpx2
from openai import AsyncOpenAI
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesProviderStream,
    build_native_responses_request,
    build_responses_provider_request,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
)
from free_claude_code.providers.endpoint import EndpointContext, RequestEndpoint

from .events import ResponsesEventAdapter
from .execution import run_responses_stream
from .presentation import (
    MessagesResponsesPresenter,
    NativeResponsesPresenter,
    ResponsesPresenterFactory,
)
from .sdk import SDKResponsesBackend


class OpenAIResponsesTransport:
    """Compose Responses requests with the shared generation lifecycle."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        admission: ProviderAdmissionController,
        provider_name: str,
        read_timeout_s: float,
        log_raw_sse_events: bool,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
        event_adapter_factory: Callable[[], ResponsesEventAdapter] | None = None,
    ) -> None:
        self._client = client
        self._endpoint_transport = endpoint_transport
        self._event_adapter_factory = event_adapter_factory
        self._admission = admission
        self._provider_name = provider_name
        self._read_timeout_s = read_timeout_s
        self._log_raw_sse_events = log_raw_sse_events

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._build_messages_body(request, reasoning=reasoning)

    def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
    ) -> AsyncIterator[str]:
        body = self._build_messages_body(request, reasoning=reasoning)
        tool_names = OpenAIToolNameCodec.from_request(request)
        message_id = f"msg_{uuid.uuid4()}"
        return self._run_stream(
            body,
            endpoint_context=endpoint_context,
            request_id=request_id,
            response_model=response_model,
            presenter_factory=lambda: MessagesResponsesPresenter(
                ResponsesProviderStream(
                    message_id=message_id,
                    model=response_model,
                    input_tokens=input_tokens,
                    tool_names=tool_names,
                    log_raw_events=self._log_raw_sse_events,
                )
            ),
        )

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._build_native_body(request, reasoning=reasoning)

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
        endpoint_context: EndpointContext | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens
        body = self._build_native_body(request, reasoning=reasoning)
        return self._run_stream(
            body,
            endpoint_context=endpoint_context,
            request_id=request_id,
            response_model=response_model,
            presenter_factory=lambda: NativeResponsesPresenter(
                public_model=response_model
            ),
        )

    @staticmethod
    def _build_messages_body(
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> JsonObject:
        try:
            return cast(
                JsonObject,
                cast(
                    ResponseCreateParamsStreaming,
                    build_responses_provider_request(request, reasoning=reasoning),
                ),
            )
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

    @staticmethod
    def _build_native_body(
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> JsonObject:
        if not request.model.strip():
            raise InvalidRequestError("Responses request model must not be empty.")
        if request.input is None or request.input == "" or request.input == []:
            raise InvalidRequestError("Responses request input must not be empty.")
        return build_native_responses_request(
            request,
            model=request.model,
            reasoning=reasoning,
        )

    def _run_stream(
        self,
        body: JsonObject,
        *,
        request_id: str | None,
        response_model: str,
        presenter_factory: ResponsesPresenterFactory,
        endpoint_context: EndpointContext | None = None,
    ) -> AsyncIterator[str]:
        return run_responses_stream(
            backend=SDKResponsesBackend(
                client=self._client,
                body=body,
                endpoint=RequestEndpoint(endpoint_context, self._endpoint_transport)
                if endpoint_context is not None
                else None,
                event_adapter_factory=self._event_adapter_factory,
            ),
            admission=self._admission,
            provider_name=self._provider_name,
            request_id=request_id,
            response_model=response_model,
            body=body,
            read_timeout_s=self._read_timeout_s,
            presenter_factory=presenter_factory,
        )
