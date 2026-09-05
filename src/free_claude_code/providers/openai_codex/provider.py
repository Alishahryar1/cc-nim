"""ChatGPT Codex backend provider using OpenAI Responses."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
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
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.model_listing import (
    optional_input_modalities,
    optional_positive_int,
)
from free_claude_code.providers.openai_responses.execution import run_responses_stream
from free_claude_code.providers.openai_responses.presentation import (
    MessagesResponsesPresenter,
    NativeResponsesPresenter,
    ResponsesPresenterFactory,
)

from .auth import OpenAIAuthManager
from .login import OPENAI_CODEX_ORIGINATOR
from .transport import CodexResponsesBackend, auth_headers, response_status_error

try:
    FCC_VERSION = version("free-claude-code")
except PackageNotFoundError:
    FCC_VERSION = "dev"


class OpenAICodexProvider(BaseProvider):
    """Use a ChatGPT subscription through OpenAI's Codex backend."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: OpenAIAuthManager,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._admission = admission
        self._client_headers = {
            "User-Agent": f"{OPENAI_CODEX_ORIGINATOR}/{FCC_VERSION}",
            "originator": OPENAI_CODEX_ORIGINATOR,
            "version": FCC_VERSION,
        }
        self._client = client or httpx.AsyncClient(
            base_url=f"{config.base_url.rstrip('/')}/",
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            headers=self._client_headers,
        )
        self._owns_client = client is None

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate and adapt the private Codex request before upstream I/O."""

        self._build_body(request, reasoning=reasoning)

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate one native Responses request before upstream I/O."""
        self._build_native_body(request, reasoning=reasoning)

    async def cleanup(self) -> None:
        """Close only provider-owned transport resources."""

        if self._owns_client:
            await self._client.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Discover models visible to the currently connected ChatGPT account."""
        return _model_infos(await self._list_models_payload())

    async def _list_models_payload(self) -> Any:
        """Fetch one Codex model catalog with each provider GET admitted once."""
        execution = self._admission.start_execution()
        authentication_recovered = False
        while execution.can_attempt:
            scope: ProviderAttemptScope | None = None
            try:
                access = await self._auth.access()
                attempt = await execution.open_attempt(
                    ProviderOperationKind.MODEL_DISCOVERY
                )
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name="OpenAI",
                    request_id=execution.request_id,
                )
                response = scope.retain(
                    await self._client.get(
                        "models",
                        params={"client_version": FCC_VERSION},
                        headers={**self._client_headers, **auth_headers(access)},
                    )
                )
                if response.status_code == 401 and not authentication_recovered:
                    error = await response_status_error(response)
                    correction = await scope.attempt.correct(error)
                    closing_scope = scope
                    scope = None
                    await closing_scope.aclose(active_error=error)
                    if correction is ProviderCorrectionAction.FINAL:
                        raise error
                    await self._auth.recover_unauthorized(access.access_token)
                    authentication_recovered = True
                    continue
                if not response.is_success:
                    raise await response_status_error(response)
                payload = response.json()
                await scope.attempt.accept()
                execution.succeed()
                return payload
            except asyncio.CancelledError:
                execution.abandon()
                raise
            except Exception as error:
                if scope is not None and not scope.attempt.accepted:
                    decision = await scope.attempt.fail(error)
                    if decision.retry_allowed:
                        continue
                execution.fail(error)
                raise
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        if execution.last_failure is not None:
            raise execution.last_failure
        execution.abandon()
        raise RuntimeError("OpenAI model discovery ended without an attempt outcome")

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream Responses output in Anthropic Messages format."""

        tool_names = OpenAIToolNameCodec.from_request(request)
        body = self._build_body(request, reasoning=reasoning)
        message_id = f"msg_{uuid.uuid4()}"
        return self._run_stream(
            body,
            request_id=request_id,
            response_model=response_model or request.model,
            presenter_factory=lambda: MessagesResponsesPresenter(
                ResponsesProviderStream(
                    message_id=message_id,
                    model=response_model or request.model,
                    input_tokens=input_tokens,
                    log_raw_events=self._config.log_raw_sse_events,
                    tool_names=tool_names,
                )
            ),
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Relay the private Codex Responses stream as native Responses SSE."""
        del input_tokens
        body = self._build_native_body(request, reasoning=reasoning)
        public_model = response_model or request.model
        return self._run_stream(
            body,
            request_id=request_id,
            response_model=public_model,
            presenter_factory=lambda: NativeResponsesPresenter(
                public_model=public_model
            ),
        )

    @staticmethod
    def _build_body(
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        try:
            body = build_responses_provider_request(request, reasoning=reasoning)
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        # The private Codex backend rejects these public Responses fields.
        # Codex itself omits the output cap and uses separate internal metadata.
        body.pop("max_output_tokens", None)
        body.pop("metadata", None)
        return body

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
        body = build_native_responses_request(
            request,
            model=request.model,
            reasoning=reasoning,
        )
        body.pop("max_output_tokens", None)
        body.pop("metadata", None)
        return body

    def _run_stream(
        self,
        body: JsonObject,
        *,
        request_id: str | None,
        response_model: str,
        presenter_factory: ResponsesPresenterFactory,
    ) -> AsyncIterator[str]:
        return run_responses_stream(
            backend=CodexResponsesBackend(
                client=self._client,
                auth=self._auth,
                body=body,
                client_headers=self._client_headers,
            ),
            admission=self._admission,
            provider_name="OpenAI",
            request_id=request_id,
            response_model=response_model,
            body=body,
            read_timeout_s=self._config.http_read_timeout,
            presenter_factory=presenter_factory,
            log_error_tracebacks=self._config.log_api_error_tracebacks,
        )


def _model_infos(payload: Any) -> frozenset[ProviderModelInfo]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("OpenAI model-list response is missing the models array.")
    infos: set[ProviderModelInfo] = set()
    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("slug")
        visibility = model.get("visibility")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or visibility != "list"
        ):
            continue
        efforts = model.get(
            "supported_reasoning_levels",
            model.get("supported_reasoning_efforts"),
        )
        infos.add(
            ProviderModelInfo(
                model_id=model_id,
                supports_thinking=_supports_reasoning(efforts),
                input_modalities=optional_input_modalities(
                    model.get("input_modalities")
                ),
                context_window_tokens=optional_positive_int(
                    model.get("context_window")
                ),
            )
        )
    if not infos:
        raise ValueError("OpenAI did not advertise any visible models.")
    return frozenset(infos)


def _supports_reasoning(levels: object) -> bool | None:
    if not isinstance(levels, list):
        return None
    for level in levels:
        if not isinstance(level, dict):
            return None
        effort = level.get("effort")
        if not isinstance(effort, str) or not effort.strip():
            return None
    return bool(levels)
