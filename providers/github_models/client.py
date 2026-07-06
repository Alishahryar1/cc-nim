"""GitHub Models provider using OpenAI-compatible chat completions."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from providers.base import ProviderConfig
from providers.defaults import GITHUB_MODELS_DEFAULT_BASE
from providers.exceptions import ModelListResponseError
from providers.model_listing import ProviderModelInfo, model_infos_from_ids
from providers.transports.http import maybe_await_aclose
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

GITHUB_MODELS_CATALOG_URL = "https://models.github.ai/catalog/models"
GITHUB_MODELS_API_VERSION = "2026-03-10"

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="GITHUB_MODELS",
)
_REQUIRED_MODEL_CAPABILITIES = frozenset({"streaming", "tool-calling"})


class GitHubModelsProvider(OpenAIChatTransport):
    """GitHub Models OpenAI-compatible inference provider."""

    def __init__(self, config: ProviderConfig):
        self._catalog_url = GITHUB_MODELS_CATALOG_URL
        self._model_list_client = httpx.AsyncClient(
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )
        super().__init__(
            config,
            provider_name="GITHUB_MODELS",
            base_url=config.base_url or GITHUB_MODELS_DEFAULT_BASE,
            api_key=config.api_key,
            default_headers=_github_models_default_headers(),
        )

    async def cleanup(self) -> None:
        """Release provider client resources."""
        await super().cleanup()
        await self._model_list_client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        """Return GitHub Models ids that support FCC's streaming tool workflow."""
        return frozenset(info.model_id for info in await self.list_model_infos())

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return stream/tool-capable GitHub Models catalog ids."""
        response = await self._model_list_client.get(
            self._catalog_url,
            headers=self._model_list_headers(),
        )
        try:
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise ModelListResponseError(
                    "GITHUB_MODELS model-list response is malformed: invalid JSON"
                ) from exc
            return model_infos_from_ids(
                _extract_supported_github_model_ids(payload),
            )
        finally:
            await maybe_await_aclose(response)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    def _model_list_headers(self) -> dict[str, str]:
        return _github_models_api_headers(self._api_key)


def _github_models_default_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_MODELS_API_VERSION,
    }


def _github_models_api_headers(api_key: str) -> dict[str, str]:
    return {
        **_github_models_default_headers(),
        "Authorization": f"Bearer {api_key}",
    }


def _extract_supported_github_model_ids(payload: Any) -> frozenset[str]:
    """Extract stream/tool-capable model ids from GitHub's catalog array."""
    if not _is_sequence(payload):
        raise ModelListResponseError(
            "GITHUB_MODELS model-list response is malformed: expected top-level array"
        )

    model_ids: set[str] = set()
    for item in payload:
        if not isinstance(item, Mapping):
            raise ModelListResponseError(
                "GITHUB_MODELS model-list response is malformed: expected every item to be an object"
            )
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelListResponseError(
                "GITHUB_MODELS model-list response is malformed: expected every item to include id"
            )
        capabilities = item.get("capabilities")
        if not _supports_streaming_tools(capabilities):
            continue
        model_ids.add(model_id)

    return frozenset(model_ids)


def _supports_streaming_tools(capabilities: Any) -> bool:
    if not _is_sequence(capabilities):
        return False
    capability_names = {item for item in capabilities if isinstance(item, str)}
    return capability_names >= _REQUIRED_MODEL_CAPABILITIES


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
