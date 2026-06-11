"""Hugging Face Inference Providers implementation (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
from typing import Any

import openai
from loguru import logger

from providers.base import ProviderConfig
from providers.defaults import HUGGINGFACE_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body, clone_body_with_clamped_max_tokens


class HuggingFaceProvider(OpenAIChatTransport):
    """Hugging Face router using ``https://router.huggingface.co/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="HUGGINGFACE",
            base_url=config.base_url or HUGGINGFACE_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with ``max_tokens`` clamped when the router rejects the budget.

        Claude clients request large output budgets; the router enforces a
        per-model ``max_tokens`` ceiling bounded by the model's context window
        and rejects larger values with a 400 that names the limit.
        """
        status_code = getattr(error, "status_code", None)
        is_bad_request = isinstance(error, openai.BadRequestError) or status_code == 400
        if not is_bad_request:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"

        retry_body = clone_body_with_clamped_max_tokens(error_text, body)
        if retry_body is not None:
            logger.warning(
                "HUGGINGFACE_STREAM: retrying with max_tokens clamped to {} after 400 error",
                retry_body["max_tokens"],
            )
            return retry_body

        # Malformed tool-call generations ("tool_use_failed") are usually
        # transient sampling flukes; one resample often succeeds.
        if "tool_use_failed" in error_text:
            logger.warning(
                "HUGGINGFACE_STREAM: retrying once after tool_use_failed 400 error"
            )
            return dict(body)

        return None
