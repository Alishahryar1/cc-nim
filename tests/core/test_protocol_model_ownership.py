"""Protocol models live with the protocol logic that consumes them."""

import subprocess
import sys
from pathlib import Path

from free_claude_code.core.anthropic import (
    MessagesRequest as PublicMessagesRequest,
)
from free_claude_code.core.anthropic import (
    MessagesResponse,
    TokenCountResponse,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest as PublicOpenAIResponsesRequest,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest


def test_anthropic_request_model_is_core_owned_and_permissive() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "provider-model",
            "messages": [{"role": "user", "content": "hello"}],
            "provider_extension": {"enabled": True},
        }
    )

    assert MessagesRequest.__module__ == "free_claude_code.core.anthropic.models"
    assert PublicMessagesRequest is MessagesRequest
    assert request.model_extra == {"provider_extension": {"enabled": True}}


def test_responses_request_model_is_core_owned_and_permissive() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "provider-model",
            "input": "hello",
            "provider_extension": {"enabled": True},
        }
    )

    assert (
        OpenAIResponsesRequest.__module__
        == "free_claude_code.core.openai_responses.models"
    )
    assert PublicOpenAIResponsesRequest is OpenAIResponsesRequest
    assert request.model_extra == {"provider_extension": {"enabled": True}}


def test_anthropic_response_models_are_protocol_owned() -> None:
    assert MessagesResponse.__module__ == "free_claude_code.core.anthropic.models"
    assert TokenCountResponse.__module__ == "free_claude_code.core.anthropic.models"


def test_api_adapter_does_not_own_protocol_models() -> None:
    api_models = Path(__file__).resolve().parents[2] / "src/free_claude_code/api/models"

    assert not (api_models / "__init__.py").exists()
    assert not (api_models / "anthropic.py").exists()
    assert not (api_models / "openai_responses.py").exists()
    assert not (api_models / "responses.py").exists()


def test_protocol_models_and_transport_consumers_are_import_order_independent() -> None:
    import_orders = (
        (
            "free_claude_code.core.trace",
            "free_claude_code.core.anthropic",
            "free_claude_code.providers.base",
            "free_claude_code.api.routes",
        ),
        ("free_claude_code.cli.managed.session",),
        (
            "free_claude_code.core.anthropic.models",
            "free_claude_code.providers.base",
            "free_claude_code.providers.transports.openai_chat.transport",
            "free_claude_code.providers.transports.anthropic_messages.transport",
            "free_claude_code.api.routes",
        ),
        (
            "free_claude_code.api.routes",
            "free_claude_code.providers.transports.anthropic_messages.transport",
            "free_claude_code.providers.transports.openai_chat.transport",
            "free_claude_code.providers.base",
            "free_claude_code.core.anthropic.models",
        ),
    )

    for modules in import_orders:
        script = "; ".join(f"import {module}" for module in modules)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
