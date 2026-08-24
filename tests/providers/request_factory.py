"""Canonical request factories for provider tests."""

from free_claude_code.core.anthropic import messages_to_inference_request
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.inference import InferenceRequest

type ProviderTestRequest = MessagesRequest | InferenceRequest


def canonical_request(request: ProviderTestRequest) -> InferenceRequest:
    """Cross the production Messages ingress at a provider-test boundary."""

    if isinstance(request, InferenceRequest):
        return request
    return messages_to_inference_request(request)


def make_messages_request(
    model: str = "test-model", **overrides: object
) -> InferenceRequest:
    """Build canonical provider input through the real Messages ingress."""

    return messages_to_inference_request(make_messages_wire_request(model, **overrides))


def make_messages_wire_request(
    model: str = "test-model", **overrides: object
) -> MessagesRequest:
    """Build a real Messages wire request with provider-test defaults."""

    data: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "system": "System prompt",
        "stop_sequences": None,
        "tools": [],
        "extra_body": {},
        "thinking": {"enabled": True},
    }
    data.update(overrides)
    return MessagesRequest.model_validate(data)
