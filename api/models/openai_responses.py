"""Pydantic models for OpenAI Responses-compatible ingress."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class OpenAIResponsesRequest(BaseModel):
    """Permissive subset of the OpenAI Responses API request shape."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    stream: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    previous_response_id: str | None = None
    store: bool | None = None
