"""Pydantic models for OpenAI Chat Completions-compatible ingress."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class OpenAIChatCompletionsRequest(BaseModel):
    """Permissive subset of the OpenAI Chat Completions API request shape."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: Any = None
    metadata: dict[str, Any] | None = None
    reasoning_effort: str | None = None

    def wants_usage(self) -> bool:
        """Return whether the client asked for a trailing usage chunk when streaming."""
        options = self.stream_options
        return isinstance(options, dict) and bool(options.get("include_usage"))
