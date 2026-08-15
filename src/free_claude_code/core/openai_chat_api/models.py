"""Pydantic models for the inbound OpenAI Chat Completions protocol.

This is the *inbound* wire format FCC serves at ``POST /v1/chat/completions``.
It is deliberately permissive (``extra="allow"``) because the ecosystem that
speaks this protocol -- Instructor, Atomic Agents, LangChain, LlamaIndex,
Aider, Cline, Continue, OpenWebUI -- each attach their own vendor fields.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ChatBase(BaseModel):
    """Pass through vendor extensions rather than rejecting them."""

    model_config = ConfigDict(extra="allow")


class ChatFunctionCall(_ChatBase):
    name: str | None = None
    arguments: str | None = None


class ChatToolCall(_ChatBase):
    id: str | None = None
    type: str | None = "function"
    function: ChatFunctionCall


class ChatMessage(_ChatBase):
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    # ``content`` may be a plain string, a multimodal part array, or null when
    # the assistant turn carries only tool calls.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None
    # de-facto standard emitted by DeepSeek/vLLM/OpenRouter for thinking text
    reasoning_content: str | None = None


class ChatFunctionDef(_ChatBase):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ChatTool(_ChatBase):
    type: str | None = "function"
    function: ChatFunctionDef


class ChatCompletionsRequest(_ChatBase):
    """An inbound OpenAI Chat Completions request."""

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    max_tokens: int | None = None
    # OpenAI's newer name for max_tokens; clients send one or the other.
    max_completion_tokens: int | None = None

    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None

    tools: list[ChatTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    # Instructor / Atomic Agents structured-output path.
    response_format: dict[str, Any] | None = None

    # Accepted and ignored -- FCC routes by model, not by these.
    n: int | None = None
    user: str | None = None
    seed: int | None = None
    logprobs: bool | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    metadata: dict[str, Any] | None = Field(default=None)

    def effective_max_tokens(self, default: int) -> int:
        """Return a positive max_tokens; Anthropic requires one, OpenAI does not."""
        for candidate in (self.max_tokens, self.max_completion_tokens):
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return default

    def stop_sequences(self) -> list[str] | None:
        """Normalise ``stop`` into Anthropic's ``stop_sequences`` list form."""
        if self.stop is None:
            return None
        if isinstance(self.stop, str):
            return [self.stop] if self.stop else None
        sequences = [item for item in self.stop if isinstance(item, str) and item]
        return sequences or None

    def wants_usage_in_stream(self) -> bool:
        """Return whether the client asked for a final usage-bearing chunk."""
        options = self.stream_options
        return bool(isinstance(options, dict) and options.get("include_usage"))
