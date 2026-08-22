"""Pydantic models for the OpenAI Chat Completions protocol."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ChatCompletionsMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None


class ChatCompletionsToolFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ChatCompletionsTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    function: ChatCompletionsToolFunction


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatCompletionsMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    stop: str | list[str] | None = None
    tools: list[ChatCompletionsTool] | None = None
    tool_choice: Any = None
    user: str | None = None
