"""Pydantic models for API responses."""

from typing import List, Dict, Any, Optional, Union, Literal

from pydantic import BaseModel

from .anthropic import ContentBlockText, ContentBlockToolUse, ContentBlockThinking


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[
        Union[
            ContentBlockText, ContentBlockToolUse, ContentBlockThinking, Dict[str, Any]
        ]
    ]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage
