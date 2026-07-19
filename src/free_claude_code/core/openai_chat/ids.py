"""Identifier helpers for OpenAI Chat Completions payloads."""

import uuid


def new_chat_completion_id() -> str:
    """Return an OpenAI-style ``chatcmpl-`` identifier."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def new_tool_call_id() -> str:
    """Return an OpenAI-style ``call_`` tool-call identifier."""
    return f"call_{uuid.uuid4().hex[:24]}"
