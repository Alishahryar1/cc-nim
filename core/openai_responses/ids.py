"""Identifier helpers for OpenAI Responses payloads."""

import uuid


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_message_item_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def new_reasoning_item_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
