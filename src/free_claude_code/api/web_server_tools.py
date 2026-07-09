"""Compatibility re-exports for :mod:`api.web_tools` (web_search / web_fetch)."""

import httpx

from free_claude_code.api.web_tools.egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
)
from free_claude_code.api.web_tools.request import is_web_server_tool_request
from free_claude_code.api.web_tools.streaming import stream_web_server_tool_response

__all__ = [
    "WebFetchEgressPolicy",
    "WebFetchEgressViolation",
    "enforce_web_fetch_egress",
    "httpx",
    "is_web_server_tool_request",
    "stream_web_server_tool_response",
]
