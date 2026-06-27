"""AgentRouter provider - Anthropic-compatible native transport."""

from providers.defaults import AGENTROUTER_DEFAULT_BASE

from .client import AgentRouterProvider

__all__ = ["AGENTROUTER_DEFAULT_BASE", "AgentRouterProvider"]
