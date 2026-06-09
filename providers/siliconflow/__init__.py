"""SiliconFlow (OpenAI-compat) adapter."""

from providers.defaults import SILICONFLOW_DEFAULT_BASE

from .client import SiliconFlowProvider

__all__ = ["SILICONFLOW_DEFAULT_BASE", "SiliconFlowProvider"]
