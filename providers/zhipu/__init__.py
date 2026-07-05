"""Zhipu AI (GLM) provider exports."""

from providers.defaults import ZHIPU_DEFAULT_BASE

from .client import ZhipuProvider

__all__ = [
    "ZHIPU_DEFAULT_BASE",
    "ZhipuProvider",
]
