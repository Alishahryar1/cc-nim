"""Providers package - implement your own provider by extending BaseProvider.

Concrete adapters (e.g. ``NvidiaNimProvider``) live in subpackages; import them
from ``providers.nvidia_nim`` etc. to avoid loading every adapter when the
``providers`` package is imported.
"""

from .base import BaseProvider, ProviderConfig

__all__ = [
    "BaseProvider",
    "ProviderConfig",
]
