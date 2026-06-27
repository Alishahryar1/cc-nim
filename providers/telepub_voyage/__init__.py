"""Telepub Voyage provider exports."""

from providers.defaults import TELEPUB_VOYAGE_DEFAULT_BASE

from .client import TelepubVoyageProvider

__all__ = [
    "TELEPUB_VOYAGE_DEFAULT_BASE",
    "TelepubVoyageProvider",
]
