"""MiniMax-specific multimodal clients."""

from .speech import (
    MINIMAX_TTS_ENDPOINTS,
    MiniMaxSpeechClient,
    MiniMaxSpeechError,
    MiniMaxSpeechRequest,
)

__all__ = [
    "MINIMAX_TTS_ENDPOINTS",
    "MiniMaxSpeechClient",
    "MiniMaxSpeechError",
    "MiniMaxSpeechRequest",
]
