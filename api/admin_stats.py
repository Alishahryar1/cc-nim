"""In-memory token usage statistics for admin dashboard."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class TokenStats:
    """Track token usage per model in memory."""

    def __init__(self) -> None:
        self._model_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )
        self._provider_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )

    def record_usage(
        self,
        model: str,
        provider_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage for a request."""
        self._model_usage[model]["input_tokens"] += input_tokens
        self._model_usage[model]["output_tokens"] += output_tokens
        self._model_usage[model]["requests"] += 1

        self._provider_usage[provider_id]["input_tokens"] += input_tokens
        self._provider_usage[provider_id]["output_tokens"] += output_tokens
        self._provider_usage[provider_id]["requests"] += 1

    def get_model_stats(self) -> list[dict[str, Any]]:
        """Return token stats per model, sorted by total tokens."""
        result = []
        for model, stats in self._model_usage.items():
            result.append({
                "model": model,
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "total_tokens": stats["input_tokens"] + stats["output_tokens"],
                "requests": stats["requests"],
            })
        return sorted(result, key=lambda x: x["total_tokens"], reverse=True)

    def get_provider_stats(self) -> list[dict[str, Any]]:
        """Return token stats per provider."""
        result = []
        for provider_id, stats in self._provider_usage.items():
            result.append({
                "provider_id": provider_id,
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "total_tokens": stats["input_tokens"] + stats["output_tokens"],
                "requests": stats["requests"],
            })
        return sorted(result, key=lambda x: x["total_tokens"], reverse=True)

    def reset(self) -> None:
        """Clear all statistics."""
        self._model_usage.clear()
        self._provider_usage.clear()


# Global instance
token_stats = TokenStats()