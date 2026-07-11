"""Provider pricing catalog + cost estimation.

Prices are per **1M tokens** in USD, sourced from vendor public pages
(2026-06). Zero cost is the safe fallback for anything unknown (avoids
overstating spend). Prefixed model ids (``provider/model``) win over the
bare model id when both are present.

Kept intentionally small — this is a display helper, not an accounting
system. Users can override an entry by editing this file; no runtime
mutation surface is exposed.
"""

from __future__ import annotations

from typing import Any

# (input_per_million_usd, output_per_million_usd)
Price = tuple[float, float]

# Local providers always cost 0.
_FREE_PROVIDERS = frozenset({"lmstudio", "llamacpp", "ollama"})

# Public rate cards, USD per 1M tokens. Bare model id and prefixed forms
# both accepted; router looks up prefixed first.
_MODEL_PRICES: dict[str, Price] = {
    # OpenAI
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/o1-mini": (1.10, 4.40),
    "openai/o3-mini": (1.10, 4.40),
    # Anthropic-native via other clouds — reference only.
    "deepseek/deepseek-chat": (0.14, 0.28),
    "deepseek/deepseek-reasoner": (0.55, 2.19),
    # Groq / Cerebras — fast + cheap.
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "cerebras/llama3.1-8b": (0.10, 0.10),
    "cerebras/gpt-oss-120b": (0.15, 0.60),
    # Gemini via Google AI Studio (free tier caveats apply).
    "gemini/gemini-2.5-flash": (0.075, 0.30),
    "gemini/gemini-3.1-flash-lite": (0.05, 0.20),
    # Mistral
    "mistral/mistral-small-latest": (0.20, 0.60),
    "mistral/devstral-small-latest": (0.20, 0.60),
    "mistral_codestral/codestral-latest": (0.30, 0.90),
    # Kimi — best-effort public rate.
    "kimi/kimi-k2.5": (0.15, 2.50),
    # Fireworks — depends on model tier; conservative default.
    "fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct": (0.90, 0.90),
}


def estimate_cost_usd(
    provider_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return a best-effort USD cost estimate. Zero when the model is unknown."""
    if provider_id in _FREE_PROVIDERS:
        return 0.0
    price = _lookup_price(provider_id, model)
    if price is None:
        return 0.0
    in_price, out_price = price
    return round(
        (input_tokens * in_price + output_tokens * out_price) / 1_000_000,
        6,
    )


def _lookup_price(provider_id: str, model: str) -> Price | None:
    prefixed = f"{provider_id}/{model}"
    if prefixed in _MODEL_PRICES:
        return _MODEL_PRICES[prefixed]
    return _MODEL_PRICES.get(model)


def known_model_count() -> int:
    """Sanity-check helper: how many entries the catalog covers."""
    return len(_MODEL_PRICES)


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a metrics-snapshot list into cost totals per provider."""
    total_usd = 0.0
    per_provider: dict[str, float] = {}
    for entry in entries:
        cost = estimate_cost_usd(
            entry.get("provider_id", ""),
            entry.get("model", ""),
            int(entry.get("input_tokens") or 0),
            int(entry.get("output_tokens") or 0),
        )
        total_usd += cost
        pid = entry.get("provider_id", "")
        per_provider[pid] = round(per_provider.get(pid, 0.0) + cost, 6)
    return {
        "total_usd": round(total_usd, 6),
        "per_provider_usd": per_provider,
        "known_models": known_model_count(),
    }
