from __future__ import annotations

from api import pricing


def test_local_providers_are_free() -> None:
    for provider_id in ("lmstudio", "llamacpp", "ollama"):
        assert (
            pricing.estimate_cost_usd(provider_id, "any-model", 1_000_000, 1_000_000)
            == 0.0
        )


def test_unknown_model_defaults_to_zero() -> None:
    assert pricing.estimate_cost_usd("openai", "no-such-model", 10_000, 10_000) == 0.0


def test_prefixed_lookup_matches_openai_gpt4o() -> None:
    cost = pricing.estimate_cost_usd("openai", "gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.50


def test_bare_model_lookup_fallback() -> None:
    cost = pricing.estimate_cost_usd("someproxy", "gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.50


def test_summarize_totals_across_providers() -> None:
    entries = [
        {
            "provider_id": "openai",
            "model": "gpt-4o",
            "input_tokens": 500_000,
            "output_tokens": 500_000,
        },
        {
            "provider_id": "ollama",
            "model": "llama3.1",
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        },
    ]
    result = pricing.summarize(entries)
    assert result["total_usd"] == 6.25  # 500k * 2.50/M + 500k * 10.00/M
    assert result["per_provider_usd"]["openai"] == 6.25
    assert result["per_provider_usd"]["ollama"] == 0.0
    assert result["known_models"] == pricing.known_model_count()


def test_known_model_count_positive() -> None:
    assert pricing.known_model_count() > 0
