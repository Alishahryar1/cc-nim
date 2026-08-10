from free_claude_code.application.model_fallback import (
    build_fallback_chain,
    eligible_candidate_refs,
    is_chat_model,
    is_free_candidate,
    rank_potency,
)
from free_claude_code.application.model_metadata import ProviderModelInfo


def _info(model_id: str) -> ProviderModelInfo:
    return ProviderModelInfo(model_id)


def test_eligible_candidate_refs_excludes_paid_and_non_chat_ordered_by_potency():
    refs = eligible_candidate_refs(
        [
            _info("open_router/small-model:free"),
            _info("open_router/opus-tier-model:free"),
            _info("open_router/anthropic/claude-opus-4.8"),
            _info("open_router/whisper-large:free"),
            _info("gemini/gemini-3.1-flash-lite"),
        ]
    )

    assert "open_router/anthropic/claude-opus-4.8" not in refs
    assert "open_router/whisper-large:free" not in refs
    assert refs.index("open_router/opus-tier-model:free") < refs.index(
        "open_router/small-model:free"
    )
    assert "gemini/gemini-3.1-flash-lite" in refs


def test_build_fallback_chain_always_keeps_primary_first():
    chain = build_fallback_chain(
        "gemini/gemini-3.1-flash-lite",
        [_info("gemini/gemini-3.1-flash-lite"), _info("open_router/some/model:free")],
    )

    assert chain[0] == "gemini/gemini-3.1-flash-lite"
    assert chain.count("gemini/gemini-3.1-flash-lite") == 1


def test_build_fallback_chain_orders_remaining_candidates_by_potency():
    chain = build_fallback_chain(
        "open_router/primary:free",
        [
            _info("open_router/small-model:free"),
            _info("open_router/opus-tier-model:free"),
            _info("open_router/sonnet-tier-model:free"),
        ],
    )

    assert chain == [
        "open_router/primary:free",
        "open_router/opus-tier-model:free",
        "open_router/sonnet-tier-model:free",
        "open_router/small-model:free",
    ]


def test_build_fallback_chain_excludes_non_chat_models():
    chain = build_fallback_chain(
        "open_router/primary:free",
        [
            _info("open_router/primary:free"),
            _info("open_router/whisper-large:free"),
            _info("open_router/text-embedding-3:free"),
        ],
    )

    assert chain == ["open_router/primary:free"]


def test_build_fallback_chain_excludes_paid_openrouter_models():
    """A fallback the user never configured must never be able to cost money."""
    chain = build_fallback_chain(
        "open_router/openai/gpt-oss-20b:free",
        [
            _info("open_router/openai/gpt-oss-20b:free"),
            _info("open_router/anthropic/claude-opus-4.8"),
            _info("open_router/tencent/hy3:free"),
        ],
    )

    assert chain == [
        "open_router/openai/gpt-oss-20b:free",
        "open_router/tencent/hy3:free",
    ]


def test_build_fallback_chain_keeps_non_openrouter_candidates_regardless_of_suffix():
    chain = build_fallback_chain(
        "open_router/openai/gpt-oss-20b:free",
        [
            _info("open_router/openai/gpt-oss-20b:free"),
            _info("gemini/gemini-3.1-flash-lite"),
            _info("nvidia_nim/nvidia/nemotron-3-super-120b-a12b"),
        ],
    )

    assert set(chain) == {
        "open_router/openai/gpt-oss-20b:free",
        "gemini/gemini-3.1-flash-lite",
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
    }


def test_build_fallback_chain_allows_an_explicitly_configured_paid_primary():
    """Filtering only applies to auto-added candidates, never the user's own choice."""
    chain = build_fallback_chain(
        "open_router/anthropic/claude-opus-4.8",
        [_info("open_router/anthropic/claude-opus-4.8")],
    )

    assert chain == ["open_router/anthropic/claude-opus-4.8"]


def test_is_free_candidate_requires_free_suffix_only_for_openrouter():
    assert is_free_candidate("open_router/some/model:free") is True
    assert is_free_candidate("open_router/anthropic/claude-opus-4.8") is False
    assert is_free_candidate("gemini/gemini-3.1-flash-lite") is True
    assert is_free_candidate("nvidia_nim/nvidia/nemotron") is True


def test_is_free_candidate_excludes_paid_and_low_tpm_gemini_models():
    # Gemini Pro models are paid-only, even with billing enabled.
    assert is_free_candidate("gemini/models/gemini-2.5-pro") is False
    assert is_free_candidate("gemini/models/gemini-3.1-pro-preview") is False
    assert is_free_candidate("gemini/models/gemini-pro-latest") is False
    # Gemma caps input at 16K tokens/minute - too low for real coding requests.
    assert is_free_candidate("gemini/models/gemma-4-31b-it") is False
    assert is_free_candidate("gemini/models/gemma-4-26b-a4b-it") is False
    # Flash/Flash-Lite stay free and usable.
    assert is_free_candidate("gemini/models/gemini-3.5-flash") is True
    assert is_free_candidate("gemini/models/gemini-3.1-flash-lite") is True


def test_gemma_is_excluded_from_derivation_candidates():
    refs = eligible_candidate_refs(
        [
            _info("gemini/models/gemma-4-31b-it"),
            _info("gemini/models/gemini-3.5-flash"),
        ]
    )
    assert refs == ["gemini/models/gemini-3.5-flash"]


def test_is_chat_model_excludes_non_chat_markers():
    assert is_chat_model("open_router/openai/gpt-oss-20b:free") is True
    assert is_chat_model("open_router/openai/whisper-large-v3") is False
    assert is_chat_model("open_router/google/text-embedding-3") is False
    # Orpheus is a text-to-speech model, not a chat model.
    assert is_chat_model("groq/canopylabs/orpheus-v1-english") is False


def test_rank_potency_frontier_outranks_big_open_weight():
    # Similarity to Claude, not raw parameter count: a Claude-like frontier model
    # outranks a large but narrower open-weight one, which still outranks an
    # unknown family.
    frontier = rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-pro")
    open_weight = rank_potency("cerebras/gpt-oss-120b")
    unknown = rank_potency("open_router/some/mystery-model:free")
    assert frontier > open_weight > unknown


def test_rank_potency_ranks_general_qwen3_above_narrow_coder_variant():
    # Opus/Fable are generalist reasoners, so the general Qwen3 build must rank
    # above the narrow coding specialist even though the coder scores high on
    # pure code benchmarks.
    general = rank_potency("open_router/qwen/qwen3-235b:free")
    coder = rank_potency("open_router/qwen/qwen3-coder-480b:free")
    assert general > coder


def test_rank_potency_ranks_thinking_generalist_above_narrow_coder():
    # A narrow coding specialist (codestral) sits below a general frontier model
    # with native thinking, matching the Claude-experience ordering.
    devstral = rank_potency("mistral/devstral-medium-latest")
    deepseek = rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-pro")
    assert deepseek > devstral


def test_rank_potency_demotes_mini_variant_below_full_size_sibling():
    # gpt-5-mini is a known family but must sit below full-size gpt-5.
    assert rank_potency("github_models/openai/gpt-5") > rank_potency(
        "github_models/openai/gpt-5-mini"
    )


def test_rank_potency_classifies_mistral_coding_and_reasoning_families():
    # devstral (coding-tuned) and magistral (reasoning) must rank in the curated
    # coding band, not sink as unknown families by size alone.
    unknown = rank_potency("mistral/ministral-8b-latest")
    assert rank_potency("mistral/devstral-medium-latest") > unknown
    assert rank_potency("mistral/magistral-medium-latest") > unknown


def test_rank_potency_puts_top_free_open_weight_coders_above_prior_gen():
    # Grounded in 2026 coding benchmarks: GLM-4.7, DeepSeek-V4 and Kimi-K2 top
    # the open-weight coding charts, above prior-generation GPT-4o.
    gpt4o = rank_potency("github_models/openai/gpt-4o")
    assert rank_potency("cerebras/zai-glm-4.7") > gpt4o
    assert rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-pro") > gpt4o
    assert rank_potency("nvidia_nim/moonshotai/kimi-k2.6") > gpt4o


def test_rank_potency_prefers_pro_variant_over_flash_within_a_tier():
    # Both match the deepseek-v\d tier; the stronger "pro" build must outrank
    # the reduced "flash" build (the old alphabetical tiebreak got this wrong,
    # so the derivation served deepseek-v4-flash while deepseek-v4-pro existed).
    pro = rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-pro")
    flash = rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-flash")
    v32 = rank_potency("nvidia_nim/deepseek-ai/deepseek-v3.2")
    assert pro > flash
    # A same-family full/base build outranks a newer flash: flash is demoted so
    # derivation prefers full models over reduced builds when pro is missing.
    assert pro > v32 > flash


def test_rank_potency_prefers_next_family_full_over_top_family_flash():
    # When deepseek-v4-pro is unavailable (common on NIM free keys that only
    # expose flash), derivation must try the next family's full model before
    # deepseek-v4-flash - otherwise flash of a top tier starves GLM/Kimi/etc.
    pro = rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-pro")
    glm = rank_potency("cerebras/zai-glm-4.7")
    flash = rank_potency("nvidia_nim/deepseek-ai/deepseek-v4-flash")
    assert pro > glm > flash
    # Gemini flash stays in its own band (not demoted across Gemini tiers).
    assert rank_potency("gemini/gemini-3.5-flash") > rank_potency(
        "gemini/gemini-2.0-flash"
    )


def test_rank_potency_orders_known_families_above_unknown():
    assert rank_potency("open_router/anthropic/claude-opus-4.8") > rank_potency(
        "open_router/some/mystery-model"
    )
    assert rank_potency("open_router/anthropic/claude-sonnet-4-5") > rank_potency(
        "open_router/openai/gpt-oss-20b:free"
    )


def test_rank_potency_orders_pro_above_flash_above_flash_lite():
    pro = rank_potency("gemini/gemini-3.1-pro-preview")
    flash = rank_potency("gemini/gemini-3.5-flash")
    flash_lite = rank_potency("gemini/gemini-3.1-flash-lite")

    assert pro > flash > flash_lite


def test_rank_potency_does_not_mistake_gemini_for_mini():
    # "gemini" contains "mini" as a substring but must not read as a small model.
    assert rank_potency("gemini/gemini-2.5-pro") > rank_potency(
        "gemini/gemini-2.5-flash"
    )


def test_music_and_video_models_are_excluded_as_non_chat():
    assert is_chat_model("gemini/models/lyria-3-pro-preview") is False
    assert is_chat_model("gemini/models/veo-3-generate") is False


def test_rank_potency_breaks_ties_by_version_then_params():
    assert rank_potency("gemini/gemini-3.5-flash") > rank_potency(
        "gemini/gemini-2.0-flash"
    )
    assert rank_potency("open_router/nvidia/nemotron-3-ultra-550b:free") > rank_potency(
        "open_router/nvidia/nemotron-3-super-120b:free"
    )


def test_prioritize_large_context_leaves_normal_requests_untouched():
    from free_claude_code.application.model_fallback import prioritize_large_context

    refs = [
        "nvidia_nim/deepseek-ai/deepseek-v4-pro",
        "cerebras/zai-glm-4.7",
        "gemini/models/gemini-3.5-flash",
    ]
    # Below the threshold the coding order is preserved exactly.
    assert prioritize_large_context(refs, input_tokens=5000) == refs


def test_prioritize_large_context_front_loads_big_window_models_for_giant_requests():
    from free_claude_code.application.model_fallback import prioritize_large_context

    refs = [
        "cerebras/zai-glm-4.7",  # 8K - overflows a compaction request
        "nvidia_nim/deepseek-ai/deepseek-v4-pro",  # 256K
        "github_models/openai/gpt-4.1",  # 128K-ish
        "gemini/models/gemini-3.5-flash",  # 1M, fastest on huge context
    ]
    ordered = prioritize_large_context(refs, input_tokens=190_000)
    # Gemini (1M) leads, then deepseek-v4 (256K); small-window models sink.
    assert ordered[0] == "gemini/models/gemini-3.5-flash"
    assert ordered[1] == "nvidia_nim/deepseek-ai/deepseek-v4-pro"
    assert ordered[-1] in ("cerebras/zai-glm-4.7", "github_models/openai/gpt-4.1")
    # Same set, just reordered.
    assert set(ordered) == set(refs)
