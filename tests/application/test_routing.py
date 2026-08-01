from unittest.mock import patch

import pytest

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
    TokenCountRequest,
)
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort


@pytest.fixture
def settings():
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_fable = None
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.model_fable_backup = None
    settings.model_opus_backup = None
    settings.model_sonnet_backup = None
    settings.model_haiku_backup = None
    settings.model_backup = None
    settings.reasoning_policy = ReasoningPreference.CLIENT
    settings.reasoning_fable = ReasoningPreference.INHERIT
    settings.reasoning_opus = ReasoningPreference.INHERIT
    settings.reasoning_sonnet = ReasoningPreference.INHERIT
    settings.reasoning_haiku = ReasoningPreference.INHERIT
    return settings


def test_model_router_resolves_default_model(settings):
    resolved = ModelRouter(settings).resolve("claude-3-opus")

    assert resolved.original_model == "claude-3-opus"
    assert resolved.provider_id == "nvidia_nim"
    assert resolved.provider_model == "fallback-model"
    assert resolved.provider_model_ref == "nvidia_nim/fallback-model"
    assert resolved.reasoning_preference is ReasoningPreference.CLIENT


def test_model_router_applies_opus_override(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"

    request = MessagesRequest(
        model="claude-opus-4-20250514",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_messages_request(request)

    assert routed.request.model == "deepseek/deepseek-r1"
    assert routed.resolved.provider_model_ref == "open_router/deepseek/deepseek-r1"
    assert routed.resolved.original_model == "claude-opus-4-20250514"
    assert routed.reasoning.control is ReasoningControl.DEFAULT
    assert request.model == "claude-opus-4-20250514"


def test_model_router_applies_fable_override(settings):
    settings.model_fable = "open_router/anthropic/claude-fable-5"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-fable-5",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "anthropic/claude-fable-5"
    assert routed.resolved.provider_model_ref == "open_router/anthropic/claude-fable-5"
    assert routed.resolved.original_model == "claude-fable-5"


def test_model_router_resolves_route_reasoning_preferences(settings):
    settings.reasoning_policy = ReasoningPreference.OFF
    settings.reasoning_fable = ReasoningPreference.HIGH
    settings.reasoning_opus = ReasoningPreference.MAX
    settings.reasoning_haiku = ReasoningPreference.OFF

    router = ModelRouter(settings)

    assert (
        router.resolve("claude-fable-5").reasoning_preference
        is ReasoningPreference.HIGH
    )
    assert (
        router.resolve("claude-opus-4-20250514").reasoning_preference
        is ReasoningPreference.MAX
    )
    assert (
        router.resolve("claude-sonnet-4-20250514").reasoning_preference
        is ReasoningPreference.OFF
    )
    assert (
        router.resolve("claude-3-haiku-20240307").reasoning_preference
        is ReasoningPreference.OFF
    )
    assert router.resolve("claude-2.1").reasoning_preference is ReasoningPreference.OFF


def test_model_router_applies_haiku_override(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "qwen2.5-7b"
    assert routed.resolved.provider_model_ref == "lmstudio/qwen2.5-7b"


def test_model_router_applies_sonnet_override(settings):
    settings.model_sonnet = "nvidia_nim/meta/llama-3.3-70b-instruct"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "meta/llama-3.3-70b-instruct"
    assert (
        routed.resolved.provider_model_ref == "nvidia_nim/meta/llama-3.3-70b-instruct"
    )


def test_model_router_routes_prefixed_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="deepseek/deepseek-chat",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-chat"
    assert routed.resolved.original_model == "deepseek/deepseek-chat"
    assert routed.resolved.provider_id == "deepseek"
    assert routed.resolved.provider_model == "deepseek-chat"
    assert routed.resolved.provider_model_ref == "deepseek/deepseek-chat"


def test_model_router_routes_wafer_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="wafer/DeepSeek-V4-Pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "DeepSeek-V4-Pro"
    assert routed.resolved.provider_id == "wafer"
    assert routed.resolved.provider_model == "DeepSeek-V4-Pro"
    assert routed.resolved.provider_model_ref == "wafer/DeepSeek-V4-Pro"


def test_model_router_routes_minimax_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="minimax/MiniMax-M3",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "MiniMax-M3"
    assert routed.resolved.provider_id == "minimax"
    assert routed.resolved.provider_model == "MiniMax-M3"
    assert routed.resolved.provider_model_ref == "minimax/MiniMax-M3"


def test_model_router_routes_gateway_encoded_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.original_model
        == "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert routed.resolved.provider_id == "nvidia_nim"
    assert routed.resolved.provider_model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.provider_model_ref
        == "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )


def test_model_router_routes_no_thinking_gateway_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.original_model
        == "claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert routed.resolved.provider_id == "nvidia_nim"
    assert routed.resolved.provider_model == "deepseek-ai/deepseek-v4-pro"
    assert routed.reasoning.control is ReasoningControl.OFF


def test_direct_provider_model_uses_root_policy_without_model_name_guessing(settings):
    settings.reasoning_policy = ReasoningPreference.LOW
    settings.reasoning_opus = ReasoningPreference.MAX

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="open_router/anthropic/claude-opus-4",
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.resolved.provider_id == "open_router"
    assert routed.resolved.provider_model == "anthropic/claude-opus-4"
    assert routed.reasoning.effort is ReasoningEffort.LOW


def test_model_router_routes_token_count_request(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    request = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_token_count_request(request)

    assert routed.request.model == "qwen2.5-7b"
    assert request.model == "claude-3-haiku-20240307"


def test_model_router_logs_mapping(settings):
    with patch("free_claude_code.application.routing.logger.debug") as mock_log:
        ModelRouter(settings).resolve("claude-2.1")

    mock_log.assert_called()
    args = mock_log.call_args[0]
    assert "MODEL MAPPING" in args[0]
    assert args[1] == "claude-2.1"
    assert args[2] == "fallback-model"


def test_model_router_preserves_typed_error_for_unknown_mapped_provider(settings):
    settings.model = "unknown/model"

    with pytest.raises(UnknownProviderError) as exc_info:
        ModelRouter(settings).resolve("claude-2.1")

    supported = "', '".join(PROVIDER_CATALOG)
    assert str(exc_info.value) == (
        f"Unknown provider_type: 'unknown'. Supported: '{supported}'"
    )


def test_model_router_resolves_a_tier_backup(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"
    settings.model_opus_backup = "gemini/gemini-3-pro"

    backups = ModelRouter(settings).resolve_backups("claude-opus-4-20250514")

    assert len(backups) == 1
    assert backups[0].original_model == "claude-opus-4-20250514"
    assert backups[0].provider_id == "gemini"
    assert backups[0].provider_model == "gemini-3-pro"
    assert backups[0].provider_model_ref == "gemini/gemini-3-pro"


def test_model_router_falls_back_to_the_global_backup(settings):
    settings.model_backup = "groq/llama-3.3-70b-versatile"
    router = ModelRouter(settings)

    tier_backups = router.resolve_backups("claude-sonnet-4-5")
    unmatched_backups = router.resolve_backups("claude-2.1")

    assert [backup.provider_model_ref for backup in tier_backups] == [
        "groq/llama-3.3-70b-versatile"
    ]
    assert [backup.provider_model_ref for backup in unmatched_backups] == [
        "groq/llama-3.3-70b-versatile"
    ]


def test_tier_backup_wins_over_the_global_backup(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"
    settings.model_haiku_backup = "gemini/gemini-3-flash"
    settings.model_backup = "groq/llama-3.3-70b-versatile"

    backups = ModelRouter(settings).resolve_backups("claude-3-haiku-20240307")

    assert [backup.provider_model_ref for backup in backups] == [
        "gemini/gemini-3-flash"
    ]


@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-chat",
        "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro",
        "claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro",
    ],
)
def test_direct_provider_refs_never_fail_over(settings, model_name):
    settings.model_backup = "groq/llama-3.3-70b-versatile"
    settings.model_opus_backup = "gemini/gemini-3-pro"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model=model_name,
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert ModelRouter(settings).resolve_backups(model_name) == ()
    assert routed.backups == ()


def test_backup_matching_its_effective_primary_is_dropped(settings):
    settings.model_opus = "gemini/gemini-3-pro"
    settings.model_opus_backup = "gemini/gemini-3-pro"
    settings.model_backup = "nvidia_nim/fallback-model"

    router = ModelRouter(settings)

    assert router.resolve_backups("claude-opus-4-20250514") == ()
    assert router.resolve_backups("claude-2.1") == ()


def test_backup_inherits_the_tier_reasoning_preference(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"
    settings.model_opus_backup = "gemini/gemini-3-pro"
    settings.reasoning_opus = ReasoningPreference.OFF

    router = ModelRouter(settings)
    primary = router.resolve("claude-opus-4-20250514")
    backups = router.resolve_backups("claude-opus-4-20250514")

    assert len(backups) == 1
    assert backups[0].reasoning_preference is ReasoningPreference.OFF
    assert backups[0].reasoning_preference is primary.reasoning_preference


def test_backup_provider_id_is_validated(settings):
    settings.model_backup = "unknown/model"

    with pytest.raises(UnknownProviderError):
        ModelRouter(settings).resolve_backups("claude-2.1")


def test_resolve_messages_request_attaches_the_backup_without_rewriting_the_body(
    settings,
):
    settings.model_opus = "open_router/deepseek/deepseek-r1"
    settings.model_opus_backup = "gemini/gemini-3-pro"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-opus-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek/deepseek-r1"
    assert len(routed.backups) == 1
    assert routed.backups[0].provider_id == "gemini"
    assert routed.backups[0].provider_model == "gemini-3-pro"


def test_requests_without_a_configured_backup_have_none(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-opus-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.backups == ()


def test_backup_chains_resolve_in_configured_order(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"
    settings.model_opus_backup = (
        "gemini/gemini-3-pro,groq/llama-3.3-70b-versatile,nvidia_nim/fallback-model"
    )

    backups = ModelRouter(settings).resolve_backups("claude-opus-4-20250514")

    assert [backup.provider_model_ref for backup in backups] == [
        "gemini/gemini-3-pro",
        "groq/llama-3.3-70b-versatile",
        "nvidia_nim/fallback-model",
    ]
    assert [backup.provider_id for backup in backups] == [
        "gemini",
        "groq",
        "nvidia_nim",
    ]
    assert all(
        backup.reasoning_preference is ReasoningPreference.CLIENT for backup in backups
    )


def test_backup_chain_drops_entries_matching_the_primary(settings):
    settings.model_opus = "gemini/gemini-3-pro"
    settings.model_opus_backup = "gemini/gemini-3-pro,groq/llama-3.3-70b-versatile"

    backups = ModelRouter(settings).resolve_backups("claude-opus-4-20250514")

    assert [backup.provider_model_ref for backup in backups] == [
        "groq/llama-3.3-70b-versatile"
    ]


def test_backup_chain_tolerates_whitespace_and_duplicates(settings):
    settings.model_backup = (
        " gemini/gemini-3-pro , groq/llama-3.3-70b-versatile ,gemini/gemini-3-pro "
    )

    backups = ModelRouter(settings).resolve_backups("claude-2.1")

    assert [backup.provider_model_ref for backup in backups] == [
        "gemini/gemini-3-pro",
        "groq/llama-3.3-70b-versatile",
    ]


def test_resolve_messages_request_attaches_the_whole_chain(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"
    settings.model_haiku_backup = "gemini/gemini-3-flash,groq/llama-3.3-70b-versatile"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "qwen2.5-7b"
    assert [backup.provider_model for backup in routed.backups] == [
        "gemini-3-flash",
        "llama-3.3-70b-versatile",
    ]
