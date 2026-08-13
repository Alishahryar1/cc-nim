"""Tests for neutral OpenAI-compatible model-list filtering."""

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
    RequiredPathValues,
    extract_openai_model_infos,
)


def test_required_path_values_match_nested_json_scalars() -> None:
    model_infos = extract_openai_model_infos(
        {
            "data": [
                {
                    "id": "included",
                    "metadata": {
                        "kind": "chat",
                        "enabled": True,
                    },
                },
                {
                    "id": "excluded",
                    "metadata": {
                        "kind": "image",
                        "enabled": True,
                    },
                },
            ]
        },
        provider_name="TEST",
        required_path_values=(
            (("metadata", "kind"), ("chat", "code")),
            (("metadata", "enabled"), (True,)),
        ),
    )

    assert model_infos == frozenset({ProviderModelInfo("included")})


@pytest.mark.parametrize(
    ("metadata", "required_path_values", "path"),
    [
        ({}, ((("kind",), ("chat",)),), "kind"),
        ({"kind": True}, ((("kind",), ("chat",)),), "kind"),
        ({"enabled": 1}, ((("enabled",), (True,)),), "enabled"),
    ],
)
def test_required_path_values_reject_missing_or_wrong_scalar_types(
    metadata: dict[str, object],
    required_path_values: RequiredPathValues,
    path: str,
) -> None:
    with pytest.raises(ModelListResponseError, match=path):
        extract_openai_model_infos(
            {"data": [{"id": "model", **metadata}]},
            provider_name="TEST",
            required_path_values=required_path_values,
        )


def test_required_path_values_reject_an_empty_included_set() -> None:
    with pytest.raises(ModelListResponseError, match="did not include any model ids"):
        extract_openai_model_infos(
            {"data": [{"id": "image", "type": "image"}]},
            provider_name="TEST",
            required_path_values=((("type",), ("chat",)),),
        )


def test_duplicate_model_ids_preserve_first_validated_capability() -> None:
    model_infos = extract_openai_model_infos(
        {
            "data": [
                {"id": "overlap", "features": ["reasoning"]},
                {"id": "overlap", "features": ["non-reasoning"]},
            ]
        },
        provider_name="TEST",
        tags_field="features",
        non_thinking_tag="non-reasoning",
    )

    assert model_infos == frozenset(
        {ProviderModelInfo("overlap", supports_thinking=True)}
    )
