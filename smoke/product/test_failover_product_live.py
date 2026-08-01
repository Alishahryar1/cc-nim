"""A quota-exhausted primary provider is served by its configured backup."""

import pytest

from free_claude_code.core.anthropic.stream_contracts import text_content
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver
from smoke.lib.http import collect_message_stream, message_payload
from smoke.lib.stub_upstream import (
    failing_primary_and_healthy_backup,
    stub_upstream,
)

pytestmark = [pytest.mark.live]

_BACKUP_REPLY = "FCC_SMOKE_FAILOVER_BACKUP"
_UNSET_ROUTES = (
    "MODEL_FABLE",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "MODEL_HAIKU",
    "MODEL_FABLE_BACKUP",
    "MODEL_OPUS_BACKUP",
    "MODEL_SONNET_BACKUP",
    "MODEL_HAIKU_BACKUP",
)


@pytest.mark.smoke_target("config")
def test_quota_exhausted_primary_is_served_by_its_backup_e2e(
    smoke_config: SmokeConfig,
) -> None:
    with failing_primary_and_healthy_backup(
        retry_after="3600",
        reply=_BACKUP_REPLY,
    ) as (primary, backup):
        driver = SmokeServerDriver(
            smoke_config,
            name="failover",
            env_overrides={
                "MODEL": "lmstudio/stub-primary",
                "MODEL_BACKUP": "llamacpp/stub-backup",
                "LM_STUDIO_BASE_URL": primary.base_url,
                "LLAMACPP_BASE_URL": backup.base_url,
                **dict.fromkeys(_UNSET_ROUTES, ""),
            },
        )
        with driver.run() as server:
            events = collect_message_stream(
                server,
                message_payload("hello", model="claude-sonnet-4-20250514"),
                smoke_config,
            )

    assert _BACKUP_REPLY in text_content(events)
    assert primary.chat_request_count == 1
    assert backup.chat_request_count == 1
    assert backup.chat_requests[0]["model"] == "stub-backup"


@pytest.mark.smoke_target("config")
def test_healthy_primary_is_never_replaced_by_its_backup_e2e(
    smoke_config: SmokeConfig,
) -> None:
    with failing_primary_and_healthy_backup(
        retry_after="3600",
        reply=_BACKUP_REPLY,
    ) as (_unused, backup):
        driver = SmokeServerDriver(
            smoke_config,
            name="failover-healthy",
            env_overrides={
                "MODEL": "llamacpp/stub-backup",
                "MODEL_BACKUP": "lmstudio/stub-primary",
                "LM_STUDIO_BASE_URL": _unused.base_url,
                "LLAMACPP_BASE_URL": backup.base_url,
                **dict.fromkeys(_UNSET_ROUTES, ""),
            },
        )
        with driver.run() as server:
            events = collect_message_stream(
                server,
                message_payload("hello", model="claude-sonnet-4-20250514"),
                smoke_config,
            )

    assert _BACKUP_REPLY in text_content(events)
    assert backup.chat_request_count == 1
    assert _unused.chat_request_count == 0


@pytest.mark.smoke_target("config")
def test_backup_chain_walks_to_the_last_healthy_link_e2e(
    smoke_config: SmokeConfig,
) -> None:
    with (
        stub_upstream(
            model_id="stub-primary", status=429, retry_after="3600"
        ) as primary,
        stub_upstream(model_id="stub-first", status=429, retry_after="3600") as first,
        stub_upstream(model_id="stub-last", reply=_BACKUP_REPLY) as last,
    ):
        driver = SmokeServerDriver(
            smoke_config,
            name="failover-chain",
            env_overrides={
                "MODEL": "lmstudio/stub-primary",
                "MODEL_BACKUP": "llamacpp/stub-first,ollama/stub-last",
                "LM_STUDIO_BASE_URL": primary.base_url,
                "LLAMACPP_BASE_URL": first.base_url,
                "OLLAMA_BASE_URL": last.base_url,
                **dict.fromkeys(_UNSET_ROUTES, ""),
            },
        )
        with driver.run() as server:
            events = collect_message_stream(
                server,
                message_payload("hello", model="claude-sonnet-4-20250514"),
                smoke_config,
            )

    assert _BACKUP_REPLY in text_content(events)
    assert primary.chat_request_count == 1
    assert first.chat_request_count == 1
    assert last.chat_request_count == 1
    assert last.chat_requests[0]["model"] == "stub-last"
