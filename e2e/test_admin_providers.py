"""Rendered provider-setup regressions for the local Admin UI."""

import pytest
from playwright.sync_api import ConsoleMessage, Page, ViewportSize, expect


def _open_admin(
    page: Page,
    admin_base_url: str,
    viewport: ViewportSize,
) -> None:
    page.set_viewport_size(viewport)
    page.emulate_media(reduced_motion="reduce")
    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")


@pytest.mark.parametrize(
    ("viewport", "desktop"),
    (
        ({"width": 1280, "height": 720}, True),
        ({"width": 390, "height": 844}, False),
    ),
)
def test_missing_provider_configuration_scrolls_to_exact_field(
    page: Page,
    admin_base_url: str,
    viewport: ViewportSize,
    desktop: bool,
) -> None:
    _open_admin(page, admin_base_url, viewport)
    card = page.locator('[data-provider="nvidia_nim"]')
    key_input = page.locator("#field-NVIDIA_NIM_API_KEY")

    expect(card.locator(".status-pill")).to_have_text("Missing key")
    expect(card.locator(".provider-meta")).to_have_text("NVIDIA_NIM_API_KEY")
    expect(card.get_by_role("button", name="Configure", exact=True)).to_be_visible()
    expect(card.get_by_role("button", name="Refresh models", exact=True)).to_have_count(
        0
    )
    expect(key_input).not_to_be_in_viewport()

    card.get_by_role("button", name="Configure", exact=True).click()

    expect(key_input).to_be_in_viewport()
    expect(key_input).to_be_focused()
    if desktop:
        sidebar = page.locator(".sidebar")
        expect(sidebar).to_have_css("position", "sticky")
        assert (
            round(
                float(
                    sidebar.evaluate("element => element.getBoundingClientRect().top")
                )
            )
            == 0
        )


def test_configured_provider_check_keeps_readiness_and_adds_models(
    page: Page,
    admin_base_url: str,
) -> None:
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="open_router"]')
    badge = card.locator(".status-pill")
    meta = card.locator(".provider-meta")

    expect(badge).to_have_text("Configured")
    expect(meta).to_have_text("OPENROUTER_API_KEY")
    expect(card.get_by_role("button", name="Edit", exact=True)).to_be_visible()
    card.get_by_role("button", name="Refresh models", exact=True).click()

    expect(card.locator(".provider-check-result")).to_have_text("2 models available")
    expect(badge).to_have_text("Configured")
    expect(meta).to_have_text("OPENROUTER_API_KEY")

    page.get_by_role("button", name="Model Config", exact=True).click()
    page.get_by_role("button", name="Show Fable Override options", exact=True).click()
    expect(
        page.get_by_role("option", name="open_router/vendor/model-a", exact=True)
    ).to_be_visible()


def test_provider_check_failure_is_separate_and_never_exposes_exception_text(
    page: Page,
    admin_base_url: str,
) -> None:
    console_messages: list[str] = []

    def record_console(message: ConsoleMessage) -> None:
        console_messages.append(message.text)

    page.on("console", record_console)
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="groq"]')
    card.get_by_role("button", name="Refresh models", exact=True).click()

    result = card.locator(".provider-check-result")
    expect(result).to_have_text(
        "Unavailable: Could not refresh this provider's models. "
        "Verify its configuration and access."
    )
    expect(card.locator(".status-pill")).to_have_text("Configured")
    expect(card.locator(".provider-meta")).to_have_text("GROQ_API_KEY")
    page_text = page.locator("body").inner_text()
    secret = "CREDENTIAL[unrecognized-format-987654321]"
    assert secret not in page_text
    assert "RuntimeError" not in page_text
    assert secret not in "\n".join(console_messages)


def test_multi_field_provider_targets_first_missing_configuration(
    page: Page,
    admin_base_url: str,
) -> None:
    _open_admin(page, admin_base_url, {"width": 1280, "height": 720})
    card = page.locator('[data-provider="cloudflare"]')
    account_input = page.locator("#field-CLOUDFLARE_ACCOUNT_ID")

    expect(card.locator(".status-pill")).to_have_text("Missing configuration")
    expect(card.locator(".provider-meta")).to_have_text(
        "CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID"
    )
    expect(account_input).not_to_be_in_viewport()

    card.get_by_role("button", name="Configure", exact=True).click()

    expect(account_input).to_be_in_viewport()
    expect(account_input).to_be_focused()
