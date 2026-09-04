import os
import re
import shlex
import sys
from pathlib import Path

import pytest
from playwright.sync_api import Page, ViewportSize, expect


def _new_terminal(page: Page, admin_base_url: str) -> str:
    page.goto(f"{admin_base_url}/admin")
    page.get_by_role("button", name="Terminal Sessions").click()
    expect(page).to_have_url(f"{admin_base_url}/admin/terminal")
    page.get_by_role("button", name="New terminal").click()
    expect(page).to_have_url(re.compile(r"/admin/terminal/[0-9a-f-]{36}$"))
    expect(page.locator(".terminal-stage .xterm")).to_be_visible()
    expect(page.locator(".xterm-rows")).to_contain_text("FCC browser-test terminal")
    expect(page.locator(".xterm-helper-textarea")).to_be_focused()
    return page.url


def _type_in_terminal(page: Page, text: str) -> None:
    terminal_input = page.locator(".xterm-helper-textarea")
    terminal_input.focus()
    page.keyboard.type(text)


def _insert_command(page: Page, *arguments: str) -> None:
    if os.name == "nt":
        command = "& " + " ".join(
            f"'{argument.replace("'", "''")}'" for argument in arguments
        )
    else:
        command = shlex.join(arguments)
    terminal_input = page.locator(".xterm-helper-textarea")
    terminal_input.focus()
    page.keyboard.insert_text(command)
    page.keyboard.press("Enter")


def _terminal_geometry(page: Page) -> dict[str, float]:
    return page.evaluate(
        """() => {
          const terminal = document.querySelector(".terminal-stage .xterm");
          const screen = document.querySelector(".terminal-stage .xterm-screen");
          const terminalRect = terminal.getBoundingClientRect();
          const screenRect = screen.getBoundingClientRect();
          return {
            terminalWidth: terminalRect.width,
            terminalHeight: terminalRect.height,
            screenWidth: screenRect.width,
            screenHeight: screenRect.height,
          };
        }"""
    )


def test_terminal_fit_stays_inside_container_across_viewport_resizes(
    page: Page,
    admin_base_url: str,
) -> None:
    terminal_url = _new_terminal(page, admin_base_url)
    session_id = terminal_url.rsplit("/", 1)[-1]
    try:
        geometries = []
        viewports: tuple[ViewportSize, ...] = (
            {"width": 760, "height": 500},
            {"width": 1280, "height": 720},
        )
        for viewport in viewports:
            page.set_viewport_size(viewport)
            page.wait_for_timeout(150)
            geometry = _terminal_geometry(page)
            geometries.append(geometry)
            assert geometry["screenWidth"] <= geometry["terminalWidth"] + 1
            assert geometry["screenHeight"] <= geometry["terminalHeight"] + 1

        assert geometries[1]["terminalHeight"] > geometries[0]["terminalHeight"] + 100
    finally:
        page.request.delete(
            f"{admin_base_url}/admin/api/terminal/sessions/{session_id}"
        )


def test_terminal_survives_navigation_and_syncs_across_tabs(
    page: Page,
    admin_base_url: str,
) -> None:
    stop_requests: list[str] = []
    page.on(
        "request",
        lambda request: (
            stop_requests.append(request.url)
            if request.method == "POST" and request.url.endswith("/stop")
            else None
        ),
    )
    terminal_url = _new_terminal(page, admin_base_url)
    session_id = terminal_url.rsplit("/", 1)[-1]
    detail_url = f"{admin_base_url}/admin/api/terminal/sessions/{session_id}"
    second = page.context.new_page()
    try:
        second.set_viewport_size({"width": 1200, "height": 760})
        second.goto(terminal_url)
        expect(second.locator(".terminal-stage .xterm")).to_be_visible()
        expect(second.locator(".xterm-rows")).to_contain_text(
            "FCC browser-test terminal"
        )
        second.bring_to_front()
        second.locator(".xterm-helper-textarea").focus()
        second.wait_for_function(
            """async (url) => {
              const response = await fetch(url);
              const session = await response.json();
              return session.rows !== 24 || session.columns !== 80;
            }""",
            arg=detail_url,
        )

        page.set_viewport_size({"width": 560, "height": 520})
        page.wait_for_timeout(150)
        expect(page.locator("[data-terminal-notice]")).not_to_contain_text(
            "Reconnecting"
        )
        expect(second.locator("[data-terminal-notice]")).not_to_contain_text(
            "Reconnecting"
        )

        _type_in_terminal(second, "shared-output")
        expect(second.locator(".xterm-rows")).to_contain_text("shared-output")
        expect(page.locator(".xterm-rows")).to_contain_text("shared-output")

        page.bring_to_front()
        page.reload()
        expect(page.locator(".xterm-rows")).to_contain_text("shared-output")
        page.get_by_role("button", name="Providers").click()
        expect(page.locator('[data-provider="nvidia_nim"]')).to_be_visible()
        page.get_by_role("button", name="Terminal Sessions").click()
        page.locator(".terminal-card").click()
        expect(page.locator(".xterm-rows")).to_contain_text("shared-output")
        assert stop_requests == []
    finally:
        second.close()
        page.request.delete(
            f"{admin_base_url}/admin/api/terminal/sessions/{session_id}"
        )


def test_terminal_rename_stop_and_delete(page: Page, admin_base_url: str) -> None:
    _new_terminal(page, admin_base_url)

    name = page.get_by_role("textbox", name="Terminal name")
    name.fill("Release shell")
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH"
            and "/admin/api/terminal/sessions/" in response.url
        )
    ):
        name.press("Enter")
    expect(name).to_have_value("Release shell")

    page.get_by_role("button", name="Stop", exact=True).click()
    expect(page.locator("[data-terminal-status]")).to_have_text("exited")
    expect(page.get_by_role("button", name="Stop", exact=True)).to_be_disabled()

    page.on("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Delete", exact=True).click()
    expect(page).to_have_url(f"{admin_base_url}/admin/terminal")
    expect(page.get_by_role("heading", name="No terminals yet")).to_be_visible()


@pytest.mark.real_terminal
@pytest.mark.skipif(os.name == "nt", reason="real browser scenario runs on Linux CI")
def test_real_terminal_engine_survives_two_tabs_reconnect_resize_and_large_input(
    page: Page,
    admin_base_url: str,
    tmp_path: Path,
) -> None:
    large_input = tmp_path / "large_input.py"
    large_input.write_text(
        "import os\n"
        "import sys\n"
        "import termios\n"
        "import tty\n"
        "descriptor = sys.stdin.fileno()\n"
        "attributes = termios.tcgetattr(descriptor)\n"
        "data = bytearray()\n"
        "try:\n"
        "    tty.setraw(descriptor)\n"
        "    sys.stdout.write('LARGE_READY\\r\\n')\n"
        "    sys.stdout.flush()\n"
        "    while len(data) < 70003:\n"
        "        data.extend(os.read(descriptor, 65536))\n"
        "finally:\n"
        "    termios.tcsetattr(descriptor, termios.TCSADRAIN, attributes)\n"
        "sys.stdout.write(f'\\r\\nLARGE:{len(data)}:{bytes(data[-3:]).decode()}\\r\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )
    alternate_screen = tmp_path / "alternate_screen.py"
    alternate_screen.write_text(
        "import sys\n"
        "import time\n"
        "sys.stdout.write('\\x1b[?1049h\\x1b[2J\\x1b[HFULL_SCREEN')\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.25)\n"
        "sys.stdout.write('\\x1b[?1049lALT_DONE\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    page.goto(f"{admin_base_url}/admin/terminal")
    page.get_by_role("button", name="New terminal").click()
    expect(page).to_have_url(re.compile(r"/admin/terminal/[0-9a-f-]{36}$"))
    terminal_url = page.url
    session_id = terminal_url.rsplit("/", 1)[-1]
    second = page.context.new_page()
    try:
        expect(page.locator(".terminal-stage .xterm")).to_be_visible()
        _insert_command(page, sys.executable, str(alternate_screen))
        expect(page.locator(".xterm-rows")).to_contain_text("ALT_DONE")

        second.goto(terminal_url)
        expect(second.locator(".xterm-rows")).to_contain_text("ALT_DONE")
        _insert_command(second, sys.executable, "-c", "print('SECOND_TAB_CONTROL')")
        expect(second.locator(".xterm-rows")).to_contain_text("SECOND_TAB_CONTROL")
        expect(page.locator(".xterm-rows")).to_contain_text("SECOND_TAB_CONTROL")

        page.reload()
        expect(page.locator(".xterm-rows")).to_contain_text("SECOND_TAB_CONTROL")

        second.set_viewport_size({"width": 640, "height": 480})
        second.wait_for_timeout(150)
        second.set_viewport_size({"width": 1280, "height": 760})
        second.wait_for_timeout(150)
        geometry = _terminal_geometry(second)
        assert geometry["screenWidth"] <= geometry["terminalWidth"] + 1
        assert geometry["screenHeight"] <= geometry["terminalHeight"] + 1

        _insert_command(second, sys.executable, str(large_input))
        expect(second.locator(".xterm-rows")).to_contain_text("LARGE_READY")
        second.locator(".xterm-helper-textarea").focus()
        second.keyboard.insert_text("a" * 70_000 + "✓")
        expect(second.locator(".xterm-rows")).to_contain_text(
            "LARGE:70003:✓", timeout=20_000
        )
        expect(page.locator(".xterm-rows")).to_contain_text(
            "LARGE:70003:✓", timeout=20_000
        )
    finally:
        second.close()
        page.request.delete(
            f"{admin_base_url}/admin/api/terminal/sessions/{session_id}"
        )
