#!/usr/bin/env python3
"""
Playwright scraper - connect to running Claude Desktop via CDP.
"""

import asyncio
import json

from playwright.async_api import async_playwright


async def main():
    """Main scraper function - connect to existing CDP session."""
    output_file = "claude_conversations.json"

    async with async_playwright() as p:
        # Connect to running Chrome/Electron via CDP
        print("Connecting to CDP on port 9222...")
        browser = await p.chromium.connect_over_cdp("http://localhost:9222")

        contexts = browser.contexts
        if not contexts:
            print("No contexts found")
            await browser.close()
            return

        context = contexts[0]
        pages = context.pages

        print(f"Found {len(pages)} pages")

        # Find the page with claude.ai
        target_page = None
        for page in pages:
            try:
                url = page.url
                print(f"Page: {url}")
                if "claude.ai" in url:
                    target_page = page
                    break
            except Exception as e:
                print(f"  Error getting URL: {e}")

        if not target_page:
            print("No claude.ai page found. Trying first page...")
            target_page = pages[0]

        print(f"Using page: {target_page.url}")

        # Wait for conversation list
        print("Waiting for conversation list...")
        try:
            await target_page.wait_for_selector(
                '[data-testid="conversation-list"], '
                'nav[aria-label="Conversations"], '
                ".conversation-list, "
                "#conversations-list",
                timeout=30000,
            )
        except Exception as e:
            print(f"Selector wait failed: {e}")
            print("Page content preview:")
            content = await target_page.content()
            print(content[:2000])

        # Extract conversations
        print("\nExtracting conversations...")
        conversations = []

        # Try multiple selectors
        items = await target_page.query_selector_all(
            '[data-testid="conversation-list"] button, '
            '[data-testid*="conversation-item"], '
            'nav[aria-label="Conversations"] button, '
            ".conversation-list-item, "
            "#conversations-list button, "
            ".sidebar button"
        )

        print(f"Found {len(items)} potential items")

        for item in items[:10]:
            try:
                text = await item.inner_text()
                if text.strip():
                    print(f"  Item: {text.strip()[:80]}")
                    id_attr = (
                        await item.get_attribute("data-conversation-id")
                        or await item.get_attribute("data-id")
                        or await item.get_attribute("href")
                        or f"item_{len(conversations)}"
                    )
                    conversations.append(
                        {
                            "title": text.strip()[:100],
                            "id": id_attr,
                            "element": item,
                        }
                    )
            except Exception:
                pass

        if not conversations:
            print(
                "No conversations found with standard selectors. Dumping page structure..."
            )
            # Get all buttons/links in sidebar
            all_clickable = await target_page.query_selector_all(
                'button, a[href*="chat"], a[href*="conversation"]'
            )
            print(f"Found {len(all_clickable)} clickable elements")
            for elem in all_clickable[:20]:
                try:
                    text = await elem.inner_text()
                    if text.strip():
                        print(f"  {text.strip()[:80]}")
                except Exception:
                    pass

        # Save results
        with open(output_file, "w") as f:
            json.dump(conversations, f, indent=2)

        print(f"\nSaved {len(conversations)} conversations to {output_file}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
