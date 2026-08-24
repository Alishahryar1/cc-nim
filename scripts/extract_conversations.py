#!/usr/bin/env python3
"""
Claude Desktop Conversation Extractor & FCC Gateway Continuation Tool

Extracts conversation metadata from Claude Desktop's IndexedDB (react-query-cache),
fetches full conversation history via Anthropic API, and enables continuation
via FCC gateway models.

Workflow:
1. Open Claude Desktop in first-party mode, view conversation list (populates cache)
2. Run: python extract_conversations.py --list
3. To fetch full history: python extract_conversations.py --fetch <conv_id> --api-key $ANTHROPIC_API_KEY
4. To continue via FCC: python extract_conversations.py --continue <conv_id> --api-key $ANTHROPIC_API_KEY --gateway-url https://localhost:8443 --new-message "Continue here"
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

LEVELDB_DUMP_PATTERN = re.compile(r"  put '(.*?)' '(.*?)'")


class ConversationExtractor:
    def __init__(self, claude_config_dir: Path | None = None):
        self.claude_dir = claude_config_dir or Path.home() / ".config" / "Claude"
        self.indexeddb_dir = (
            self.claude_dir / "IndexedDB" / "https_claude.ai_0.indexeddb.leveldb"
        )
        self.leveldbutil = Path("/tmp/leveldb/build/leveldbutil")

    def dump_indexeddb(self) -> str:
        """Dump the IndexedDB to a temporary file using leveldbutil."""
        if not self.indexeddb_dir.exists():
            raise FileNotFoundError(f"IndexedDB not found: {self.indexeddb_dir}")

        # Copy to temp dir to avoid lock
        with tempfile.TemporaryDirectory() as tmpdir:
            db_copy = Path(tmpdir) / "db_copy"
            shutil.copytree(self.indexeddb_dir, db_copy)
            (db_copy / "LOCK").unlink(missing_ok=True)

            # Dump the log file
            log_file = db_copy / "000003.log"
            if not log_file.exists():
                raise FileNotFoundError("Log file 000003.log not found")

            result = subprocess.run(
                [str(self.leveldbutil), "dump", str(log_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"leveldbutil failed: {result.stderr}")

            return result.stdout

    def parse_dump(self, dump_content: str) -> list[tuple[bytes, bytes]]:
        """Parse leveldbutil dump output into key-value byte pairs."""
        entries = []
        for match in LEVELDB_DUMP_PATTERN.finditer(dump_content):
            key_escaped = match.group(1)
            val_escaped = match.group(2)

            try:
                key_bytes = (
                    key_escaped.encode("utf-8")
                    .decode("unicode_escape")
                    .encode("latin-1")
                )
                val_bytes = (
                    val_escaped.encode("utf-8")
                    .decode("unicode_escape")
                    .encode("latin-1")
                )
            except Exception:
                # Handle escape sequences that might have non-latin-1 chars
                key_bytes = (
                    key_escaped.encode("utf-8", errors="surrogateescape")
                    .decode("unicode_escape", errors="ignore")
                    .encode("latin-1", errors="ignore")
                )
                val_bytes = (
                    val_escaped.encode("utf-8", errors="surrogateescape")
                    .decode("unicode_escape", errors="ignore")
                    .encode("latin-1", errors="ignore")
                )

            entries.append((key_bytes, val_bytes))

        return entries

    def extract_conversations(
        self, entries: list[tuple[bytes, bytes]]
    ) -> list[dict[str, Any]]:
        """Extract conversation metadata from react-query-cache entries."""
        conversations = []

        for key_bytes, val_bytes in entries:
            # Check for react-query-cache key (starts with \x00\x01\x01\x01\x01\x11)
            if key_bytes.startswith(b"\x00\x01\x01\x01\x01\x11"):
                # Parse the protobuf-like value
                convs = self._parse_react_query_cache(val_bytes)
                conversations.extend(convs)

            # Also check data entries (start with \x00\x00\x00\x002\x02)
            elif (
                key_bytes.startswith(b"\x00\x00\x00\x002\x02") and len(val_bytes) > 100
            ):
                val_str = val_bytes.decode("utf-8", errors="ignore")
                if "conversations_v2:anon" in val_str:
                    convs = self._parse_cached_data(val_bytes)
                    conversations.extend(convs)

        return conversations

    def _parse_react_query_cache(self, data: bytes) -> list[dict[str, Any]]:
        """Parse the react-query-cache value (118 bytes)."""
        # This is a protobuf-like format:
        # - Field 1: "react-query-cache" (string)
        # - Field 2: nested message with buster, conversations_v2:anon, timestamp, clientState, mutations, queries
        conversations = []

        data_str = data.decode("utf-8", errors="ignore")

        # Extract timestamp from the data
        ts_match = re.search(r"timestampN(\x00.{8})", data_str)
        if ts_match:
            try:
                ts_bytes = ts_match.group(1).encode("latin-1")
                if len(ts_bytes) >= 8:
                    ns = int.from_bytes(ts_bytes[:8], "big")
                    dt = datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)
                    print(f"  Cache timestamp: {dt.isoformat()}")
            except Exception:
                pass

        # The 'queries' field appears to be empty (just { {)
        # This means no conversations are currently cached
        if "queries" in data_str:
            queries_idx = data_str.index("queries")
            after_queries = data_str[queries_idx:]
            if "queries" in after_queries and "{" in after_queries:
                # Check if queries is empty
                if (
                    "a\x00@\x00\x00" in after_queries
                    or "a\\x00@\\x00\\x00" in after_queries
                ):
                    print("  Cache appears EMPTY (no conversations cached)")

        return conversations

    def _parse_cached_data(self, data: bytes) -> list[dict[str, Any]]:
        """Parse the full cached data entry containing react-query-cache."""
        conversations = []
        # The data format: outer message with field 1 (key) and field 2 (value)
        # The value contains the same structure as react-query-cache entries
        print(f"  Found cached data entry ({len(data)} bytes)")
        return conversations


def fetch_conversation_list(api_key: str) -> list[dict[str, Any]]:
    """Fetch conversation list from Anthropic API."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = requests.get(
        "https://api.anthropic.com/v1/conversations",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def fetch_conversation_messages(
    api_key: str, conversation_id: str
) -> list[dict[str, Any]]:
    """Fetch full conversation messages from Anthropic API."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = requests.get(
        f"https://api.anthropic.com/v1/conversations/{conversation_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("messages", [])


def continue_via_fcc_gateway(
    messages: list[dict[str, Any]],
    new_message: str,
    gateway_url: str,
    api_key: str = "freecc",
    model: str = "claude-sonnet-nim-0001",
) -> requests.Response:
    """Send conversation + new message to FCC gateway."""
    url = f"{gateway_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": [*messages, {"role": "user", "content": new_message}],
    }
    return requests.post(url, json=payload, headers=headers, timeout=120)


def anthropic_to_fcc_messages(
    anthropic_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic API message format to FCC/Anthropic Messages API format."""
    fcc_messages = []
    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle content that might be a list of blocks
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
            content = "\n".join(text_parts)

        if content:
            fcc_messages.append({"role": role, "content": content})

    return fcc_messages


def main():
    parser = argparse.ArgumentParser(
        description="Extract Claude Desktop conversations and continue via FCC gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--list", action="store_true", help="List conversations from IndexedDB cache"
    )
    parser.add_argument(
        "--list-api", action="store_true", help="List conversations via Anthropic API"
    )
    parser.add_argument(
        "--fetch", type=str, help="Fetch full conversation from Anthropic API by ID"
    )
    parser.add_argument(
        "--api-key", type=str, help="Anthropic API key (or set ANTHROPIC_API_KEY env)"
    )
    parser.add_argument(
        "--continue",
        dest="continue_conv",
        type=str,
        help="Continue conversation via FCC gateway by ID",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="https://localhost:8443",
        help="FCC gateway URL (default: https://localhost:8443)",
    )
    parser.add_argument(
        "--new-message",
        type=str,
        default="Please continue",
        help="New message to send when continuing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-nim-0001",
        help="FCC gateway model alias (default: claude-sonnet-nim-0001)",
    )
    parser.add_argument(
        "--force-dump",
        action="store_true",
        help="Force re-dump of IndexedDB even if cached dump exists",
    )

    args = parser.parse_args()

    if not any([args.list, args.list_api, args.fetch, args.continue_conv]):
        parser.print_help()
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if (args.list_api or args.fetch or args.continue_conv) and not api_key:
        print(
            "Error: --api-key or ANTHROPIC_API_KEY environment variable required for API operations"
        )
        sys.exit(1)

    if args.list_api or args.fetch or args.continue_conv:
        assert api_key is not None
        api_key_str: str = api_key
    else:
        api_key_str = ""

    if args.list_api:
        print("=== Fetching conversations from Anthropic API ===\n")
        try:
            conversations = fetch_conversation_list(api_key_str)
            if not conversations:
                print("No conversations found.")
            else:
                print(f"Found {len(conversations)} conversations:\n")
                for i, conv in enumerate(conversations):
                    print(f"  {i + 1}. {conv.get('name', 'Untitled')[:60]}")
                    print(f"     ID: {conv.get('uuid', conv.get('id', 'N/A'))}")
                    print(f"     Created: {conv.get('created_at', 'N/A')}")
                    print(f"     Updated: {conv.get('updated_at', 'N/A')}")
                    print()
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    if args.list:
        print("=== Extracting conversations from Claude Desktop IndexedDB ===\n")

        extractor = ConversationExtractor()
        try:
            dump = extractor.dump_indexeddb()
            entries = extractor.parse_dump(dump)
            print(f"Parsed {len(entries)} key-value entries\n")

            conversations = extractor.extract_conversations(entries)

            if not conversations:
                print("No conversations found in cache.")
                print("\nTo populate the cache:")
                print("1. Open Claude Desktop in first-party mode (without FCC config)")
                print("2. View the conversation list in the sidebar")
                print("3. Run this tool again with --list")
                print("\nAlternatively, use the Anthropic API directly:")
                print(
                    "  python extract_conversations.py --list-api --api-key $ANTHROPIC_API_KEY"
                )
            else:
                print(f"Found {len(conversations)} conversations in cache:\n")
                for i, conv in enumerate(conversations):
                    print(f"  {i + 1}. ID: {conv.get('id', 'N/A')}")
                    print(f"     Title: {conv.get('title', 'N/A')}")
                    print(f"     Updated: {conv.get('updated_at', 'N/A')}")
                    print()

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.fetch:
        print(f"=== Fetching conversation {args.fetch} from Anthropic API ===\n")
        try:
            messages = fetch_conversation_messages(api_key_str, args.fetch)
            fcc_messages = anthropic_to_fcc_messages(messages)

            print(
                f"Fetched {len(messages)} messages, converted to {len(fcc_messages)} FCC messages\n"
            )

            # Save to file for inspection
            output_file = f"conversation_{args.fetch}.json"
            with open(output_file, "w") as f:
                json.dump(fcc_messages, f, indent=2)
            print(f"Saved to: {output_file}")

            # Print preview
            for i, msg in enumerate(fcc_messages[:5]):
                role = msg.get("role", "?")
                content = msg.get("content", "")[:100]
                print(f"  {i + 1}. [{role}] {content}...")
            if len(fcc_messages) > 5:
                print(f"  ... and {len(fcc_messages) - 5} more messages")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.continue_conv:
        print(f"=== Continuing conversation {args.continue_conv} via FCC Gateway ===\n")

        try:
            # Fetch full conversation history
            print(f"Fetching conversation {args.continue_conv} from Anthropic API...")
            messages = fetch_conversation_messages(api_key_str, args.continue_conv)
            fcc_messages = anthropic_to_fcc_messages(messages)

            print(f"Loaded {len(fcc_messages)} messages from history")
            print(f"Sending to FCC gateway: {args.gateway_url}")
            print(f"Model: {args.model}")
            print(f"New message: {args.new_message}\n")

            # Send to FCC gateway
            response = continue_via_fcc_gateway(
                fcc_messages,
                args.new_message,
                args.gateway_url,
                api_key="freecc",
                model=args.model,
            )

            if response.status_code == 200:
                result = response.json()
                print("=== Response ===")
                content = result.get("content", [])
                for block in content:
                    if block.get("type") == "text":
                        print(block.get("text", ""))
            else:
                print(f"Error ({response.status_code}): {response.text}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
