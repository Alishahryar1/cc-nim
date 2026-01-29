import json
from typing import Dict, Optional


class CLIParser:
    """Helper to structure raw CLI events for the bot."""

    @staticmethod
    def parse_event(event: Dict) -> Optional[Dict]:
        """Filters and formats events for the bot client."""
        etype = event.get("type")

        # We are mainly interested in content updates, tool calls, and status
        if etype == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])

            # Simple text content
            text_parts = [c["text"] for c in content if c.get("type") == "text"]
            if text_parts:
                return {"type": "content", "text": "".join(text_parts)}

            # Tool calls
            tool_calls = [c for c in content if c.get("type") == "tool_use"]
            if tool_calls:
                return {"type": "tool_start", "tools": tool_calls}

        elif etype == "tool_output":
            return {"type": "tool_result", "output": event.get("output")}

        elif etype == "error":
            return {"type": "error", "message": event.get("error")}

        elif etype == "exit":
            return {
                "type": "complete",
                "status": "success" if event.get("code") == 0 else "failed",
            }

        return None
