# src/free_claude_code/core/rtk.py
import re

from loguru import logger


class RTKCompressor:
    """
    Advanced Rust Token Killer (RTK) style output compressor.
    Surgically removes ANSI codes, horizontal line bloat, repetitive logs,
    and redundant library middleware tracebacks while preserving raw core error signals.
    """

    # Regex to match ANSI escape sequences (colors, cursor movements, etc.)
    ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(self, aggressiveness: str = "medium", verbosity: int = 1):
        self.aggressiveness = aggressiveness.lower()
        self.verbosity = verbosity

    def compress(self, command: str, output: str) -> str:
        if not output:
            return output

        cmd_lower = command.strip().lower()

        # Phase 1: Heavy infrastructure cleanup (ANSI codes & Horizontal line capping)
        clean_output = self.ANSI_ESCAPE.sub("", output)
        clean_output = self._truncate_horizontal_bloat(clean_output)

        original_lines = len(clean_output.splitlines())

        # Phase 2: Contextual Smart Compression
        if any(k in cmd_lower for k in ["pytest", "test", "cargo test"]):
            result = self._smart_test_compress(clean_output)
        elif "git diff" in cmd_lower:
            result = self._smart_diff_compress(clean_output)
        elif "git status" in cmd_lower:
            result = self._smart_status_compress(clean_output)
        elif any(k in cmd_lower for k in ["grep", "find", "ls"]):
            result = self._smart_search_compress(clean_output)
        else:
            result = self._smart_generic_compress(clean_output)

        new_lines = len(result.splitlines())
        if self.verbosity >= 2:
            logger.debug(
                f"RTK optimized '{command}': {original_lines} -> {new_lines} lines"
            )

        return result

    def _truncate_horizontal_bloat(self, output: str) -> str:
        """Prevents token highjacking from ultra-long single lines (JSON dumps, minified files)."""
        max_line_len = 160 if self.aggressiveness == "high" else 300
        lines = output.splitlines()
        processed = []

        for line in lines:
            if len(line) > max_line_len:
                # Keep the beginning and end of the line, snip the middle fluff
                half = max_line_len // 2
                processed.append(
                    f"{line[:half]} ... [RTK Line Snip] ... {line[-half:]}"
                )
            else:
                processed.append(line)
        return "\n".join(processed)

    def _smart_test_compress(self, output: str) -> str:
        """Keeps core failures and optimizes tracebacks by skipping library noise."""
        lines = output.splitlines()
        smart_lines = []
        capture_traceback = False
        traceback_buffer = []

        for line in lines:
            upper_line = line.upper()

            if any(
                err in upper_line
                for err in ["FAIL", "ERROR", "TRACED", "EXCEPTION", "FAILED", "ERRORS"]
            ):
                capture_traceback = True
                if traceback_buffer:
                    smart_lines.extend(self._collapse_traceback(traceback_buffer))
                    traceback_buffer = []
                smart_lines.append(line)
            elif capture_traceback:
                # Stop capturing if we hit a clean line break or a new test execution section
                if line.strip() == "" or "====" in line or "----" in line:
                    capture_traceback = False
                    if traceback_buffer:
                        smart_lines.extend(self._collapse_traceback(traceback_buffer))
                        traceback_buffer = []
                traceback_buffer.append(line)
            elif "===" in line or "---" in line or "PASSED" in upper_line:
                if traceback_buffer:
                    smart_lines.extend(self._collapse_traceback(traceback_buffer))
                    traceback_buffer = []
                smart_lines.append(line)

        if traceback_buffer:
            smart_lines.extend(self._collapse_traceback(traceback_buffer))

        return "\n".join(smart_lines) if smart_lines else output

    def _collapse_traceback(self, traceback_lines: list[str]) -> list[str]:
        """Surgically drops internal library steps while protecting local workspace code errors."""
        if len(traceback_lines) <= 12 or self.aggressiveness == "low":
            return traceback_lines

        collapsed = []
        # Always preserve the initial invocation entry context
        collapsed.extend(traceback_lines[:4])
        collapsed.append("  ... [RTK hidden intermediate framework/library frames] ...")
        # Always preserve the actual ending line where the exception mutation broke
        collapsed.extend(traceback_lines[-6:])
        return collapsed

    def _smart_diff_compress(self, output: str) -> str:
        """Keeps file headers, hunk indicators, and actual modifications, drops context bloat."""
        lines = output.splitlines()
        compressed = [
            line
            for line in lines
            if line.startswith(("diff --git", "@@", "+", "-"))
            and not line.startswith(("+++ b/", "--- a/"))
        ]
        limit = 100 if self.aggressiveness == "high" else 250
        result = "\n".join(compressed[:limit])
        if len(compressed) > limit:
            result += "\n... [RTK truncated diff]"
        return result

    def _smart_status_compress(self, output: str) -> str:
        """Groups file changes cleanly without git banner text."""
        lines = output.splitlines()
        compressed = []
        for line in lines:
            clean = line.strip()
            if clean.startswith(
                ("modified:", "new file:", "deleted:", "??", "changes to be committed:")
            ):
                compressed.append(f"  {clean}")
        return "\n".join(compressed) if compressed else output

    def _smart_search_compress(self, output: str) -> str:
        """Deduplicates and caps repetitive search results."""
        lines = output.splitlines()
        unique_lines = list(dict.fromkeys(lines))
        limit = 20 if self.aggressiveness == "high" else 40

        if len(unique_lines) > limit:
            return (
                "\n".join(unique_lines[:limit])
                + f"\n... [RTK truncated {len(unique_lines) - limit} lines]"
            )
        return "\n".join(unique_lines)

    def _smart_generic_compress(self, output: str) -> str:
        """Strips double whitespace and trims excessive text blocks."""
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", output)
        limit = 1000 if self.aggressiveness == "high" else 3000

        if len(cleaned) > limit:
            return cleaned[:limit] + "\n... [RTK truncated]"
        return cleaned
