"""Edge case tests for api/detection.py."""

from unittest.mock import patch

from api.detection import (
    is_filepath_extraction_request,
    is_prefix_detection_request,
    is_safety_classifier_request,
)
from api.models.anthropic import Message, MessagesRequest


def _make_request(content: str, **kwargs) -> MessagesRequest:
    return MessagesRequest(
        model="claude-3-sonnet",
        max_tokens=100,
        messages=[Message(role="user", content=content)],
        **kwargs,
    )


class TestIsPrefixDetectionRequest:
    def test_output_marker_handling(self):
        """Content with Command: but Output: after cmd_start; output has < or \\n\\n."""
        content = "<policy_spec> Command:\nls -la\nOutput:\na.txt\nb.txt\n\nmore"
        req = _make_request(content)
        is_req, cmd = is_prefix_detection_request(req)
        assert is_req is True
        assert "ls -la" in cmd

    def test_prefix_detection_with_empty_command_section(self):
        """Command: at end with no content returns empty command."""
        req = _make_request("<policy_spec> Command: ")
        is_req, cmd = is_prefix_detection_request(req)
        assert is_req is True
        assert cmd == ""

    def test_exception_in_try_returns_false(self):
        """Exception in try block (e.g. content slice) returns False, ''."""
        req = _make_request("<policy_spec> Command: x")

        # Return object that raises when sliced - triggers except in is_prefix_detection_request
        class BadStr(str):
            def __getitem__(self, key):
                raise TypeError("bad slice")

        with patch(
            "api.detection.extract_text_from_content",
            return_value=BadStr("<policy_spec> Command: x"),
        ):
            is_req, cmd = is_prefix_detection_request(req)
        assert is_req is False
        assert cmd == ""


class TestIsSafetyClassifierRequest:
    _SYSTEM = (
        "You are a security monitor. Respond with <block>yes</block> "
        "or <block>no</block>."
    )
    _USER = (
        "<transcript>\nUser: review the repo\n"
        "WebFetch https://example.com: fetch\n</transcript>\n<block> immediately."
    )

    def test_classifier_request_detected(self):
        req = _make_request(self._USER, system=self._SYSTEM)
        assert is_safety_classifier_request(req) is True

    def test_markers_split_across_system_and_user(self):
        req = _make_request(
            "<transcript>\nWebFetch x\n</transcript>", system=self._SYSTEM
        )
        assert is_safety_classifier_request(req) is True

    def test_request_with_tools_is_not_classifier(self):
        req = _make_request(self._USER, system=self._SYSTEM, tools=[{"name": "search"}])
        assert is_safety_classifier_request(req) is False

    def test_missing_transcript_marker(self):
        req = _make_request("<block> immediately", system=self._SYSTEM)
        assert is_safety_classifier_request(req) is False

    def test_missing_verdict_instruction(self):
        req = _make_request(
            "<transcript>\nWebFetch x\n</transcript>", system="just chatting"
        )
        assert is_safety_classifier_request(req) is False

    def test_xml_content_without_verdict_instruction(self):
        req = _make_request(
            "Explain this format: <transcript> ... </transcript> and a <block> tag."
        )
        assert is_safety_classifier_request(req) is False


class TestIsFilepathExtractionRequest:
    def test_output_marker_minus_one_returns_false(self):
        """Output: not found after Command: returns False."""
        content = "Command:\nls\nfilepaths"
        req = _make_request(content)
        is_fp, cmd, out = is_filepath_extraction_request(req)
        assert is_fp is False
        assert cmd == ""
        assert out == ""

    def test_output_has_angle_bracket_splits(self):
        """Output containing < is split and first part used."""
        content = "Command:\nls\nOutput:\na.txt b.txt <extra>\nfilepaths"
        req = _make_request(content)
        is_fp, _cmd, out = is_filepath_extraction_request(req)
        assert is_fp is True
        assert "<" not in out
        assert out == "a.txt b.txt"

    def test_output_has_double_newline_splits(self):
        """Output containing \\n\\n is split and first part used."""
        content = "Command:\nls\nOutput:\na.txt\nb.txt\n\nmore text\nfilepaths"
        req = _make_request(content)
        is_fp, _cmd, out = is_filepath_extraction_request(req)
        assert is_fp is True
        assert "more" not in out
