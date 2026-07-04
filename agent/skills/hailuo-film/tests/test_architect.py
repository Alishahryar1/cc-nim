"""Tests for scripts/architect.py — no network, no browser."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Make scripts importable without installing the package.
sys_path = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(sys_path))

import architect


SAMPLE_PROXY_RESPONSE = json.dumps({
    "content": [
        {
            "type": "text",
            "text": json.dumps({
                "concept": "Drone sunrise teaser",
                "format_mode": "cinematic film",
                "assets": [
                    {
                        "id": "hero-pilot",
                        "type": "character",
                        "description": "Rugged drone operator at dawn",
                        "prompt": "SCENE CONTEXT\nA rugged drone operator...\nPOSITIVE LOCKS\nAlways include...",
                    },
                    {
                        "id": "desert-dawn",
                        "type": "location",
                        "description": "Desert at sunrise",
                        "prompt": "SCENE CONTEXT\nWide desert...\nPOSITIVE LOCKS",
                    },
                ],
                "shots": [
                    {
                        "id": "shot-01",
                        "title": "Pilot checks drone",
                        "duration_seconds": 6,
                        "aspect_ratio": "16:9",
                        "resolution": "720p",
                        "first_frame_asset_id": "hero-pilot",
                        "prompt": "SCENE CONTEXT\nPilot kneels...\nPOSITIVE LOCKS",
                    },
                ],
            }),
        }
    ]
})


def test_load_recipe_exists():
    text = architect.load_recipe()
    assert "16-section" in text
    assert "POSITIVE LOCKS" in text


def test_build_system_prompt_includes_recipe():
    recipe = architect.load_recipe()
    prompt = architect.build_system_prompt(recipe, max_shots=3)
    assert "16-section Seedance recipe" in prompt
    assert "at most 3 shots" in prompt


def test_extract_json_strips_fences():
    assert architect.extract_json("```json\n{}\n```") == "{}"
    assert architect.extract_json("```\n{}\n```") == "{}"
    assert architect.extract_json('{"a":1}') == '{"a":1}'


def test_validate_plan_ok():
    inner = json.loads(SAMPLE_PROXY_RESPONSE)["content"][0]["text"]
    plan = json.loads(architect.extract_json(inner))
    plan = architect.validate_plan(plan)
    assert plan["concept"] == "Drone sunrise teaser"
    assert len(plan["assets"]) == 2
    assert len(plan["shots"]) == 1


def test_validate_plan_missing_key():
    bad = {"concept": "x", "format_mode": "y", "assets": [], "shots": [{"id": "s1"}]}
    with pytest.raises(architect.ArchitectError) as exc:
        architect.validate_plan(bad)
    assert "shot[0] missing key" in str(exc.value)


def test_validate_plan_unknown_first_frame_asset():
    bad = {
        "concept": "x",
        "format_mode": "y",
        "assets": [],
        "shots": [
            {
                "id": "s1",
                "title": "t",
                "duration_seconds": 6,
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "first_frame_asset_id": "missing",
                "prompt": "p" * 120,
            }
        ],
    }
    with pytest.raises(architect.ArchitectError) as exc:
        architect.validate_plan(bad)
    assert "unknown first_frame_asset_id" in str(exc.value)


@patch("architect.urlopen")
def test_run_happy_path(mock_urlopen):
    response = Mock()
    response.read.return_value = SAMPLE_PROXY_RESPONSE.encode("utf-8")
    mock_urlopen.return_value.__enter__ = Mock(return_value=response)
    mock_urlopen.return_value.__exit__ = Mock(return_value=False)

    plan = architect.run("Drone sunrise teaser", proxy_url="http://localhost:9999")
    assert plan["format_mode"] == "cinematic film"
    assert plan["shots"][0]["id"] == "shot-01"


@patch("architect.urlopen")
def test_run_invalid_json(mock_urlopen):
    response = Mock()
    response.read.return_value = b"not json"
    mock_urlopen.return_value.__enter__ = Mock(return_value=response)
    mock_urlopen.return_value.__exit__ = Mock(return_value=False)

    with pytest.raises(architect.ArchitectError) as exc:
        architect.run("x", proxy_url="http://localhost:9999")
    assert "invalid JSON" in str(exc.value)


@patch("architect.urlopen")
def test_run_proxy_unreachable(mock_urlopen):
    from urllib.error import URLError
    mock_urlopen.side_effect = URLError("refused")

    with pytest.raises(architect.ArchitectError) as exc:
        architect.run("x", proxy_url="http://localhost:9999")
    assert "Cannot reach FCC proxy" in str(exc.value)


# FCC proxy streams Anthropic-style SSE. The architect must collect text_delta
# fragments and ignore thinking_delta, then parse the assembled JSON.
SAMPLE_SSE_RESPONSE = "\n".join([
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"m1","content":[]}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me plan."}}',
    '',
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":0}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"{\\"concept\\":"}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"\\"x\\"}"}}',
    '',
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":1}',
    '',
    'event: message_stop',
    'data: {"type":"message_stop"}',
    '',
])


def test_parse_sse_collects_text_deltas_only():
    text = architect._parse_sse(SAMPLE_SSE_RESPONSE)
    assert text == '{"concept":"x"}'


def test_parse_sse_returns_none_for_non_sse():
    assert architect._parse_sse('{"content":[{"type":"text","text":"hi"}]}') is None
    assert architect._parse_sse("plain text response") is None


# The FCC proxy returns a complete JSON envelope followed by a trailing SSE
# diagnostic stream (e.g. "Provider stream ended without message_stop."). The
# architect must take the envelope's text block and ignore the SSE tail.
def _build_hybrid_response() -> str:
    inner_plan_text = json.loads(SAMPLE_PROXY_RESPONSE)["content"][0]["text"]
    envelope = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-flash",
        "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": inner_plan_text},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    sse_tail = (
        "\n\nevent: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Provider stream ended without message_stop."}}\n\n'
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
    )
    return json.dumps(envelope) + sse_tail


SAMPLE_HYBRID_RESPONSE = _build_hybrid_response()


def test_extract_envelope_text_prefers_envelope_over_sse():
    # The envelope text is the real answer; the SSE tail is an error string.
    text = architect._extract_envelope_text(SAMPLE_HYBRID_RESPONSE)
    assert text is not None
    assert "Provider stream ended" not in text
    assert "cinematic film" in text


@patch("architect.urlopen")
def test_run_hybrid_envelope_then_sse(mock_urlopen):
    response = Mock()
    response.read.return_value = SAMPLE_HYBRID_RESPONSE.encode("utf-8")
    mock_urlopen.return_value.__enter__ = Mock(return_value=response)
    mock_urlopen.return_value.__exit__ = Mock(return_value=False)

    plan = architect.run("Drone sunrise teaser", proxy_url="http://localhost:9999")
    assert plan["format_mode"] == "cinematic film"
    assert plan["shots"][0]["id"] == "shot-01"


@patch("architect.urlopen")
def test_run_sse_stream(mock_urlopen):
    # Build an SSE stream carrying the valid plan as text_delta fragments.
    inner = json.loads(SAMPLE_PROXY_RESPONSE)["content"][0]["text"]
    plan_json = architect.extract_json(inner)
    # Split the plan JSON into chunks to exercise multi-delta assembly.
    chunks = [plan_json[i:i + 40] for i in range(0, len(plan_json), 40)]
    sse_lines = [
        'event: message_start',
        'data: {"type":"message_start","message":{"content":[]}}',
        '',
        'event: content_block_start',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        '',
    ]
    for chunk in chunks:
        sse_lines.append('event: content_block_delta')
        sse_lines.append('data: ' + json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        }))
        sse_lines.append('')
    sse_lines += [
        'event: content_block_stop',
        'data: {"type":"content_block_stop","index":0}',
        '',
        'event: message_stop',
        'data: {"type":"message_stop"}',
        '',
    ]
    sse = "\n".join(sse_lines)

    response = Mock()
    response.read.return_value = sse.encode("utf-8")
    mock_urlopen.return_value.__enter__ = Mock(return_value=response)
    mock_urlopen.return_value.__exit__ = Mock(return_value=False)

    plan = architect.run("Drone sunrise teaser", proxy_url="http://localhost:9999")
    assert plan["format_mode"] == "cinematic film"
    assert plan["shots"][0]["id"] == "shot-01"
