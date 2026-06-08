"""Tests for NimSettings validators in config/nim.py"""

import pytest
from pydantic import ValidationError

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from config.nim import NimSettings


class TestNimSettingsValidators:
    """Test custom field validators in NimSettings."""

    @pytest.mark.parametrize(
        "field,invalid_val",
        [
            ("max_tokens", "abc"),
            ("max_tokens", [1, 2, 3]),
            ("max_tokens", {"key": "val"}),
            ("min_tokens", "not-an-int"),
            ("min_tokens", [0]),
        ],
    )
    def test_validate_int_fields_error(self, field, invalid_val):
        """Test that validate_int_fields raises ValueError for invalid types/values."""
        with pytest.raises(ValidationError) as excinfo:
            NimSettings(**{field: invalid_val})

        # Check that the error message is what we expect from config/nim.py
        # Pydantic wraps the ValueError, so we check the string representation of the errors
        assert f"{field} must be an int. Got {type(invalid_val).__name__}." in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        "field,invalid_val",
        [
            ("temperature", "hot"),
            ("temperature", [1.0]),
            ("top_p", "maybe"),
            ("presence_penalty", {"penalty": 0.5}),
        ],
    )
    def test_validate_float_fields_error(self, field, invalid_val):
        """Test that validate_float_fields raises ValueError for invalid types/values."""
        with pytest.raises(ValidationError) as excinfo:
            NimSettings(**{field: invalid_val})

        assert f"{field} must be a float. Got {type(invalid_val).__name__}." in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        "field,val,expected",
        [
            ("max_tokens", None, ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS),
            ("max_tokens", "", ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS),
            ("max_tokens", "100", 100),
            ("min_tokens", None, 0),
            ("min_tokens", "", 0),
            ("min_tokens", 10, 10),
        ],
    )
    def test_validate_int_fields_success(self, field, val, expected):
        """Test happy paths and defaults for validate_int_fields."""
        s = NimSettings(**{field: val})
        assert getattr(s, field) == expected

    @pytest.mark.parametrize(
        "field,val,expected",
        [
            ("temperature", None, 1.0),
            ("temperature", "", 1.0),
            ("temperature", "0.5", 0.5),
            ("top_p", None, 1.0),
            ("top_p", "", 1.0),
            ("presence_penalty", "0.0", 0.0),
        ],
    )
    def test_validate_float_fields_success(self, field, val, expected):
        """Test happy paths and defaults for validate_float_fields."""
        s = NimSettings(**{field: val})
        assert getattr(s, field) == expected

    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, -1),
            ("", -1),
            ("5", 5),
            (10, 10),
        ],
    )
    def test_validate_top_k(self, val, expected):
        """Test validate_top_k logic."""
        s = NimSettings(top_k=val)
        assert s.top_k == expected

    def test_validate_top_k_error(self):
        """Test validate_top_k raises error for out of range."""
        with pytest.raises(ValidationError, match="top_k must be -1 or >= 0"):
            NimSettings(top_k=-2)

    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, None),
            ("", None),
            ("123", 123),
        ],
    )
    def test_parse_optional_int(self, val, expected):
        """Test parse_optional_int logic."""
        s = NimSettings(seed=val)
        assert s.seed == expected

    def test_parse_optional_int_error(self):
        """Test parse_optional_int raises error for invalid input."""
        with pytest.raises(ValidationError, match="seed must be an int or empty/None"):
            NimSettings(seed="abc")
