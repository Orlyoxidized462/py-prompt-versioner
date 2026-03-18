"""
Unit tests for py_prompt_versioner.models.PromptMetadata
"""

import pytest
from pydantic import ValidationError

from py_prompt_versioner.models import PromptMetadata


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestPromptMetadataValid:
    """Tests for valid PromptMetadata instantiation."""

    def test_all_required_fields(self):
        """Model instantiates successfully when all required fields are provided."""
        meta = PromptMetadata(version="v1", model="gpt-4o")
        assert meta.version == "v1"
        assert meta.model == "gpt-4o"

    def test_default_temperature(self):
        """temperature defaults to 0.7 when not provided."""
        meta = PromptMetadata(version="v1", model="gpt-4o")
        assert meta.temperature == 0.7

    def test_explicit_temperature(self):
        """temperature is stored correctly when explicitly provided."""
        meta = PromptMetadata(version="v1", model="gpt-4o", temperature=0.3)
        assert meta.temperature == 0.3

    def test_additional_metadata_default_empty(self):
        """additional_metadata defaults to an empty dict when not provided."""
        meta = PromptMetadata(version="v1", model="gpt-4o")
        assert meta.additional_metadata == {}

    def test_additional_metadata_provided(self):
        """additional_metadata stores arbitrary key-value pairs."""
        extra = {"top_p": 0.9, "max_tokens": 512, "notes": "draft"}
        meta = PromptMetadata(version="v2", model="gpt-3.5-turbo", additional_metadata=extra)
        assert meta.additional_metadata == extra

    def test_temperature_zero(self):
        """temperature = 0.0 (deterministic) is a valid value."""
        meta = PromptMetadata(version="v1", model="gpt-4o", temperature=0.0)
        assert meta.temperature == 0.0

    def test_temperature_one(self):
        """temperature = 1.0 is a valid boundary value."""
        meta = PromptMetadata(version="v1", model="gpt-4o", temperature=1.0)
        assert meta.temperature == 1.0

    def test_version_arbitrary_string(self):
        """version accepts arbitrary non-empty strings."""
        meta = PromptMetadata(version="release-2024-q1", model="claude-3")
        assert meta.version == "release-2024-q1"


# ---------------------------------------------------------------------------
# Validation / error-path tests
# ---------------------------------------------------------------------------


class TestPromptMetadataInvalid:
    """Tests for invalid PromptMetadata inputs that should raise ValidationError."""

    def test_missing_version_raises(self):
        """Omitting required field 'version' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptMetadata(model="gpt-4o")
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "version" in field_names

    def test_missing_model_raises(self):
        """Omitting required field 'model' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PromptMetadata(version="v1")
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "model" in field_names

    def test_missing_both_required_fields(self):
        """Omitting both required fields raises ValidationError with two errors."""
        with pytest.raises(ValidationError) as exc_info:
            PromptMetadata()
        errors = exc_info.value.errors()
        assert len(errors) >= 2

    def test_invalid_temperature_non_numeric(self):
        """Providing a non-numeric string for temperature raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptMetadata(version="v1", model="gpt-4o", temperature="hot")

    def test_none_version_raises(self):
        """Passing None for version raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptMetadata(version=None, model="gpt-4o")

    def test_none_model_raises(self):
        """Passing None for model raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptMetadata(version="v1", model=None)
