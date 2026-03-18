"""
Unit tests for py_prompt_versioner.core.PromptManager
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from py_prompt_versioner.core import PromptManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_FRONTMATTER = (
    "---\n"
    "version: v1\n"
    "model: gpt-4o\n"
    "temperature: 0.7\n"
    "---\n"
    "\n"
    "Summarize this: {{ text }}"
)

VALID_FRONTMATTER_NO_VARS = (
    "---\n"
    "version: v2\n"
    "model: claude-3\n"
    "temperature: 0.5\n"
    "---\n"
    "\n"
    "You are a helpful assistant."
)

INVALID_FRONTMATTER_MISSING_MODEL = (
    "---\n"
    "version: v1\n"
    "temperature: 0.7\n"
    "---\n"
    "\n"
    "Some content."
)


def _make_prompt_file(tmp_path: Path, task: str, version: str, content: str) -> Path:
    """Create a prompt .md file inside tmp_path and return the PromptManager base path."""
    task_dir = tmp_path / task
    task_dir.mkdir(parents=True)
    (task_dir / f"{version}.md").write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# PromptManager initialisation
# ---------------------------------------------------------------------------


class TestPromptManagerInit:
    """Tests for PromptManager.__init__"""

    def test_default_base_path(self):
        """Default base_path resolves to './prompts'."""
        pm = PromptManager()
        assert pm.base_path == Path("./prompts")

    def test_custom_base_path_string(self, tmp_path):
        """Custom string path is stored as a Path object."""
        pm = PromptManager(path=str(tmp_path))
        assert pm.base_path == tmp_path

    def test_custom_base_path_path_object(self, tmp_path):
        """Passing a Path object as path is also accepted."""
        pm = PromptManager(path=tmp_path)
        assert pm.base_path == tmp_path


# ---------------------------------------------------------------------------
# get_prompt — happy-path tests
# ---------------------------------------------------------------------------


class TestGetPromptValid:
    """Tests for PromptManager.get_prompt() with valid inputs."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        """Result dict has 'metadata' and 'content' keys."""
        base = _make_prompt_file(tmp_path, "summarise", "v1", VALID_FRONTMATTER)
        pm = PromptManager(path=base)
        result = pm.get_prompt("summarise", "v1", variables={"text": "hello"})
        assert "metadata" in result
        assert "content" in result

    def test_metadata_fields_are_correct(self, tmp_path):
        """Metadata parsed from frontmatter matches expected values."""
        base = _make_prompt_file(tmp_path, "summarise", "v1", VALID_FRONTMATTER)
        pm = PromptManager(path=base)
        result = pm.get_prompt("summarise", "v1", variables={"text": "hello"})
        meta = result["metadata"]
        assert meta.version == "v1"
        assert meta.model == "gpt-4o"
        assert meta.temperature == 0.7

    def test_jinja2_variable_injection(self, tmp_path):
        """Jinja2 template variables are rendered into the prompt body."""
        base = _make_prompt_file(tmp_path, "summarise", "v1", VALID_FRONTMATTER)
        pm = PromptManager(path=base)
        result = pm.get_prompt("summarise", "v1", variables={"text": "the quick brown fox"})
        assert "the quick brown fox" in result["content"]
        assert "{{ text }}" not in result["content"]

    def test_no_variables_renders_cleanly(self, tmp_path):
        """Calling without variables on a template with no placeholders works fine."""
        base = _make_prompt_file(tmp_path, "assistant", "v2", VALID_FRONTMATTER_NO_VARS)
        pm = PromptManager(path=base)
        result = pm.get_prompt("assistant", "v2")
        assert "You are a helpful assistant." in result["content"]

    def test_none_variables_renders_cleanly(self, tmp_path):
        """Passing variables=None is equivalent to passing no variables."""
        base = _make_prompt_file(tmp_path, "assistant", "v2", VALID_FRONTMATTER_NO_VARS)
        pm = PromptManager(path=base)
        result = pm.get_prompt("assistant", "v2", variables=None)
        assert result["content"].strip() != ""

    def test_version_in_content_can_be_any_string(self, tmp_path):
        """Version identifier in filename is arbitrary (e.g. 'v10', 'latest')."""
        content = (
            "---\n"
            "version: latest\n"
            "model: gpt-4o\n"
            "temperature: 0.0\n"
            "---\n"
            "\n"
            "Final prompt."
        )
        base = _make_prompt_file(tmp_path, "task", "latest", content)
        pm = PromptManager(path=base)
        result = pm.get_prompt("task", "latest")
        assert result["metadata"].version == "latest"

    def test_extra_frontmatter_fields_ignored_in_metadata(self, tmp_path):
        """Extra YAML keys not declared on PromptMetadata are silently ignored by Pydantic.

        PromptMetadata does not define a catch-all validator, so unknown fields
        do not populate additional_metadata — they are discarded. The model
        should still instantiate successfully and additional_metadata stays empty.
        """
        content = (
            "---\n"
            "version: v1\n"
            "model: gpt-4o\n"
            "temperature: 0.7\n"
            "author: alice\n"
            "---\n"
            "\n"
            "Hello."
        )
        base = _make_prompt_file(tmp_path, "extra", "v1", content)
        pm = PromptManager(path=base)
        result = pm.get_prompt("extra", "v1")
        # Pydantic ignores undeclared fields; additional_metadata is not auto-populated.
        assert result["metadata"].version == "v1"
        assert result["metadata"].additional_metadata == {}

    def test_multiple_variables_injected(self, tmp_path):
        """Multiple Jinja2 variables are all rendered correctly."""
        content = (
            "---\n"
            "version: v1\n"
            "model: gpt-4o\n"
            "temperature: 0.7\n"
            "---\n"
            "\n"
            "Dear {{ name }}, please review {{ doc }}."
        )
        base = _make_prompt_file(tmp_path, "letter", "v1", content)
        pm = PromptManager(path=base)
        result = pm.get_prompt("letter", "v1", variables={"name": "Bob", "doc": "PR #42"})
        assert "Dear Bob" in result["content"]
        assert "PR #42" in result["content"]


# ---------------------------------------------------------------------------
# get_prompt — error-path tests
# ---------------------------------------------------------------------------


class TestGetPromptErrors:
    """Tests for PromptManager.get_prompt() failure modes."""

    def test_file_not_found_raises(self, tmp_path):
        """Non-existent prompt path raises FileNotFoundError."""
        pm = PromptManager(path=tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            pm.get_prompt("nonexistent_task", "v99")
        assert "nonexistent_task" in str(exc_info.value) or "v99" in str(exc_info.value)

    def test_missing_model_in_frontmatter_raises_validation_error(self, tmp_path):
        """Frontmatter missing required 'model' field raises Pydantic ValidationError."""
        base = _make_prompt_file(tmp_path, "bad_task", "v1", INVALID_FRONTMATTER_MISSING_MODEL)
        pm = PromptManager(path=base)
        with pytest.raises(ValidationError):
            pm.get_prompt("bad_task", "v1")

    def test_wrong_task_name_raises(self, tmp_path):
        """Correct base path but wrong task name raises FileNotFoundError."""
        _make_prompt_file(tmp_path, "real_task", "v1", VALID_FRONTMATTER)
        pm = PromptManager(path=tmp_path)
        with pytest.raises(FileNotFoundError):
            pm.get_prompt("wrong_task", "v1")

    def test_wrong_version_raises(self, tmp_path):
        """Correct task name but wrong version raises FileNotFoundError."""
        _make_prompt_file(tmp_path, "my_task", "v1", VALID_FRONTMATTER)
        pm = PromptManager(path=tmp_path)
        with pytest.raises(FileNotFoundError):
            pm.get_prompt("my_task", "v99")
