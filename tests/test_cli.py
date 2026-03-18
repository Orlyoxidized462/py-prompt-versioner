"""
Unit tests for py_prompt_versioner.cli

Uses typer.testing.CliRunner so no real filesystem side-effects bleed out
(the runner is given a temporary directory via pytest's tmp_path fixture
and the CLI commands operate within it using the --path option).
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from py_prompt_versioner.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


class TestVersionCommand:
    """Tests for the `version` CLI command."""

    def test_version_exits_zero(self):
        """version command exits with code 0."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_output_contains_version_string(self):
        """version command prints the version string."""
        result = runner.invoke(app, ["version"])
        assert "v0.1.0" in result.output

    def test_version_output_contains_app_name(self):
        """version command output mentions the application name."""
        result = runner.invoke(app, ["version"])
        assert "Prompt-Versioner" in result.output


# ---------------------------------------------------------------------------
# init command — happy paths
# ---------------------------------------------------------------------------


class TestInitCommandValid:
    """Tests for the `init` CLI command with valid inputs."""

    def test_init_default_path_exits_zero(self, tmp_path):
        """init with --path resolves without error (exit code 0)."""
        result = runner.invoke(app, ["init", "--path", str(tmp_path / "prompts")])
        assert result.exit_code == 0

    def test_init_creates_sample_dir(self, tmp_path):
        """init creates the sample_task subdirectory."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert (prompts_dir / "sample_task").is_dir()

    def test_init_creates_v1_md_file(self, tmp_path):
        """init creates the v1.md boilerplate prompt file."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert (prompts_dir / "sample_task" / "v1.md").is_file()

    def test_init_v1_md_contains_frontmatter(self, tmp_path):
        """The created v1.md contains YAML frontmatter with expected keys."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        content = (prompts_dir / "sample_task" / "v1.md").read_text()
        assert "version: v1" in content
        assert "model: gpt-4o" in content
        assert "temperature:" in content

    def test_init_v1_md_contains_jinja_placeholder(self, tmp_path):
        """The created v1.md contains the Jinja2 {{ text }} placeholder."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        content = (prompts_dir / "sample_task" / "v1.md").read_text()
        assert "{{ text }}" in content

    def test_init_output_mentions_success(self, tmp_path):
        """init output contains a success message on first run."""
        prompts_dir = tmp_path / "prompts"
        result = runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert "Created" in result.output or "Success" in result.output

    def test_init_output_mentions_prompt_manager(self, tmp_path):
        """init output tells the user how to use PromptManager."""
        prompts_dir = tmp_path / "prompts"
        result = runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert "PromptManager" in result.output

    def test_init_custom_path_short_flag(self, tmp_path):
        """init accepts -p as a shorthand for --path."""
        custom = tmp_path / "my_prompts"
        result = runner.invoke(app, ["init", "-p", str(custom)])
        assert result.exit_code == 0
        assert (custom / "sample_task" / "v1.md").is_file()

    def test_init_nested_path_created(self, tmp_path):
        """init creates nested directories (parents=True behaviour)."""
        nested = tmp_path / "a" / "b" / "c"
        result = runner.invoke(app, ["init", "--path", str(nested)])
        assert result.exit_code == 0
        assert nested.is_dir()


# ---------------------------------------------------------------------------
# init command — second run (idempotent)
# ---------------------------------------------------------------------------


class TestInitCommandIdempotent:
    """Tests for running `init` when prompts directory already exists."""

    def test_init_twice_exits_zero(self, tmp_path):
        """Running init a second time still exits with code 0."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        result = runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert result.exit_code == 0

    def test_init_twice_does_not_overwrite_existing_file(self, tmp_path):
        """Second init does not overwrite the existing v1.md."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])

        # Modify the file
        v1 = prompts_dir / "sample_task" / "v1.md"
        v1.write_text("custom content")

        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert v1.read_text() == "custom content"

    def test_init_twice_warns_already_exists(self, tmp_path):
        """Second init prints a warning that the sample already exists."""
        prompts_dir = tmp_path / "prompts"
        runner.invoke(app, ["init", "--path", str(prompts_dir)])
        result = runner.invoke(app, ["init", "--path", str(prompts_dir)])
        assert "already exists" in result.output or "!" in result.output


# ---------------------------------------------------------------------------
# init command — error paths
# ---------------------------------------------------------------------------


class TestInitCommandErrors:
    """Tests for `init` failure modes."""

    def test_init_path_is_file_exits_nonzero(self, tmp_path):
        """init exits with non-zero code when --path points to an existing file."""
        existing_file = tmp_path / "iam_a_file"
        existing_file.write_text("I am a file, not a directory")
        result = runner.invoke(app, ["init", "--path", str(existing_file)])
        assert result.exit_code != 0

    def test_init_path_is_file_shows_error_message(self, tmp_path):
        """init prints an error message when --path is an existing file."""
        existing_file = tmp_path / "iam_a_file"
        existing_file.write_text("I am a file")
        result = runner.invoke(app, ["init", "--path", str(existing_file)])
        assert "Error" in result.output or "not a directory" in result.output
