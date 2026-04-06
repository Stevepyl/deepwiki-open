"""Tests for api.tools.grep — GrepTool."""

from unittest.mock import patch

import pytest

from api.tools.grep import GrepTool, _grep_with_stdlib


@pytest.fixture()
def grep(repo_path):
    return GrepTool(repo_path)


# ---------------------------------------------------------------------------
# Basic search (uses whichever backend is available)
# ---------------------------------------------------------------------------

class TestGrepBasic:

    @pytest.mark.asyncio
    async def test_finds_pattern_in_python_files(self, grep):
        result = await grep.execute({"pattern": "def main"})
        assert result.metadata["matches"] >= 1
        assert "main.py" in result.output

    @pytest.mark.asyncio
    async def test_finds_pattern_with_include_filter(self, grep):
        result = await grep.execute({
            "pattern": "def",
            "include": "*.py",
        })
        assert result.metadata["matches"] >= 1
        # Should NOT match .tsx files
        assert "button.tsx" not in result.output

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty(self, grep):
        result = await grep.execute({"pattern": "xyzzy_nonexistent_pattern_42"})
        assert result.metadata["matches"] == 0
        assert "No files found" in result.output

    @pytest.mark.asyncio
    async def test_regex_syntax_works(self, grep):
        result = await grep.execute({"pattern": r"def\s+\w+\("})
        assert result.metadata["matches"] >= 1

    @pytest.mark.asyncio
    async def test_invalid_regex_returns_error(self, grep):
        result = await grep.execute({"pattern": "[invalid"})
        assert "error" in result.output.lower()


# ---------------------------------------------------------------------------
# .git exclusion
# ---------------------------------------------------------------------------

class TestGrepGitExclusion:

    @pytest.mark.asyncio
    async def test_does_not_search_inside_dot_git(self, grep):
        """The .git/HEAD file contains 'ref:' — must NOT appear in results."""
        result = await grep.execute({"pattern": "refs/heads/main"})
        assert ".git" not in result.output

    @pytest.mark.asyncio
    async def test_searches_hidden_dirs_like_github(self, grep):
        """Files in .github/ should be found (--hidden includes them)."""
        result = await grep.execute({"pattern": "ubuntu-latest"})
        assert result.metadata["matches"] >= 1
        assert ".github" in result.output


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

class TestGrepPathTraversal:

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, grep, fake_repo):
        result = await grep.execute({
            "pattern": "anything",
            "path": str(fake_repo / ".." / "escape"),
        })
        assert result.metadata.get("error") == "path_traversal"

    @pytest.mark.asyncio
    async def test_relative_subdir_path_works(self, grep):
        result = await grep.execute({
            "pattern": "def",
            "path": "src",
        })
        assert result.metadata["matches"] >= 1


# ---------------------------------------------------------------------------
# stdlib fallback
# ---------------------------------------------------------------------------

class TestGrepStdlibFallback:
    """Test the Python stdlib fallback path by forcing ripgrep to be "not found"."""

    @pytest.mark.asyncio
    async def test_stdlib_finds_matches(self, grep):
        with patch("api.tools.grep.find_ripgrep", return_value=None):
            result = await grep.execute({"pattern": "def main"})
        assert result.metadata["matches"] >= 1
        assert "main.py" in result.output

    @pytest.mark.asyncio
    async def test_stdlib_respects_include_filter(self, grep):
        with patch("api.tools.grep.find_ripgrep", return_value=None):
            result = await grep.execute({
                "pattern": "def",
                "include": "*.py",
            })
        assert "button.tsx" not in result.output

    @pytest.mark.asyncio
    async def test_stdlib_excludes_dot_git(self, grep):
        with patch("api.tools.grep.find_ripgrep", return_value=None):
            result = await grep.execute({"pattern": "refs/heads/main"})
        assert ".git" not in result.output

    @pytest.mark.asyncio
    async def test_stdlib_includes_dot_github(self, grep):
        with patch("api.tools.grep.find_ripgrep", return_value=None):
            result = await grep.execute({"pattern": "ubuntu-latest"})
        assert result.metadata["matches"] >= 1
        assert ".github" in result.output


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestGrepOutput:

    @pytest.mark.asyncio
    async def test_output_contains_line_numbers(self, grep):
        result = await grep.execute({"pattern": "def main"})
        assert "Line " in result.output

    @pytest.mark.asyncio
    async def test_output_grouped_by_file(self, grep):
        result = await grep.execute({"pattern": "def"})
        # Multiple matches in main.py should appear under one header
        lines = result.output.splitlines()
        file_headers = [l for l in lines if l.endswith(":") and "Line" not in l]
        assert len(file_headers) >= 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestGrepSchema:

    def test_function_schema_structure(self, grep):
        schema = grep.to_function_schema()
        assert schema["function"]["name"] == "grep"
        props = schema["function"]["parameters"]["properties"]
        assert "pattern" in props
        assert "path" in props
        assert "include" in props
