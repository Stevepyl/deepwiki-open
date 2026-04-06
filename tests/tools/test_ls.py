"""Tests for api.tools.ls -- ListTool."""

from unittest.mock import patch

import pytest

from api.tools.ls import ListTool


@pytest.fixture()
def ls(repo_path):
    return ListTool(repo_path)


# ---------------------------------------------------------------------------
# Basic tree output
# ---------------------------------------------------------------------------

class TestLsBasic:

    @pytest.mark.asyncio
    async def test_lists_repo_root(self, ls, fake_repo):
        result = await ls.execute({})
        assert result.metadata["count"] >= 4
        assert "main.py" in result.output
        assert "README.md" in result.output

    @pytest.mark.asyncio
    async def test_tree_shows_directory_structure(self, ls, fake_repo):
        result = await ls.execute({})
        # Tree output should contain directory markers (trailing /)
        assert "src/" in result.output

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        tool = ListTool(str(tmp_path))
        result = await tool.execute({"path": str(empty)})
        assert "(empty)" in result.output
        assert result.metadata["count"] == 0


# ---------------------------------------------------------------------------
# .git exclusion
# ---------------------------------------------------------------------------

class TestLsGitExclusion:

    @pytest.mark.asyncio
    async def test_does_not_list_dot_git_contents(self, ls):
        result = await ls.execute({})
        output_lower = result.output.lower()
        assert ".git/head" not in output_lower
        assert ".git\\head" not in output_lower

    @pytest.mark.asyncio
    async def test_lists_dot_github_contents(self, ls):
        result = await ls.execute({})
        assert ".github" in result.output


# ---------------------------------------------------------------------------
# Subdirectory listing
# ---------------------------------------------------------------------------

class TestLsSubdir:

    @pytest.mark.asyncio
    async def test_relative_path_restricts_listing(self, ls):
        result = await ls.execute({"path": "src"})
        assert "main.py" in result.output
        # Should not include root-level files
        assert "README.md" not in result.output

    @pytest.mark.asyncio
    async def test_absolute_path_within_repo(self, ls, fake_repo):
        abs_src = str(fake_repo / "src")
        result = await ls.execute({"path": abs_src})
        assert "main.py" in result.output


# ---------------------------------------------------------------------------
# Not-a-directory
# ---------------------------------------------------------------------------

class TestLsNotDirectory:

    @pytest.mark.asyncio
    async def test_file_path_returns_error(self, ls, fake_repo):
        result = await ls.execute({"path": str(fake_repo / "README.md")})
        assert result.metadata.get("error") == "not_directory"
        assert "not a directory" in result.output.lower()


# ---------------------------------------------------------------------------
# Extra ignore patterns
# ---------------------------------------------------------------------------

class TestLsIgnore:

    @pytest.mark.asyncio
    async def test_extra_ignore_excludes_files(self, ls):
        result = await ls.execute({"ignore": ["*.md"]})
        assert "README.md" not in result.output


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

class TestLsPathTraversal:

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, ls, fake_repo):
        result = await ls.execute({
            "path": str(fake_repo / ".." / "escape"),
        })
        assert result.metadata.get("error") == "path_traversal"

    @pytest.mark.asyncio
    async def test_absolute_path_outside_repo_blocked(self, ls):
        result = await ls.execute({"path": "/tmp"})
        assert result.metadata.get("error") == "path_traversal"


# ---------------------------------------------------------------------------
# stdlib fallback
# ---------------------------------------------------------------------------

class TestLsStdlibFallback:

    @pytest.mark.asyncio
    async def test_pathlib_lists_files(self, ls):
        with patch("api.tools.ls.find_ripgrep", return_value=None):
            result = await ls.execute({})
        assert result.metadata["count"] >= 4
        assert "main.py" in result.output

    @pytest.mark.asyncio
    async def test_pathlib_excludes_dot_git(self, ls):
        with patch("api.tools.ls.find_ripgrep", return_value=None):
            result = await ls.execute({})
        lines = result.output.lower().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("("):
                assert stripped != "head" or ".github" in result.output


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestLsTruncation:

    @pytest.mark.asyncio
    async def test_truncation_over_limit(self, tmp_path):
        """Create > LIMIT (100) files and verify truncation."""
        for i in range(110):
            (tmp_path / f"file_{i:03d}.txt").write_text(f"content {i}")

        tool = ListTool(str(tmp_path))
        result = await tool.execute({})
        assert result.metadata["truncated"] is True
        assert "truncated" in result.output.lower()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestLsSchema:

    def test_function_schema_structure(self, ls):
        schema = ls.to_function_schema()
        assert schema["function"]["name"] == "ls"
        props = schema["function"]["parameters"]["properties"]
        assert "path" in props
        assert "ignore" in props
