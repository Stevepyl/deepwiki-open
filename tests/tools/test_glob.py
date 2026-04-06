"""Tests for api.tools.glob — GlobTool."""

from unittest.mock import patch

import pytest

from api.tools.glob import GlobTool


@pytest.fixture()
def glob(repo_path):
    return GlobTool(repo_path)


# ---------------------------------------------------------------------------
# Basic file matching
# ---------------------------------------------------------------------------

class TestGlobBasic:

    @pytest.mark.asyncio
    async def test_find_python_files(self, glob):
        result = await glob.execute({"pattern": "**/*.py"})
        assert result.metadata["count"] >= 2
        assert "main.py" in result.output
        assert "utils.py" in result.output

    @pytest.mark.asyncio
    async def test_find_tsx_files(self, glob):
        result = await glob.execute({"pattern": "**/*.tsx"})
        assert result.metadata["count"] >= 1
        assert "button.tsx" in result.output

    @pytest.mark.asyncio
    async def test_find_markdown_files(self, glob):
        result = await glob.execute({"pattern": "*.md"})
        assert result.metadata["count"] >= 1
        assert "README.md" in result.output

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty(self, glob):
        result = await glob.execute({"pattern": "**/*.nonexistent"})
        assert result.metadata["count"] == 0
        assert "No files found" in result.output


# ---------------------------------------------------------------------------
# .git exclusion
# ---------------------------------------------------------------------------

class TestGlobGitExclusion:

    @pytest.mark.asyncio
    async def test_does_not_list_dot_git_contents(self, glob):
        result = await glob.execute({"pattern": "**/*"})
        output_lower = result.output.lower()
        assert ".git/head" not in output_lower
        assert ".git\\head" not in output_lower

    @pytest.mark.asyncio
    async def test_lists_dot_github_contents(self, glob):
        result = await glob.execute({"pattern": "**/*.yml"})
        assert result.metadata["count"] >= 1
        assert ".github" in result.output


# ---------------------------------------------------------------------------
# Subdirectory search
# ---------------------------------------------------------------------------

class TestGlobSubdir:

    @pytest.mark.asyncio
    async def test_relative_path_restricts_search(self, glob):
        result = await glob.execute({
            "pattern": "**/*.py",
            "path": "src",
        })
        assert result.metadata["count"] >= 1
        # Should find files but only within src/
        for line in result.output.splitlines():
            if line.strip() and not line.startswith("("):
                assert "src" in line


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

class TestGlobPathTraversal:

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, glob, fake_repo):
        result = await glob.execute({
            "pattern": "**/*",
            "path": str(fake_repo / ".." / "escape"),
        })
        assert result.metadata.get("error") == "path_traversal"

    @pytest.mark.asyncio
    async def test_absolute_path_within_repo_works(self, glob, fake_repo):
        abs_src = str(fake_repo / "src")
        result = await glob.execute({
            "pattern": "*.py",
            "path": abs_src,
        })
        assert result.metadata["count"] >= 1


# ---------------------------------------------------------------------------
# stdlib fallback
# ---------------------------------------------------------------------------

class TestGlobStdlibFallback:

    @pytest.mark.asyncio
    async def test_pathlib_finds_python_files(self, glob):
        with patch("api.tools.glob.find_ripgrep", return_value=None):
            result = await glob.execute({"pattern": "*.py"})
        assert result.metadata["count"] >= 2

    @pytest.mark.asyncio
    async def test_pathlib_excludes_dot_git(self, glob):
        with patch("api.tools.glob.find_ripgrep", return_value=None):
            result = await glob.execute({"pattern": "*"})
        assert ".git" not in result.output or ".github" in result.output
        # More precise: .git/HEAD should never appear
        for line in result.output.splitlines():
            parts = line.strip().split("/")
            # No path segment should be exactly ".git" followed by a child
            segments = [p for p in parts if p == ".git"]
            if segments:
                # This line mentions .git — it must be inside .github, not .git itself
                assert ".github" in line or ".gitignore" in line

    @pytest.mark.asyncio
    async def test_pathlib_includes_dot_github(self, glob):
        with patch("api.tools.glob.find_ripgrep", return_value=None):
            result = await glob.execute({"pattern": "*.yml"})
        assert ".github" in result.output


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestGlobTruncation:

    @pytest.mark.asyncio
    async def test_truncation_over_max_files(self, tmp_path):
        """Create > MAX_FILES (100) files and verify truncation."""
        for i in range(110):
            (tmp_path / f"file_{i:03d}.txt").write_text(f"content {i}")

        glob = GlobTool(str(tmp_path))
        result = await glob.execute({"pattern": "*.txt"})
        assert result.metadata["truncated"] is True
        assert result.metadata["count"] == 110
        assert "truncated" in result.output.lower()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestGlobSchema:

    def test_function_schema_structure(self, glob):
        schema = glob.to_function_schema()
        assert schema["function"]["name"] == "glob"
        props = schema["function"]["parameters"]["properties"]
        assert "pattern" in props
        assert "path" in props
        assert "pattern" in schema["function"]["parameters"]["required"]
