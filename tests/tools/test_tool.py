"""Tests for api.tools.tool — shared utilities and base classes."""

import os

import pytest

from api.tools.tool import (
    ToolResult,
    find_ripgrep,
    load_description,
    truncate_output,
    validate_path_within_repo,
)


# ---------------------------------------------------------------------------
# truncate_output
# ---------------------------------------------------------------------------

class TestTruncateOutput:

    def test_short_text_unchanged(self):
        text = "line1\nline2\nline3"
        assert truncate_output(text) == text

    def test_truncates_by_line_count(self):
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        result = truncate_output(text, max_lines=10, max_bytes=999_999)
        assert "...90 lines truncated" in result
        # First 10 lines should still be present
        assert "line 0" in result
        assert "line 9" in result

    def test_truncates_by_byte_limit(self):
        # Each line is ~10 bytes; 50 bytes should keep ~5 lines
        lines = [f"aaaaaaaaaa" for _ in range(20)]
        text = "\n".join(lines)
        result = truncate_output(text, max_lines=9999, max_bytes=50)
        assert "truncated" in result

    def test_empty_string(self):
        assert truncate_output("") == ""


# ---------------------------------------------------------------------------
# validate_path_within_repo
# ---------------------------------------------------------------------------

class TestValidatePathWithinRepo:

    def test_valid_subpath(self, tmp_path):
        repo = str(tmp_path)
        sub = str(tmp_path / "src")
        (tmp_path / "src").mkdir()
        result = validate_path_within_repo(sub, repo)
        assert result.endswith("src")

    def test_repo_root_itself_is_valid(self, tmp_path):
        repo = str(tmp_path)
        result = validate_path_within_repo(repo, repo)
        assert os.path.isabs(result)

    def test_rejects_parent_traversal(self, tmp_path):
        repo = str(tmp_path / "myrepo")
        (tmp_path / "myrepo").mkdir()
        with pytest.raises(ValueError, match="outside"):
            validate_path_within_repo(str(tmp_path / "myrepo" / ".." / "other"), repo)

    def test_rejects_absolute_escape(self, tmp_path):
        repo = str(tmp_path / "myrepo")
        (tmp_path / "myrepo").mkdir()
        with pytest.raises(ValueError, match="outside"):
            validate_path_within_repo("/tmp", repo)

    def test_rejects_sibling_with_prefix(self, tmp_path):
        """Ensure /repo-other is NOT accepted as child of /repo."""
        repo = tmp_path / "repo"
        sibling = tmp_path / "repo-other"
        repo.mkdir()
        sibling.mkdir()
        with pytest.raises(ValueError, match="outside"):
            validate_path_within_repo(str(sibling), str(repo))


# ---------------------------------------------------------------------------
# find_ripgrep
# ---------------------------------------------------------------------------

class TestFindRipgrep:

    def test_returns_string_or_none(self):
        result = find_ripgrep()
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# load_description
# ---------------------------------------------------------------------------

class TestLoadDescription:

    def test_substitutes_variables(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("OS: ${os}, Shell: ${shell}, Lines: ${maxLines}, Bytes: ${maxBytes}")
        result = load_description(str(txt))
        assert "${os}" not in result
        assert "${shell}" not in result
        assert "${maxLines}" not in result
        assert "${maxBytes}" not in result


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:

    def test_default_metadata_is_empty_dict(self):
        r = ToolResult(title="test", output="ok")
        assert r.metadata == {}

    def test_metadata_not_shared_across_instances(self):
        r1 = ToolResult(title="a", output="1")
        r2 = ToolResult(title="b", output="2")
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata
