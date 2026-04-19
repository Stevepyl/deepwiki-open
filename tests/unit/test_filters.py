"""
Unit tests for api/utils/filters.py and api/agent/filtered_tools.py.

Tests cover:
- _parse_filter_string: null, empty, URLs, multi-line
- ParsedFilters.from_strings and .use_inclusion_mode / .is_empty
- should_exclude_path: exclusion mode (dir segment, file exact match, nested)
- should_exclude_path: inclusion mode inversion
- FilteredToolWrapper: read pre-check, glob post-filter
- wrap_tools_with_filters: empty filters → passthrough
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.agent.filtered_tools import FilteredToolWrapper, wrap_tools_with_filters
from api.tools.tool import ToolResult
from api.utils.filters import ParsedFilters, _parse_filter_string, should_exclude_path


# ---------------------------------------------------------------------------
# _parse_filter_string
# ---------------------------------------------------------------------------


class TestParseFilterString:
    def test_none_returns_empty(self):
        assert _parse_filter_string(None) == ()

    def test_empty_string_returns_empty(self):
        assert _parse_filter_string("") == ()

    def test_whitespace_only_returns_empty(self):
        assert _parse_filter_string("   \n   \n  ") == ()

    def test_single_value(self):
        assert _parse_filter_string("node_modules") == ("node_modules",)

    def test_multi_line_values(self):
        result = _parse_filter_string("node_modules\n.git\ndist")
        assert set(result) == {"node_modules", ".git", "dist"}

    def test_strips_whitespace_around_items(self):
        result = _parse_filter_string("  node_modules  \n  .git  ")
        assert "node_modules" in result
        assert ".git" in result

    def test_url_encoded_value(self):
        result = _parse_filter_string("my%20dir")
        assert "my dir" in result

    def test_skips_blank_lines(self):
        result = _parse_filter_string("a\n\n\nb")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ParsedFilters
# ---------------------------------------------------------------------------


class TestParsedFilters:
    def test_empty(self):
        f = ParsedFilters.empty()
        assert f.is_empty
        assert not f.use_inclusion_mode

    def test_exclusion_mode(self):
        f = ParsedFilters.from_strings(excluded_dirs="node_modules")
        assert not f.is_empty
        assert not f.use_inclusion_mode

    def test_inclusion_mode_via_included_dirs(self):
        f = ParsedFilters.from_strings(included_dirs="src")
        assert f.use_inclusion_mode

    def test_inclusion_mode_via_included_files(self):
        f = ParsedFilters.from_strings(included_files=".py")
        assert f.use_inclusion_mode

    def test_from_strings_parses_all_fields(self):
        f = ParsedFilters.from_strings(
            excluded_dirs="dist",
            excluded_files="package-lock.json",
            included_dirs="src",
            included_files=".py",
        )
        assert "dist" in f.excluded_dirs
        assert "package-lock.json" in f.excluded_files
        assert "src" in f.included_dirs
        assert ".py" in f.included_files

    def test_frozen_immutable(self):
        f = ParsedFilters.from_strings(excluded_dirs="a")
        with pytest.raises((AttributeError, TypeError)):
            f.excluded_dirs = ("b",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# should_exclude_path — exclusion mode
# ---------------------------------------------------------------------------


class TestShouldExcludePathExclusionMode:
    def _excl(self, dirs="", files="") -> ParsedFilters:
        return ParsedFilters.from_strings(excluded_dirs=dirs, excluded_files=files)

    def test_empty_filters_never_excludes(self):
        f = ParsedFilters.empty()
        assert not should_exclude_path("node_modules/lodash/index.js", f)
        assert not should_exclude_path("secrets/api_key.txt", f)

    def test_excluded_dir_segment_match(self):
        f = self._excl(dirs="node_modules")
        assert should_exclude_path("node_modules/lodash/index.js", f)

    def test_excluded_dir_in_nested_path(self):
        f = self._excl(dirs="secrets")
        assert should_exclude_path("api/secrets/api_keys.py", f)

    def test_non_excluded_path_not_excluded(self):
        f = self._excl(dirs="node_modules")
        assert not should_exclude_path("src/main.py", f)

    def test_excluded_file_exact_match(self):
        f = self._excl(files="package-lock.json")
        assert should_exclude_path("package-lock.json", f)
        assert should_exclude_path("subdir/package-lock.json", f)

    def test_excluded_file_no_glob(self):
        f = self._excl(files="*.min.js")
        # Inheriting the legacy quirk: exact match only, no fnmatch
        assert not should_exclude_path("bundle.min.js", f)

    def test_excluded_dir_dotslash_prefix_stripped(self):
        # User may type "./node_modules/" in the textarea
        f = ParsedFilters.from_strings(excluded_dirs="./node_modules/")
        assert should_exclude_path("node_modules/foo/bar.js", f)

    def test_multiple_excluded_dirs(self):
        f = self._excl(dirs="dist\nbuild")
        assert should_exclude_path("dist/app.js", f)
        assert should_exclude_path("build/output.js", f)
        assert not should_exclude_path("src/app.js", f)


# ---------------------------------------------------------------------------
# should_exclude_path — inclusion mode
# ---------------------------------------------------------------------------


class TestShouldExcludePathInclusionMode:
    def _incl(self, dirs="", files="") -> ParsedFilters:
        return ParsedFilters.from_strings(included_dirs=dirs, included_files=files)

    def test_included_dir_keeps_file(self):
        f = self._incl(dirs="src")
        assert not should_exclude_path("src/main.py", f)

    def test_non_included_dir_excluded(self):
        f = self._incl(dirs="src")
        assert should_exclude_path("node_modules/lodash/index.js", f)
        assert should_exclude_path("tests/test_main.py", f)

    def test_included_file_suffix(self):
        f = self._incl(files=".py")
        assert not should_exclude_path("api/main.py", f)

    def test_non_included_file_excluded(self):
        f = self._incl(files=".py")
        assert should_exclude_path("package.json", f)

    def test_excluded_dirs_ignored_in_inclusion_mode(self):
        # exclusion values are silently dropped when inclusion mode active
        f = ParsedFilters.from_strings(
            excluded_dirs="src",  # would exclude 'src' in exclusion mode
            included_dirs="src",  # inclusion mode wins
        )
        assert not should_exclude_path("src/main.py", f)

    def test_empty_inclusion_includes_all(self):
        # ParsedFilters with no rules at all: inclusion mode inactive → empty = never exclude
        f = ParsedFilters.empty()
        assert not should_exclude_path("anything/at/all.py", f)


# ---------------------------------------------------------------------------
# FilteredToolWrapper — read (pre-check)
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str, output: str = "file content") -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = f"Mock {name} tool"
    tool.parameters_schema = {}
    tool.to_function_schema = MagicMock(return_value={})
    tool.execute = AsyncMock(return_value=ToolResult(title=name, output=output))
    return tool


class TestFilteredToolWrapperRead:
    def _wrapper(self, excluded_dirs="", excluded_files="") -> FilteredToolWrapper:
        filters = ParsedFilters.from_strings(
            excluded_dirs=excluded_dirs, excluded_files=excluded_files
        )
        return FilteredToolWrapper(_make_mock_tool("read"), filters, "/repo")

    def test_excluded_file_blocked(self):
        wrapper = self._wrapper(excluded_dirs="secrets")
        result = asyncio.run(
            wrapper.execute({"file_path": "secrets/api_keys.py"})
        )
        assert "Blocked" in result.output
        assert result.metadata.get("filtered")
        wrapper._tool.execute.assert_not_called()

    def test_allowed_file_passes_through(self):
        wrapper = self._wrapper(excluded_dirs="secrets")
        asyncio.run(
            wrapper.execute({"file_path": "src/main.py"})
        )
        wrapper._tool.execute.assert_called_once()

    def test_empty_filters_always_passes(self):
        filters = ParsedFilters.empty()
        wrapper = FilteredToolWrapper(_make_mock_tool("read"), filters, "/repo")
        asyncio.run(
            wrapper.execute({"file_path": "secrets/key.txt"})
        )
        wrapper._tool.execute.assert_called_once()


# ---------------------------------------------------------------------------
# FilteredToolWrapper — glob (post-filter)
# ---------------------------------------------------------------------------


class TestFilteredToolWrapperGlob:
    def _wrapper(self, excluded_dirs="") -> FilteredToolWrapper:
        filters = ParsedFilters.from_strings(excluded_dirs=excluded_dirs)
        glob_output = "src/main.py\nnode_modules/lodash/index.js\nsrc/utils.py"
        return FilteredToolWrapper(_make_mock_tool("glob", glob_output), filters, "/repo")

    def test_excluded_paths_stripped(self):
        wrapper = self._wrapper(excluded_dirs="node_modules")
        result = asyncio.run(
            wrapper.execute({"pattern": "**/*.py"})
        )
        assert "node_modules" not in result.output
        assert "src/main.py" in result.output

    def test_removal_count_in_output(self):
        wrapper = self._wrapper(excluded_dirs="node_modules")
        result = asyncio.run(
            wrapper.execute({"pattern": "**/*.py"})
        )
        assert "1 path" in result.output or "hidden by filter" in result.output

    def test_no_filter_no_change(self):
        filters = ParsedFilters.empty()
        glob_output = "src/main.py\nnode_modules/index.js"
        wrapper = FilteredToolWrapper(_make_mock_tool("glob", glob_output), filters, "/repo")
        result = asyncio.run(
            wrapper.execute({"pattern": "**/*.py"})
        )
        assert result.output == glob_output

    def test_bash_passes_through(self):
        filters = ParsedFilters.from_strings(excluded_dirs="secrets")
        tool = _make_mock_tool("bash")
        wrapper = FilteredToolWrapper(tool, filters, "/repo")
        asyncio.run(
            wrapper.execute({"command": "cat secrets/key.txt", "description": "cat"})
        )
        tool.execute.assert_called_once()


# ---------------------------------------------------------------------------
# wrap_tools_with_filters
# ---------------------------------------------------------------------------


class TestWrapToolsWithFilters:
    def test_empty_filters_returns_same_dict(self):
        tools = {"read": _make_mock_tool("read")}
        result = wrap_tools_with_filters(tools, ParsedFilters.empty(), "/repo")
        assert result is tools

    def test_non_empty_filters_wraps_each_tool(self):
        filters = ParsedFilters.from_strings(excluded_dirs="secrets")
        tools = {
            "read": _make_mock_tool("read"),
            "glob": _make_mock_tool("glob"),
        }
        result = wrap_tools_with_filters(tools, filters, "/repo")
        assert result is not tools
        assert all(isinstance(v, FilteredToolWrapper) for v in result.values())

    def test_wrapped_tools_preserve_name(self):
        filters = ParsedFilters.from_strings(excluded_dirs="secrets")
        tools = {"read": _make_mock_tool("read")}
        result = wrap_tools_with_filters(tools, filters, "/repo")
        assert result["read"].name == "read"
