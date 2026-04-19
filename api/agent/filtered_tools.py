"""
FilteredToolWrapper — applies ParsedFilters to agent tool execution.

Design rationale
----------------
Rather than modifying the five filesystem tools (ls, glob, read, grep, bash),
we wrap them at the call site inside the wiki generator.  The wrapper
implements the same duck-typed interface the agent loop expects (name,
description, parameters_schema, to_function_schema, execute) without
inheriting Tool ABC, keeping the existing Tool hierarchy untouched.

Per-tool strategy
-----------------
  read  — pre-check: if the requested file_path is excluded, return an
           informational ToolResult without opening the file.
  ls    — pre-check: if the requested path resolves to an excluded directory,
           block the listing.
  grep  — pre-check: if the search root (path param) is excluded, block.
           Individual result lines are not post-filtered (the grouped tree
           format makes safe line-level filtering fragile).
  glob  — post-filter: output is one relative path per line; excluded paths
           are removed from the list before returning.
  bash  — pass-through; no reliable way to parse arbitrary shell commands.
           The BashTool sandbox gap is documented in
           handbooks/bash-agent-sandbox-gap.md.

All other tool names (task, todowrite, and future tools) pass through
unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from api.tools.tool import ToolResult
from api.utils.filters import ParsedFilters, should_exclude_path

logger = logging.getLogger(__name__)

# Tools that receive an explicit filesystem path parameter which we can pre-check.
_PRE_CHECK_TOOLS: dict[str, str] = {
    "read": "file_path",
    "ls": "path",
    "grep": "path",
}

# Tools whose text output is one relative path per line (post-filterable).
_POST_FILTER_TOOLS: frozenset[str] = frozenset({"glob"})


def _to_rel_path(raw_path: str, repo_path: str) -> str:
    """Return a repo-relative path, stripping the repo prefix when absolute."""
    if os.path.isabs(raw_path):
        try:
            return os.path.relpath(raw_path, repo_path)
        except ValueError:
            return raw_path
    return raw_path


class FilteredToolWrapper:
    """Duck-typed wrapper that enforces ParsedFilters on a single Tool instance."""

    def __init__(self, tool: Any, filters: ParsedFilters, repo_path: str) -> None:
        self._tool = tool
        self._filters = filters
        self._repo_path = repo_path

    # --- Duck-typed Tool interface ---

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._tool.parameters_schema

    def to_function_schema(self) -> dict[str, Any]:
        return self._tool.to_function_schema()

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        tool_name = self._tool.name

        # --- Pre-check for path-bearing tools ---
        if tool_name in _PRE_CHECK_TOOLS:
            param_key = _PRE_CHECK_TOOLS[tool_name]
            raw = params.get(param_key) or ""
            if raw:
                rel = _to_rel_path(raw, self._repo_path)
                if should_exclude_path(rel, self._filters):
                    logger.info(
                        "filtered_tools: blocked %s on excluded path %r", tool_name, rel
                    )
                    return ToolResult(
                        title=raw,
                        output=(
                            f"Blocked: '{rel}' is excluded by the repository filter. "
                            "Choose a different path."
                        ),
                        metadata={"filtered": True},
                    )

        # --- Execute the underlying tool ---
        result = await self._tool.execute(params)

        # --- Post-filter for glob (one path per line) ---
        if tool_name in _POST_FILTER_TOOLS:
            result = self._filter_path_list_output(result)

        return result

    def _filter_path_list_output(self, result: ToolResult) -> ToolResult:
        """Strip excluded paths from a newline-separated path list output."""
        lines = result.output.splitlines()
        kept: list[str] = []
        removed = 0

        for line in lines:
            stripped = line.strip()
            # Preserve blank lines and annotation lines (parenthesised messages).
            if not stripped or stripped.startswith("("):
                kept.append(line)
                continue
            if should_exclude_path(stripped, self._filters):
                removed += 1
            else:
                kept.append(line)

        if removed == 0:
            return result

        new_output = "\n".join(kept)
        if removed:
            new_output += f"\n({removed} path{'s' if removed != 1 else ''} hidden by filter)"

        metadata = dict(result.metadata)
        metadata["filtered_count"] = removed
        return ToolResult(title=result.title, output=new_output, metadata=metadata)


def wrap_tools_with_filters(
    tools: dict[str, Any],
    filters: ParsedFilters,
    repo_path: str,
) -> dict[str, Any]:
    """Return a new tools dict with each tool wrapped by FilteredToolWrapper.

    If ``filters.is_empty`` the original dict is returned unchanged to avoid
    any per-call overhead when the user provided no filter rules.
    """
    if filters.is_empty:
        return tools
    return {
        name: FilteredToolWrapper(tool, filters, repo_path)
        for name, tool in tools.items()
    }
