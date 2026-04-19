"""
Repository file-tree helpers.

Extracted from api/api.py:/local_repo/structure handler so the logic can be
reused by the agent wiki generator without duplicating the os.walk filters.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.utils.filters import ParsedFilters


_SKIP_DIRS = frozenset({"__pycache__", "node_modules", ".venv", ".git"})
_SKIP_FILES = frozenset({"__init__.py", ".DS_Store"})


def build_file_tree(
    repo_path: str,
    filters: "ParsedFilters | None" = None,
    max_files: int = 5_000,
) -> str:
    """Return a newline-separated, sorted list of relative file paths.

    When *filters* is provided, paths are additionally checked against
    ParsedFilters.should_exclude_path() so that user-supplied include/exclude
    rules are respected in the file-tree hint sent to the planner LLM.

    Callers that do not pass *filters* receive the same output as before
    (no behavioral change).
    """
    from api.utils.filters import should_exclude_path  # lazy import avoids circular dep

    lines: list[str] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in _SKIP_DIRS
        ]
        for file in files:
            if file.startswith(".") or file in _SKIP_FILES:
                continue
            rel_dir = os.path.relpath(root, repo_path)
            rel_file = os.path.join(rel_dir, file) if rel_dir != "." else file
            if filters is not None and should_exclude_path(rel_file, filters):
                continue
            lines.append(rel_file)
            if len(lines) >= max_files:
                lines.append(f"... (truncated at {max_files} files)")
                return "\n".join(sorted(lines[:-1]) + [lines[-1]])
    return "\n".join(sorted(lines))


def read_repo_readme(repo_path: str) -> str:
    """Return the contents of the repository's README.md (case-insensitive).

    Searches only the repository root.  Returns an empty string if no README
    is found or if the file cannot be read.
    """
    try:
        for entry in os.listdir(repo_path):
            if entry.lower() == "readme.md":
                readme_path = os.path.join(repo_path, entry)
                with open(readme_path, encoding="utf-8", errors="replace") as f:
                    return f.read()
    except OSError:
        pass
    return ""
