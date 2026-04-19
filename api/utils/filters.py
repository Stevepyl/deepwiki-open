"""
Path filter logic for agent wiki generation.

Mirrors the semantics of data_pipeline.py:should_process_file() so that the
agent tools apply the same include/exclude rules that the legacy RAG pipeline
enforces during indexing. This lets users who exclude `secrets/` in the UI
trust that the agent will not read, list, or glob those paths.

Two modes (same as legacy):
  Exclusion mode  — active when no included_* values are provided.
                    File is excluded if it matches any excluded_dirs segment
                    or any excluded_files exact name.
  Inclusion mode  — active when included_dirs or included_files is non-empty.
                    excluded_* values are completely ignored.
                    File is included only when it matches an included_dirs
                    segment OR an included_files name/suffix.

Parsing:
  Filter strings arrive as raw textarea text (newline-separated, URL-encoded)
  matching the format produced by src/components/UserSelector.tsx.
  Parsing mirrors api/websocket_wiki.py:96-107.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import unquote


def _parse_filter_string(s: str | None) -> tuple[str, ...]:
    """Parse a newline-separated, URL-encoded filter string into a tuple of clean values."""
    if not s:
        return ()
    return tuple(
        unquote(item.strip())
        for item in s.split("\n")
        if item.strip()
    )


@dataclass(frozen=True)
class ParsedFilters:
    """Immutable filter state derived from an AgentWikiRequest."""

    excluded_dirs: tuple[str, ...] = ()
    excluded_files: tuple[str, ...] = ()
    included_dirs: tuple[str, ...] = ()
    included_files: tuple[str, ...] = ()

    @property
    def use_inclusion_mode(self) -> bool:
        """True when the user supplied at least one include rule.

        In inclusion mode all excluded_* values are ignored, mirroring
        the data_pipeline.py:read_all_documents:187-202 behaviour.
        """
        return bool(self.included_dirs or self.included_files)

    @property
    def is_empty(self) -> bool:
        """True when no filter rules exist — allows callers to skip wrapping."""
        return not (
            self.excluded_dirs
            or self.excluded_files
            or self.included_dirs
            or self.included_files
        )

    @classmethod
    def empty(cls) -> "ParsedFilters":
        return cls()

    @classmethod
    def from_strings(
        cls,
        excluded_dirs: str | None = None,
        excluded_files: str | None = None,
        included_dirs: str | None = None,
        included_files: str | None = None,
    ) -> "ParsedFilters":
        return cls(
            excluded_dirs=_parse_filter_string(excluded_dirs),
            excluded_files=_parse_filter_string(excluded_files),
            included_dirs=_parse_filter_string(included_dirs),
            included_files=_parse_filter_string(included_files),
        )


def should_exclude_path(rel_path: str, filters: ParsedFilters) -> bool:
    """Return True if *rel_path* should be hidden from the agent.

    This is the logical inverse of data_pipeline.py:should_process_file():
      should_exclude_path(p, f) == not should_process_file(p, ..., f.*)

    Parameters
    ----------
    rel_path:
        A relative file path (relative to the repository root).
        Absolute paths are silently accepted but should be avoided.
    filters:
        Parsed filter state.  If ``filters.is_empty``, always returns False.
    """
    if filters.is_empty:
        return False

    path_parts = os.path.normpath(rel_path).split(os.sep)
    file_name = os.path.basename(rel_path)

    if filters.use_inclusion_mode:
        # Inclusion mode: exclude the file unless it matches an include rule.
        is_included = False

        if filters.included_dirs:
            for raw in filters.included_dirs:
                clean = raw.strip("./").rstrip("/")
                if clean and clean in path_parts:
                    is_included = True
                    break

        if not is_included and filters.included_files:
            for pattern in filters.included_files:
                if file_name == pattern or file_name.endswith(pattern):
                    is_included = True
                    break

        # If no rules at all, include everything (mirrors data_pipeline:274-275)
        if not filters.included_dirs and not filters.included_files:
            is_included = True

        return not is_included
    else:
        # Exclusion mode: exclude the file if it matches any exclude rule.
        for raw in filters.excluded_dirs:
            clean = raw.strip("./").rstrip("/")
            if clean and clean in path_parts:
                return True

        for pattern in filters.excluded_files:
            if file_name == pattern:
                return True

        return False
