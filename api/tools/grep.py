"""
Grep tool — search file contents using regular expressions.

Python port of OpenCode's grep.ts. Uses ripgrep (rg) when available,
falls back to Python re + os.walk for environments without rg installed.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from api.tools.tool import Tool, ToolResult, find_ripgrep, load_description, validate_path_within_repo

logger = logging.getLogger(__name__)

MAX_LINE_LENGTH = 2000
MAX_MATCHES = 100

# Directories to skip in both rg and stdlib fallback paths
_NOISE_DIRS = {"node_modules", "__pycache__", ".git"}


def _grep_with_ripgrep(
    rg_path: str,
    pattern: str,
    search_path: str,
    include: str | None,
) -> list[dict]:
    """Run ripgrep and parse output into match dicts."""
    # NOTE: --glob order matters in ripgrep. Exclude patterns must come AFTER
    # include patterns, otherwise the include re-matches excluded paths.
    args = [
        rg_path,
        "-nH",
        "--hidden",
        "--no-messages",
        "--regexp",
        pattern,
    ]
    if include:
        args.extend(["--glob", include])
    args.extend(["--glob=!.git", search_path])

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        raise RuntimeError(f"ripgrep failed to run: {exc}") from exc

    # exit 0 = matches found, 1 = no matches, 2 = error (e.g. invalid regex)
    if result.returncode == 1:
        return []
    if result.returncode == 2:
        stderr = result.stderr.strip()
        if stderr:
            raise RuntimeError(f"ripgrep error: {stderr}")
        return []
    if result.returncode not in (0, 2):
        raise RuntimeError(f"ripgrep error (exit {result.returncode}): {result.stderr.strip()}")

    matches: list[dict] = []
    mtime_cache: dict[str, float] = {}
    for line in result.stdout.splitlines():
        if not line:
            continue
        # Default ripgrep -nH format: filepath:linenum:line_text
        # Left-to-right colon parsing. Safe on Unix/macOS where filenames
        # rarely contain colons; not safe for Windows drive letters (C:\...),
        # but this tool only targets Linux/macOS server environments.
        first_colon = line.find(":")
        if first_colon < 0:
            continue
        rest = line[first_colon + 1:]
        second_colon = rest.find(":")
        if second_colon < 0:
            continue
        file_path = line[:first_colon]
        line_num_str = rest[:second_colon]
        line_text = rest[second_colon + 1:]
        try:
            line_num = int(line_num_str)
        except ValueError:
            continue
        if file_path not in mtime_cache:
            try:
                mtime_cache[file_path] = Path(file_path).stat().st_mtime
            except OSError:
                mtime_cache[file_path] = 0.0
        matches.append({
            "path": file_path,
            "line_num": line_num,
            "line_text": line_text,
            "mtime": mtime_cache[file_path],
        })

    return matches


def _grep_with_stdlib(
    pattern: str,
    search_path: str,
    include: str | None,
) -> list[dict]:
    """Fallback: search using Python re + os.walk."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    # Convert glob include pattern to a simple suffix filter
    suffix_filter: str | None = None
    if include and "*." in include:
        suffix_filter = include.split("*.")[-1]

    matches: list[dict] = []
    for root, dirs, files in os.walk(search_path):
        # Skip .git and common noise dirs; keep other hidden dirs (.github, .vscode)
        # to match ripgrep --hidden --glob=!.git behaviour
        dirs[:] = [d for d in dirs if d not in _NOISE_DIRS]
        for fname in files:
            if suffix_filter and not fname.endswith("." + suffix_filter):
                continue
            file_path = os.path.join(root, fname)
            try:
                mtime = Path(file_path).stat().st_mtime
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line_text in enumerate(f, 1):
                        if compiled.search(line_text):
                            matches.append({
                                "path": file_path,
                                "line_num": line_num,
                                "line_text": line_text.rstrip("\n"),
                                "mtime": mtime,
                            })
            except (OSError, PermissionError):
                continue

    return matches


class GrepTool(Tool):
    """Search file contents using regular expressions."""

    name = "grep"
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for in file contents.",
            },
            "path": {
                "type": "string",
                "description": (
                    "The directory to search in. "
                    "Defaults to the repository root."
                ),
            },
            "include": {
                "type": "string",
                "description": 'File glob filter (e.g. "*.py", "*.{ts,tsx}").',
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        txt_path = Path(__file__).parent / "grep.txt"
        self.description = (
            load_description(str(txt_path))
            if txt_path.exists()
            else "Search file contents using regular expressions."
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern", "")
        if not pattern:
            raise ValueError("pattern is required")

        # Resolve and validate search path
        raw_path = params.get("path") or self.repo_path
        search_path = (
            raw_path if Path(raw_path).is_absolute()
            else str(Path(self.repo_path) / raw_path)
        )
        try:
            search_path = validate_path_within_repo(search_path, self.repo_path)
        except ValueError as exc:
            return ToolResult(
                title=pattern,
                output=f"Error: {exc}",
                metadata={"matches": 0, "truncated": False, "error": "path_traversal"},
            )

        include = params.get("include")

        logger.info("grep: pattern=%r path=%s include=%s", pattern, search_path, include)

        rg_path = find_ripgrep()
        try:
            if rg_path:
                matches = _grep_with_ripgrep(rg_path, pattern, search_path, include)
            else:
                logger.warning("ripgrep not found, falling back to stdlib grep")
                matches = _grep_with_stdlib(pattern, search_path, include)
        except (RuntimeError, ValueError) as exc:
            return ToolResult(
                title=pattern,
                output=f"Error: {exc}",
                metadata={"matches": 0, "truncated": False},
            )

        if not matches:
            return ToolResult(
                title=pattern,
                output="No files found",
                metadata={"matches": 0, "truncated": False},
            )

        # Sort by modification time (most-recent first), matching OpenCode behaviour
        matches.sort(key=lambda m: m["mtime"], reverse=True)

        total = len(matches)
        truncated = total > MAX_MATCHES
        final = matches[:MAX_MATCHES]

        output_lines = [f"Found {total} match{'es' if total != 1 else ''}"
                        + (f" (showing first {MAX_MATCHES})" if truncated else "")]

        current_file = ""
        for m in final:
            if current_file != m["path"]:
                if current_file:
                    output_lines.append("")
                current_file = m["path"]
                output_lines.append(f"{m['path']}:")
            line_text = m["line_text"]
            if len(line_text) > MAX_LINE_LENGTH:
                line_text = line_text[:MAX_LINE_LENGTH] + "..."
            output_lines.append(f"  Line {m['line_num']}: {line_text}")

        if truncated:
            output_lines.extend([
                "",
                f"(Results truncated: showing {MAX_MATCHES} of {total} matches "
                f"({total - MAX_MATCHES} hidden). Consider using a more specific path or pattern.)",
            ])

        return ToolResult(
            title=pattern,
            output="\n".join(output_lines),
            metadata={"matches": total, "truncated": truncated},
        )
