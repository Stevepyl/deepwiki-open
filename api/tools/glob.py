"""
Glob tool — find files by name pattern.

Python port of OpenCode's glob.ts. Uses ripgrep --files when available
(respects .gitignore automatically), falls back to pathlib.Path.rglob().
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from api.tools.tool import Tool, ToolResult, find_ripgrep, load_description, validate_path_within_repo

logger = logging.getLogger(__name__)

MAX_FILES = 100

# Directories to skip in the pathlib fallback (ripgrep handles via .gitignore)
_NOISE_DIRS = {"node_modules", "__pycache__", ".git"}


def _glob_with_ripgrep(rg_path: str, pattern: str, search_path: str) -> list[dict]:
    """Use `rg --files --glob <pattern>` to list matching files respecting .gitignore."""
    # NOTE: --glob order matters in ripgrep. Exclude patterns must come AFTER
    # include patterns, otherwise the include re-matches excluded paths.
    args = [
        rg_path,
        "--files",
        "--hidden",
        "--no-messages",
        "--glob",
        pattern,
        "--glob=!.git",
        search_path,
    ]
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

    if result.returncode not in (0, 1):
        raise RuntimeError(f"ripgrep error: {result.stderr.strip()}")

    files: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        files.append({"path": str(p), "mtime": mtime})

    return files


def _glob_with_pathlib(pattern: str, search_path: str) -> list[dict]:
    """Fallback: use pathlib Path.rglob() to match files."""
    base = Path(search_path)
    files: list[dict] = []
    try:
        for p in base.rglob(pattern):
            if not p.is_file():
                continue
            # Skip .git and common noise dirs; keep other hidden dirs
            # (.github, .vscode) to match ripgrep --hidden --glob=!.git
            parts = p.relative_to(base).parts
            if any(part in _NOISE_DIRS for part in parts):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            files.append({"path": str(p), "mtime": mtime})
    except Exception as exc:
        logger.warning("pathlib glob error: %s", exc)

    return files


class GlobTool(Tool):
    """Find files by name pattern, sorted by modification time."""

    name = "glob"
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    'The glob pattern to match files against (e.g. "**/*.py", "src/**/*.ts").'
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "The directory to search in. "
                    "Defaults to the repository root. "
                    'DO NOT pass "undefined" or "null" — simply omit for default.'
                ),
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        txt_path = Path(__file__).parent / "glob.txt"
        self.description = (
            load_description(str(txt_path))
            if txt_path.exists()
            else "Find files by name pattern sorted by modification time."
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        pattern = params.get("pattern", "")
        if not pattern:
            raise ValueError("pattern is required")

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
                metadata={"count": 0, "truncated": False, "error": "path_traversal"},
            )

        logger.info("glob: pattern=%r path=%s", pattern, search_path)

        rg_path = find_ripgrep()
        try:
            if rg_path:
                files = _glob_with_ripgrep(rg_path, pattern, search_path)
            else:
                logger.warning("ripgrep not found, falling back to pathlib glob")
                files = _glob_with_pathlib(pattern, search_path)
        except RuntimeError as exc:
            return ToolResult(
                title=pattern,
                output=f"Error: {exc}",
                metadata={"count": 0, "truncated": False},
            )

        if not files:
            return ToolResult(
                title=pattern,
                output="No files found",
                metadata={"count": 0, "truncated": False},
            )

        # Sort by modification time, most-recent first
        files.sort(key=lambda f: f["mtime"], reverse=True)

        truncated = len(files) > MAX_FILES
        final = files[:MAX_FILES]

        output_lines = [f["path"] for f in final]
        if truncated:
            output_lines.extend([
                "",
                f"(Results are truncated: showing first {MAX_FILES} results. "
                "Consider using a more specific path or pattern.)",
            ])

        return ToolResult(
            title=pattern,
            output="\n".join(output_lines),
            metadata={"count": len(files), "truncated": truncated},
        )
