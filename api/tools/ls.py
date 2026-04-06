"""
List tool -- display directory structure as a tree.

Python port of OpenCode's ls.ts. Uses ripgrep --files when available
(respects .gitignore automatically), falls back to pathlib.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from api.tools.tool import Tool, ToolResult, find_ripgrep, load_description, validate_path_within_repo

logger = logging.getLogger(__name__)

LIMIT = 100

# Directories to skip by default (ripgrep handles most via .gitignore)
IGNORE_PATTERNS = [
    "node_modules/",
    "__pycache__/",
    ".git/",
    "dist/",
    "build/",
    "target/",
    "vendor/",
    "bin/",
    "obj/",
    ".idea/",
    ".vscode/",
    ".zig-cache/",
    "zig-out",
    ".coverage",
    "coverage/",
    "tmp/",
    "temp/",
    ".cache/",
    "cache/",
    "logs/",
    ".venv/",
    "venv/",
    "env/",
]


def _list_with_ripgrep(
    rg_path: str,
    search_path: str,
    extra_ignores: list[str] | None = None,
) -> list[str]:
    """Use `rg --files` to list files, respecting .gitignore and ignore patterns."""
    glob_args: list[str] = []
    for pattern in IGNORE_PATTERNS:
        glob_args.extend(["--glob", f"!{pattern}*"])
    for pattern in (extra_ignores or []):
        glob_args.extend(["--glob", f"!{pattern}"])

    args = [rg_path, "--files", "--hidden", "--no-messages", *glob_args, search_path]

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

    files: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            files.append(line)
        if len(files) > LIMIT:
            break
    return files


def _list_with_pathlib(
    search_path: str,
    extra_ignores: list[str] | None = None,
) -> list[str]:
    """Fallback: use pathlib to walk the directory."""
    base = Path(search_path)
    noise_dirs = {p.rstrip("/") for p in IGNORE_PATTERNS}
    if extra_ignores:
        noise_dirs.update(extra_ignores)

    files: list[str] = []
    try:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            parts = p.relative_to(base).parts
            if any(part in noise_dirs for part in parts):
                continue
            files.append(str(p))
            if len(files) >= LIMIT:
                break
    except Exception as exc:
        logger.warning("pathlib list error: %s", exc)

    return files


def _build_tree(files: list[str], search_path: str) -> str:
    """Build a directory tree string from a list of file paths."""
    base = Path(search_path)

    dirs: set[str] = set()
    files_by_dir: dict[str, list[str]] = {}

    for filepath in files:
        try:
            rel = str(Path(filepath).relative_to(base))
        except ValueError:
            rel = filepath
        dir_part = str(Path(rel).parent)
        if dir_part == ".":
            parts: list[str] = []
        else:
            parts = dir_part.split("/")

        # Register all parent directories
        for i in range(len(parts) + 1):
            dir_path = "." if i == 0 else "/".join(parts[:i])
            dirs.add(dir_path)

        # Add file to its directory bucket
        if dir_part not in files_by_dir:
            files_by_dir[dir_part] = []
        files_by_dir[dir_part].append(Path(rel).name)

    # Pre-compute parent -> children mapping to avoid O(n*d) scans
    children_map: dict[str, list[str]] = {}
    for d in dirs:
        parent = str(Path(d).parent)
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(d)

    def render_dir(dir_path: str, depth: int) -> str:
        if depth > 20:
            return ""

        indent = "  " * depth
        output = ""

        if depth > 0:
            output += f"{indent}{Path(dir_path).name}/\n"

        child_indent = "  " * (depth + 1)

        # Render subdirectories first
        for child in sorted(children_map.get(dir_path, [])):
            output += render_dir(child, depth + 1)

        # Render files
        for filename in sorted(files_by_dir.get(dir_path, [])):
            output += f"{child_indent}{filename}\n"

        return output

    return f"{search_path}/\n" + render_dir(".", 0)


class ListTool(Tool):
    """List files and directories as a tree structure."""

    name = "ls"
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "The absolute path to the directory to list. "
                    "Defaults to the repository root. "
                    'DO NOT pass "undefined" or "null" -- simply omit for default.'
                ),
            },
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of additional glob patterns to ignore.",
            },
        },
        "required": [],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        txt_path = Path(__file__).parent / "ls.txt"
        self.description = (
            load_description(str(txt_path))
            if txt_path.exists()
            else "List files and directories as a tree structure."
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        raw_path = params.get("path") or self.repo_path
        search_path = (
            raw_path if Path(raw_path).is_absolute()
            else str(Path(self.repo_path) / raw_path)
        )
        try:
            search_path = validate_path_within_repo(search_path, self.repo_path)
        except ValueError as exc:
            return ToolResult(
                title=search_path,
                output=f"Error: {exc}",
                metadata={"count": 0, "truncated": False, "error": "path_traversal"},
            )

        if not Path(search_path).is_dir():
            return ToolResult(
                title=search_path,
                output=f"Error: '{search_path}' is not a directory.",
                metadata={"count": 0, "truncated": False, "error": "not_directory"},
            )

        extra_ignores = params.get("ignore") or []

        logger.info("ls: path=%s ignore=%s", search_path, extra_ignores)

        rg_path = find_ripgrep()
        try:
            if rg_path:
                raw_files = _list_with_ripgrep(rg_path, search_path, extra_ignores)
            else:
                logger.warning("ripgrep not found, falling back to pathlib ls")
                raw_files = _list_with_pathlib(search_path, extra_ignores)
        except RuntimeError as exc:
            return ToolResult(
                title=search_path,
                output=f"Error: {exc}",
                metadata={"count": 0, "truncated": False},
            )

        truncated = len(raw_files) > LIMIT
        files = raw_files[:LIMIT]

        if not files:
            return ToolResult(
                title=Path(search_path).name,
                output=f"{search_path}/ (empty)",
                metadata={"count": 0, "truncated": False},
            )

        output = _build_tree(files, search_path)
        if truncated:
            output += (
                f"\n(Results truncated: showing first {LIMIT} files. "
                "Use a more specific path to narrow results.)"
            )

        rel_title = str(Path(search_path).relative_to(self.repo_path)) if search_path != self.repo_path else "."

        return ToolResult(
            title=rel_title,
            output=output,
            metadata={"count": len(raw_files), "truncated": truncated},
        )
