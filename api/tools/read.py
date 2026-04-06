"""
Read tool -- read file contents with line numbers, or list directory entries.

Python port of OpenCode's read.ts. Simplifications:
- No LSP warm-up, no instruction loading, no Effect library
- No image/PDF attachment support (text-only for agent context)
- Binary detection via extension + byte sampling (same heuristic as OpenCode)
"""

import logging
import os
from pathlib import Path
from typing import Any

from api.tools.tool import Tool, ToolResult, load_description, validate_path_within_repo

logger = logging.getLogger(__name__)

DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_LINE_SUFFIX = f"... (line truncated to {MAX_LINE_LENGTH} chars)"
MAX_BYTES = 50 * 1024  # 50 KB

# Extensions that are always considered binary
_BINARY_EXTENSIONS = frozenset({
    ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".class", ".jar",
    ".war", ".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".bin", ".dat", ".obj", ".o", ".a",
    ".lib", ".wasm", ".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".ico", ".webp", ".mp3", ".mp4", ".avi", ".mov", ".pdf",
})


def _is_binary_file(filepath: str) -> bool:
    """Check if a file is binary by extension and byte sampling."""
    ext = Path(filepath).suffix.lower()
    if ext in _BINARY_EXTENSIONS:
        return True

    try:
        file_size = os.path.getsize(filepath)
    except OSError:
        return False

    if file_size == 0:
        return False

    sample_size = min(4096, file_size)
    try:
        with open(filepath, "rb") as f:
            data = f.read(sample_size)
    except OSError:
        return False

    if b"\x00" in data:
        return True

    non_printable = sum(
        1 for byte in data
        if byte < 9 or (13 < byte < 32)
    )
    return non_printable / len(data) > 0.3


def _read_lines(
    filepath: str,
    offset: int = 1,
    limit: int = DEFAULT_READ_LIMIT,
) -> dict[str, Any]:
    """
    Read lines from a file with offset/limit support.

    Returns dict with keys: raw (list[str]), count (int), cut (bool), more (bool), offset (int).
    """
    start = offset - 1  # Convert 1-indexed to 0-indexed
    raw: list[str] = []
    bytes_used = 0
    count = 0
    cut = False
    more = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_text in f:
            line_text = line_text.rstrip("\n").rstrip("\r")
            count += 1

            if count <= start:
                continue

            if len(raw) >= limit:
                more = True
                continue

            # Truncate long lines
            if len(line_text) > MAX_LINE_LENGTH:
                line_text = line_text[:MAX_LINE_LENGTH] + MAX_LINE_SUFFIX

            size = len(line_text.encode("utf-8")) + (1 if raw else 0)
            if bytes_used + size > MAX_BYTES:
                cut = True
                more = True
                break

            raw.append(line_text)
            bytes_used += size

    return {"raw": raw, "count": count, "cut": cut, "more": more, "offset": offset}


def _list_directory(dirpath: str) -> list[str]:
    """List directory entries, sorted, with trailing / for subdirectories."""
    entries: list[str] = []
    try:
        for item in sorted(os.listdir(dirpath)):
            full = os.path.join(dirpath, item)
            if os.path.isdir(full):
                entries.append(item + "/")
            else:
                entries.append(item)
    except OSError as exc:
        logger.warning("Failed to list directory %s: %s", dirpath, exc)
    return entries


def _suggest_similar(filepath: str) -> str:
    """If a file doesn't exist, suggest similar names from the same directory."""
    dirpath = str(Path(filepath).parent)
    base = Path(filepath).name.lower()

    try:
        items = os.listdir(dirpath)
    except OSError:
        return f"File not found: {filepath}"

    similar = [
        os.path.join(dirpath, item)
        for item in items
        if base in item.lower() or item.lower() in base
    ][:3]

    if similar:
        suggestions = "\n".join(similar)
        return f"File not found: {filepath}\n\nDid you mean one of these?\n{suggestions}"
    return f"File not found: {filepath}"


class ReadTool(Tool):
    """Read file contents with line numbers, or list directory entries."""

    name = "read"
    parameters_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file or directory to read.",
            },
            "offset": {
                "type": "number",
                "description": (
                    "The line number to start reading from (1-indexed). "
                    "Defaults to 1."
                ),
            },
            "limit": {
                "type": "number",
                "description": (
                    f"The maximum number of lines to read. Defaults to {DEFAULT_READ_LIMIT}."
                ),
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        txt_path = Path(__file__).parent / "read.txt"
        self.description = (
            load_description(str(txt_path))
            if txt_path.exists()
            else "Read file contents with line numbers, or list directory entries."
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        file_path = params.get("file_path", "")
        if not file_path:
            return ToolResult(
                title="read",
                output="Error: file_path is required.",
                metadata={"error": "missing_param"},
            )

        offset = int(params.get("offset", 1))
        limit = int(params.get("limit", DEFAULT_READ_LIMIT))

        if offset < 1:
            return ToolResult(
                title="read",
                output="Error: offset must be >= 1.",
                metadata={"error": "invalid_offset"},
            )

        if limit < 1:
            return ToolResult(
                title="read",
                output="Error: limit must be >= 1.",
                metadata={"error": "invalid_limit"},
            )

        # Resolve relative paths against repo root
        if not Path(file_path).is_absolute():
            file_path = str(Path(self.repo_path) / file_path)

        try:
            file_path = validate_path_within_repo(file_path, self.repo_path)
        except ValueError as exc:
            return ToolResult(
                title="read",
                output=f"Error: {exc}",
                metadata={"error": "path_traversal"},
            )

        # Compute relative title
        try:
            title = str(Path(file_path).relative_to(self.repo_path))
        except ValueError:
            title = file_path

        # Check existence
        if not os.path.exists(file_path):
            return ToolResult(
                title=title,
                output=_suggest_similar(file_path),
                metadata={"error": "not_found"},
            )

        # --- Directory mode ---
        if os.path.isdir(file_path):
            entries = _list_directory(file_path)
            start = offset - 1
            sliced = entries[start:start + limit]
            truncated = start + len(sliced) < len(entries)

            output_parts = [
                f"<path>{file_path}</path>",
                "<type>directory</type>",
                "<entries>",
                "\n".join(sliced),
            ]
            if truncated:
                output_parts.append(
                    f"\n(Showing {len(sliced)} of {len(entries)} entries. "
                    f"Use offset={offset + len(sliced)} to continue.)"
                )
            else:
                output_parts.append(f"\n({len(entries)} entries)")
            output_parts.append("</entries>")

            return ToolResult(
                title=title,
                output="\n".join(output_parts),
                metadata={"truncated": truncated, "count": len(entries)},
            )

        # --- File mode ---
        if _is_binary_file(file_path):
            return ToolResult(
                title=title,
                output=f"Error: Cannot read binary file: {file_path}",
                metadata={"error": "binary_file"},
            )

        try:
            result = _read_lines(file_path, offset=offset, limit=limit)
        except OSError as exc:
            return ToolResult(
                title=title,
                output=f"Error reading file: {exc}",
                metadata={"error": "read_error"},
            )

        raw = result["raw"]
        count = result["count"]
        file_cut = result["cut"]
        file_more = result["more"]
        file_offset = result["offset"]

        # Check offset is within range
        if count < file_offset and not (count == 0 and file_offset == 1):
            return ToolResult(
                title=title,
                output=f"Error: Offset {file_offset} is out of range for this file ({count} lines).",
                metadata={"error": "offset_out_of_range"},
            )

        # Build output with line numbers
        output = f"<path>{file_path}</path>\n<type>file</type>\n<content>\n"
        output += "\n".join(f"{i + file_offset}: {line}" for i, line in enumerate(raw))

        last = file_offset + len(raw) - 1
        next_offset = last + 1
        truncated = file_more or file_cut

        if file_cut:
            output += (
                f"\n\n(Output capped at {MAX_BYTES // 1024} KB. "
                f"Showing lines {file_offset}-{last}. "
                f"Use offset={next_offset} to continue.)"
            )
        elif file_more:
            output += (
                f"\n\n(Showing lines {file_offset}-{last} of {count}. "
                f"Use offset={next_offset} to continue.)"
            )
        else:
            output += f"\n\n(End of file - total {count} lines)"

        output += "\n</content>"

        return ToolResult(
            title=title,
            output=output,
            metadata={"truncated": truncated, "count": count},
        )
