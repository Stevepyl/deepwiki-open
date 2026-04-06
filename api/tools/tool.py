"""
Base framework for agent tools.

Design mirrors OpenCode's Tool.define() pattern but simplified for Python:
- No permission system (tools run server-side within cloned repo)
- No Effect/Zod — uses dataclasses + dict schemas
- Async execute() for WebSocket compatibility
"""

import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_LINES = 2000
MAX_BYTES = 50 * 1024  # 50 KB


@dataclass
class ToolResult:
    """Structured result returned by every tool execution."""

    title: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


def truncate_output(text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES) -> str:
    """
    Truncate output to stay within line/byte limits.

    Mirrors OpenCode's Truncate.output() — no file dump because the agent
    loop assembles context in memory rather than via files.
    """
    lines = text.split("\n")
    total_bytes = len(text.encode("utf-8"))

    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return text

    out: list[str] = []
    bytes_used = 0

    for line in lines[:max_lines]:
        size = len(line.encode("utf-8")) + (1 if out else 0)
        if bytes_used + size > max_bytes:
            break
        out.append(line)
        bytes_used += size

    removed = len(lines) - len(out)
    preview = "\n".join(out)
    return f"{preview}\n\n...{removed} lines truncated. Use a more specific path or pattern to narrow results."


def load_description(txt_path: str) -> str:
    """Load tool description from a .txt file, substituting runtime variables."""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = (
        text.replace("${os}", sys.platform)
        .replace("${shell}", os.environ.get("SHELL", "bash"))
        .replace("${maxLines}", str(MAX_LINES))
        .replace("${maxBytes}", str(MAX_BYTES))
    )
    return text


def find_ripgrep() -> str | None:
    """Return path to ripgrep binary, or None if not found."""
    return shutil.which("rg")


def validate_path_within_repo(path: str, repo_path: str) -> str:
    """
    Resolve a user-provided path and verify it stays within the repo boundary.

    Returns the resolved absolute path string.
    Raises ValueError if the path escapes the repository root.
    """
    resolved = str(Path(path).resolve())
    repo_resolved = str(Path(repo_path).resolve())
    if not resolved.startswith(repo_resolved + os.sep) and resolved != repo_resolved:
        raise ValueError(
            f"Path '{path}' resolves to '{resolved}' which is outside "
            f"the repository root '{repo_resolved}'."
        )
    return resolved


class Tool(ABC):
    """Abstract base class for all agent tools."""

    #: Unique tool identifier (used in LLM function calling)
    name: str

    #: Human-readable description loaded from the paired .txt file
    description: str

    #: JSON Schema for the tool's parameters, compatible with OpenAI function-calling format.
    #: Shape: {"type": "object", "properties": {...}, "required": [...]}
    parameters_schema: dict[str, Any]

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Args:
            params: Validated parameter dict matching parameters_schema.

        Returns:
            ToolResult with title, output text, and optional metadata.
        """

    def to_function_schema(self) -> dict[str, Any]:
        """
        Return an OpenAI-compatible function schema for this tool.
        Used when building the tools list for LLM calls.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
