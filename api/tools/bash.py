"""
Bash tool — execute shell commands within a repository.

Python port of OpenCode's bash.ts. Simplifications:
- No tree-sitter command parsing (no permission system needed server-side)
- No PowerShell support (server runs on Linux/macOS)
- Uses asyncio.create_subprocess_shell for async execution
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from api.tools.tool import Tool, ToolResult, load_description, truncate_output, validate_path_within_repo

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes, mirrors OpenCode default


class BashTool(Tool):
    """Execute shell commands in the repository directory."""

    name = "bash"
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Optional timeout in milliseconds. "
                    f"Defaults to {DEFAULT_TIMEOUT_MS}ms (2 minutes)."
                ),
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Working directory for the command. "
                    "Defaults to the repository root. Use this instead of 'cd' patterns."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "Clear, concise description of what this command does in 5-10 words. "
                    "Examples: 'Lists files in current directory', 'Shows git status'."
                ),
            },
        },
        "required": ["command", "description"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        txt_path = Path(__file__).parent / "bash.txt"
        self.description = (
            load_description(str(txt_path)).replace("${directory}", self.repo_path)
            if txt_path.exists()
            else "Execute shell commands in the repository."
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        command = params["command"]
        description = params.get("description", command[:60])
        timeout_ms = params.get("timeout", DEFAULT_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000.0

        # Resolve and validate working directory
        workdir = params.get("workdir")
        if workdir:
            raw_cwd = str(Path(workdir).resolve()) if Path(workdir).is_absolute() else str(
                Path(self.repo_path) / workdir
            )
            try:
                cwd = validate_path_within_repo(raw_cwd, self.repo_path)
            except ValueError as exc:
                return ToolResult(
                    title=description,
                    output=f"Error: {exc}",
                    metadata={"exit": None, "description": description, "error": "path_traversal"},
                )
        else:
            cwd = self.repo_path

        logger.info("bash execute: %s (cwd=%s, timeout=%ss)", description, cwd, timeout_s)

        output = ""
        expired = False
        exit_code: int | None = None

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )

            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_s,
                )
                output = stdout_bytes.decode("utf-8", errors="replace")
                exit_code = proc.returncode
            except asyncio.TimeoutError:
                expired = True
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

        except Exception as exc:
            logger.error("bash tool error: %s", exc)
            return ToolResult(
                title=description,
                output=f"Error executing command: {exc}",
                metadata={"exit": None, "description": description, "error": str(exc)},
            )

        if expired:
            output += (
                "\n\n<bash_metadata>\n"
                f"bash tool terminated command after exceeding timeout {timeout_ms}ms"
                "\n</bash_metadata>"
            )

        truncated = truncate_output(output)

        return ToolResult(
            title=description,
            output=truncated,
            metadata={
                "exit": exit_code,
                "description": description,
                "timed_out": expired,
            },
        )
