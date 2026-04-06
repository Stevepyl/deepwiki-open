"""
Todo tool -- manage a structured task list for the agent session.

Python port of OpenCode's todowrite.ts. Simplified:
- No Effect library, no database persistence
- In-memory storage scoped to the tool instance lifetime
- Each call replaces the entire todo list (same semantics as OpenCode)
"""

import json
import logging
from pathlib import Path
from typing import Any

from api.tools.tool import Tool, ToolResult, load_description

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
VALID_PRIORITIES = {"high", "medium", "low"}


def _validate_todos(todos: list[dict[str, str]]) -> list[dict[str, str]]:
    """Validate and normalize the todo list. Returns a new list (no mutation)."""
    validated: list[dict[str, str]] = []
    for i, todo in enumerate(todos):
        content = todo.get("content", "").strip()
        if not content:
            raise ValueError(f"Todo #{i + 1} has empty content.")

        status = todo.get("status", "pending")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Todo #{i + 1} has invalid status '{status}'. "
                f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
            )

        priority = todo.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Todo #{i + 1} has invalid priority '{priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
            )

        validated.append({
            "content": content,
            "status": status,
            "priority": priority,
        })

    return validated


class TodoTool(Tool):
    """Manage a structured task list for the current session."""

    name = "todowrite"
    parameters_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete updated todo list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Brief description of the task.",
                        },
                        "status": {
                            "type": "string",
                            "description": (
                                "Current status of the task: "
                                "pending, in_progress, completed, cancelled."
                            ),
                        },
                        "priority": {
                            "type": "string",
                            "description": "Priority level: high, medium, low.",
                        },
                    },
                    "required": ["content", "status", "priority"],
                },
            },
        },
        "required": ["todos"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self._todos: list[dict[str, str]] = []
        txt_path = Path(__file__).parent / "todo.txt"
        self.description = (
            load_description(str(txt_path))
            if txt_path.exists()
            else "Manage a structured task list for the current session."
        )

    @property
    def todos(self) -> list[dict[str, str]]:
        """Return a copy of the current todo list (immutable access)."""
        return [dict(t) for t in self._todos]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        raw_todos = params.get("todos")
        if raw_todos is None:
            return ToolResult(
                title="todos",
                output="Error: 'todos' parameter is required.",
                metadata={"error": "missing_param"},
            )

        if not isinstance(raw_todos, list):
            return ToolResult(
                title="todos",
                output="Error: 'todos' must be an array.",
                metadata={"error": "invalid_type"},
            )

        try:
            validated = _validate_todos(raw_todos)
        except ValueError as exc:
            return ToolResult(
                title="todos",
                output=f"Error: {exc}",
                metadata={"error": "validation"},
            )

        # Replace the entire list (immutable swap, matching OpenCode semantics)
        self._todos = [dict(t) for t in validated]

        pending_count = sum(1 for t in validated if t["status"] not in ("completed", "cancelled"))

        logger.info("todo: updated %d todos (%d pending)", len(validated), pending_count)

        return ToolResult(
            title=f"{pending_count} todos",
            output=json.dumps(validated, indent=2, ensure_ascii=False),
            metadata={"todos": [dict(t) for t in validated], "pending": pending_count},
        )
