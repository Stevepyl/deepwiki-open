"""
Agent tool registry for DeepWiki.

Usage:
    from api.tools import get_all_tools, get_tools_schema, get_tool

    # Get all tool instances for a given repo
    tools = get_all_tools(repo_path="/path/to/repo")

    # Get OpenAI-compatible function schemas for LLM calls
    schemas = get_tools_schema(repo_path="/path/to/repo")

    # Execute a tool by name
    tool = get_tool("grep", repo_path="/path/to/repo")
    result = await tool.execute({"pattern": "def main"})
"""

from typing import Any

from api.tools.bash import BashTool
from api.tools.glob import GlobTool
from api.tools.grep import GrepTool
from api.tools.ls import ListTool
from api.tools.read import ReadTool
from api.tools.task import AgentInfo, TaskTool
from api.tools.todo import TodoTool
from api.tools.tool import Tool, ToolResult

__all__ = [
    "Tool",
    "ToolResult",
    "AgentInfo",
    "BashTool",
    "GrepTool",
    "GlobTool",
    "ListTool",
    "ReadTool",
    "TaskTool",
    "TodoTool",
    "get_tool",
    "get_all_tools",
    "get_tools_schema",
]

# Tool classes registered by name — add new tools here
_TOOL_CLASSES: dict[str, type[Tool]] = {
    "bash": BashTool,
    "grep": GrepTool,
    "glob": GlobTool,
    "ls": ListTool,
    "read": ReadTool,
    "task": TaskTool,
    "todowrite": TodoTool,
}


def get_all_tools(repo_path: str) -> list[Tool]:
    """Return instantiated tool objects scoped to the given repository path."""
    return [cls(repo_path) for cls in _TOOL_CLASSES.values()]


def get_tool(name: str, repo_path: str) -> Tool:
    """
    Return a single tool instance by name.

    Raises:
        KeyError: if the tool name is not registered.
    """
    if name not in _TOOL_CLASSES:
        raise KeyError(f"Unknown tool '{name}'. Available: {list(_TOOL_CLASSES)}")
    return _TOOL_CLASSES[name](repo_path)


def get_tools_schema(repo_path: str) -> list[dict[str, Any]]:
    """
    Return OpenAI function-calling schemas for all tools.

    These are passed as the `tools` parameter when making LLM API calls.
    """
    return [tool.to_function_schema() for tool in get_all_tools(repo_path)]
