"""
Task tool -- sub-agent dispatcher for delegating work to specialized agents.

Python port of OpenCode's task.ts. Simplifications:
- No Effect library, no session persistence to database
- No permission system (server-side execution)
- Executor injected via dependency injection (agent loop not yet implemented)
- In-memory session tracking scoped to tool instance lifetime
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from api.tools.tool import Tool, ToolResult, load_description

logger = logging.getLogger(__name__)

AgentMode = Literal["primary", "subagent", "all"]


@dataclass(frozen=True)
class AgentInfo:
    """Minimal agent descriptor for task tool's description generation.

    Mirrors OpenCode's Agent.Info but stripped to essential fields.
    The full agent configuration lives in the agent loop (not yet implemented).
    """

    name: str
    description: str
    mode: AgentMode = "subagent"


# Type alias for the executor callback.
# Signature: (prompt, agent_name, task_id, repo_path) -> ToolResult
TaskExecutor = Callable[
    [str, str, Optional[str], str],
    Awaitable[ToolResult],
]

_TXT_PATH = Path(__file__).parent / "task.txt"


def _build_agent_list(agents: list[AgentInfo]) -> str:
    """Format agent list for the {agents} placeholder in task.txt."""
    if not agents:
        return "(No agents configured)"
    lines = []
    for agent in sorted(agents, key=lambda a: a.name):
        if agent.mode == "primary":
            continue
        lines.append(f"- {agent.name}: {agent.description}")
    return "\n".join(lines) if lines else "(No subagents available)"


def _load_task_description(agents: list[AgentInfo]) -> str:
    """Load and populate the task tool description from task.txt."""
    if _TXT_PATH.exists():
        raw = load_description(str(_TXT_PATH))
        if "{agents}" not in raw:
            logger.warning("task.txt is missing the {agents} placeholder")
        return raw.replace("{agents}", _build_agent_list(agents))
    return "Dispatch tasks to specialized sub-agents."


class TaskTool(Tool):
    """Dispatch tasks to specialized sub-agents.

    This tool uses dependency injection: pass ``agents`` and ``executor`` at
    construction time, or create a reconfigured copy via ``with_agents()`` /
    ``with_executor()``.  The ``get_all_tools()`` registry creates a default
    instance with no agents and no executor; the agent loop should call
    ``with_agents(...).with_executor(...)`` to obtain a fully configured copy.
    """

    name = "task"
    parameters_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A short (3-5 words) summary of the task.",
            },
            "prompt": {
                "type": "string",
                "description": "The full task description for the agent to perform.",
            },
            "subagent_type": {
                "type": "string",
                "description": "The type of specialized agent to use for this task.",
            },
            "task_id": {
                "type": "string",
                "description": (
                    "Pass a prior task_id to resume a previous subagent session "
                    "instead of creating a fresh one."
                ),
            },
        },
        "required": ["description", "prompt", "subagent_type"],
    }

    def __init__(
        self,
        repo_path: str,
        agents: Optional[list[AgentInfo]] = None,
        executor: Optional[TaskExecutor] = None,
    ) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self._agents = list(agents) if agents else []
        self._executor = executor
        self._agent_map: dict[str, AgentInfo] = {
            a.name: a for a in self._agents if a.mode != "primary"
        }
        self.description = _load_task_description(self._agents)

    def with_agents(self, agents: list[AgentInfo]) -> "TaskTool":
        """Return a new TaskTool with updated agent list (immutable)."""
        return TaskTool(
            repo_path=self.repo_path,
            agents=agents,
            executor=self._executor,
        )

    def with_executor(self, executor: TaskExecutor) -> "TaskTool":
        """Return a new TaskTool with the executor injected (immutable)."""
        return TaskTool(
            repo_path=self.repo_path,
            agents=self._agents,
            executor=executor,
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        description = params.get("description", "").strip()
        prompt = params.get("prompt", "").strip()
        subagent_type = params.get("subagent_type", "").strip()
        task_id = params.get("task_id")

        # --- Validate required fields ---
        if not description:
            return ToolResult(
                title="task",
                output="Error: 'description' parameter is required.",
                metadata={"error": "missing_param"},
            )

        if not prompt:
            return ToolResult(
                title="task",
                output="Error: 'prompt' parameter is required.",
                metadata={"error": "missing_param"},
            )

        if not subagent_type:
            return ToolResult(
                title="task",
                output="Error: 'subagent_type' parameter is required.",
                metadata={"error": "missing_param"},
            )

        # --- Validate agent type ---
        if subagent_type not in self._agent_map:
            available = ", ".join(sorted(self._agent_map.keys()))
            return ToolResult(
                title=description,
                output=(
                    f"Error: Unknown agent type '{subagent_type}'. "
                    f"Available agents: {available or '(none)'}"
                ),
                metadata={"error": "unknown_agent"},
            )

        # --- Check executor availability ---
        if self._executor is None:
            return ToolResult(
                title=description,
                output=(
                    "Error: Task execution is not available. "
                    "The agent loop has not been initialized yet."
                ),
                metadata={"error": "no_executor"},
            )

        # --- Generate session ID if not resuming ---
        session_id = task_id or str(uuid.uuid4())

        logger.info(
            "task: dispatching to '%s' (session=%s, resume=%s)",
            subagent_type,
            session_id,
            task_id is not None,
        )

        # --- Execute via injected callback ---
        try:
            result = await self._executor(prompt, subagent_type, session_id, self.repo_path)
        except Exception as exc:
            logger.error("task executor error: %s", exc)
            return ToolResult(
                title=description,
                output="Error: the sub-agent task failed. Check server logs for details.",
                metadata={
                    "error": "executor_error",
                    "error_detail": str(exc),
                    "task_id": session_id,
                    "subagent_type": subagent_type,
                },
            )

        # --- Wrap result with task_id for session resumption ---
        # Spread executor metadata first, then override with task-level keys
        # so that task_id and subagent_type are always authoritative.
        output = "\n".join([
            f"task_id: {session_id}",
            "",
            "<task_result>",
            result.output,
            "</task_result>",
        ])

        return ToolResult(
            title=description,
            output=output,
            metadata={
                **result.metadata,
                "task_id": session_id,
                "subagent_type": subagent_type,
            },
        )
