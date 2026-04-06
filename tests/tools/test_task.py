"""Tests for api.tools.task -- TaskTool."""

import pytest

from api.tools.task import AgentInfo, TaskTool, _build_agent_list, _load_task_description
from api.tools.tool import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents():
    return [
        AgentInfo(name="coder", description="Write code", mode="subagent"),
        AgentInfo(name="researcher", description="Search docs", mode="subagent"),
        AgentInfo(name="orchestrator", description="Main agent", mode="primary"),
    ]


async def _echo_executor(prompt, agent_name, task_id, repo_path):
    """Mock executor that echoes back its inputs."""
    return ToolResult(
        title=agent_name,
        output=f"Executed: {prompt}",
        metadata={"agent": agent_name, "repo": repo_path},
    )


async def _failing_executor(prompt, agent_name, task_id, repo_path):
    """Mock executor that raises an exception."""
    raise RuntimeError("agent crashed")


@pytest.fixture()
def agents():
    return _make_agents()


@pytest.fixture()
def task(repo_path, agents):
    return TaskTool(repo_path, agents=agents, executor=_echo_executor)


# ---------------------------------------------------------------------------
# AgentInfo dataclass
# ---------------------------------------------------------------------------

class TestAgentInfo:

    def test_defaults(self):
        agent = AgentInfo(name="test", description="A test agent")
        assert agent.mode == "subagent"

    def test_frozen(self):
        agent = AgentInfo(name="test", description="desc")
        with pytest.raises(AttributeError):
            agent.name = "mutated"


# ---------------------------------------------------------------------------
# _build_agent_list
# ---------------------------------------------------------------------------

class TestBuildAgentList:

    def test_formats_subagents(self, agents):
        result = _build_agent_list(agents)
        assert "- coder: Write code" in result
        assert "- researcher: Search docs" in result

    def test_excludes_primary_agents(self, agents):
        result = _build_agent_list(agents)
        assert "orchestrator" not in result

    def test_empty_agents(self):
        assert "(No agents configured)" in _build_agent_list([])

    def test_only_primary_agents(self):
        agents = [AgentInfo(name="main", description="Main", mode="primary")]
        assert "(No subagents available)" in _build_agent_list(agents)

    def test_sorted_by_name(self):
        agents = [
            AgentInfo(name="zebra", description="Z agent"),
            AgentInfo(name="alpha", description="A agent"),
        ]
        result = _build_agent_list(agents)
        lines = result.strip().splitlines()
        assert lines[0].startswith("- alpha:")
        assert lines[1].startswith("- zebra:")


# ---------------------------------------------------------------------------
# _load_task_description
# ---------------------------------------------------------------------------

class TestLoadTaskDescription:

    def test_loads_description_with_agents(self, agents):
        desc = _load_task_description(agents)
        assert len(desc) > 20
        assert "coder" in desc

    def test_loads_description_without_agents(self):
        desc = _load_task_description([])
        assert len(desc) > 0


# ---------------------------------------------------------------------------
# Immutable reconfiguration
# ---------------------------------------------------------------------------

class TestTaskImmutability:

    def test_with_agents_returns_new_instance(self, repo_path, agents):
        tool = TaskTool(repo_path)
        new_tool = tool.with_agents(agents)
        assert new_tool is not tool
        assert "coder" in new_tool._agent_map
        assert "coder" not in tool._agent_map

    def test_with_executor_returns_new_instance(self, repo_path, agents):
        tool = TaskTool(repo_path, agents=agents)
        new_tool = tool.with_executor(_echo_executor)
        assert new_tool is not tool
        assert new_tool._executor is _echo_executor
        assert tool._executor is None


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestTaskParamValidation:

    @pytest.mark.asyncio
    async def test_missing_description(self, task):
        result = await task.execute({
            "prompt": "do something",
            "subagent_type": "coder",
        })
        assert result.metadata.get("error") == "missing_param"

    @pytest.mark.asyncio
    async def test_missing_prompt(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "subagent_type": "coder",
        })
        assert result.metadata.get("error") == "missing_param"

    @pytest.mark.asyncio
    async def test_missing_subagent_type(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "do something",
        })
        assert result.metadata.get("error") == "missing_param"

    @pytest.mark.asyncio
    async def test_empty_strings_treated_as_missing(self, task):
        result = await task.execute({
            "description": "  ",
            "prompt": "do something",
            "subagent_type": "coder",
        })
        assert result.metadata.get("error") == "missing_param"


# ---------------------------------------------------------------------------
# Unknown agent type
# ---------------------------------------------------------------------------

class TestTaskUnknownAgent:

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_error(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "do something",
            "subagent_type": "nonexistent",
        })
        assert result.metadata.get("error") == "unknown_agent"
        assert "nonexistent" in result.output

    @pytest.mark.asyncio
    async def test_error_lists_available_agents(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "do something",
            "subagent_type": "nonexistent",
        })
        assert "coder" in result.output
        assert "researcher" in result.output

    @pytest.mark.asyncio
    async def test_no_agents_configured(self, repo_path):
        tool = TaskTool(repo_path, agents=[], executor=_echo_executor)
        result = await tool.execute({
            "description": "Fix bug",
            "prompt": "do something",
            "subagent_type": "coder",
        })
        assert result.metadata.get("error") == "unknown_agent"
        assert "(none)" in result.output


# ---------------------------------------------------------------------------
# No executor
# ---------------------------------------------------------------------------

class TestTaskNoExecutor:

    @pytest.mark.asyncio
    async def test_no_executor_returns_error(self, repo_path, agents):
        tool = TaskTool(repo_path, agents=agents, executor=None)
        result = await tool.execute({
            "description": "Fix bug",
            "prompt": "do something",
            "subagent_type": "coder",
        })
        assert result.metadata.get("error") == "no_executor"


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestTaskExecution:

    @pytest.mark.asyncio
    async def test_dispatches_to_executor(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "Please fix the login bug",
            "subagent_type": "coder",
        })
        assert "Executed: Please fix the login bug" in result.output
        assert result.metadata["subagent_type"] == "coder"
        assert "task_id" in result.metadata

    @pytest.mark.asyncio
    async def test_output_wraps_in_task_result_tags(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "fix it",
            "subagent_type": "coder",
        })
        assert "<task_result>" in result.output
        assert "</task_result>" in result.output

    @pytest.mark.asyncio
    async def test_output_includes_task_id(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "fix it",
            "subagent_type": "coder",
        })
        assert result.output.startswith("task_id:")

    @pytest.mark.asyncio
    async def test_generates_uuid_session_id(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "fix it",
            "subagent_type": "coder",
        })
        task_id = result.metadata["task_id"]
        # UUID format: 8-4-4-4-12
        assert len(task_id) == 36
        assert task_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_resume_with_existing_task_id(self, task):
        result = await task.execute({
            "description": "Resume fix",
            "prompt": "continue fixing",
            "subagent_type": "coder",
            "task_id": "my-custom-session-id",
        })
        assert result.metadata["task_id"] == "my-custom-session-id"

    @pytest.mark.asyncio
    async def test_executor_metadata_merged(self, task):
        result = await task.execute({
            "description": "Fix bug",
            "prompt": "fix it",
            "subagent_type": "coder",
        })
        # Echo executor returns {"agent": ..., "repo": ...}
        assert result.metadata["agent"] == "coder"
        assert "repo" in result.metadata


# ---------------------------------------------------------------------------
# Executor errors
# ---------------------------------------------------------------------------

class TestTaskExecutorError:

    @pytest.mark.asyncio
    async def test_executor_exception_caught(self, repo_path, agents):
        tool = TaskTool(repo_path, agents=agents, executor=_failing_executor)
        result = await tool.execute({
            "description": "Fix bug",
            "prompt": "fix it",
            "subagent_type": "coder",
        })
        assert result.metadata.get("error") == "executor_error"
        assert "agent crashed" in result.metadata["error_detail"]

    @pytest.mark.asyncio
    async def test_executor_error_preserves_task_id(self, repo_path, agents):
        tool = TaskTool(repo_path, agents=agents, executor=_failing_executor)
        result = await tool.execute({
            "description": "Fix bug",
            "prompt": "fix it",
            "subagent_type": "coder",
        })
        assert "task_id" in result.metadata
        assert result.metadata["subagent_type"] == "coder"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestTaskSchema:

    def test_function_schema_structure(self, task):
        schema = task.to_function_schema()
        assert schema["function"]["name"] == "task"
        props = schema["function"]["parameters"]["properties"]
        assert "description" in props
        assert "prompt" in props
        assert "subagent_type" in props
        assert "task_id" in props
        required = schema["function"]["parameters"]["required"]
        assert "description" in required
        assert "prompt" in required
        assert "subagent_type" in required
        assert "task_id" not in required
