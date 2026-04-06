"""Tests for api.tools.bash — BashTool."""

import pytest

from api.tools.bash import BashTool


@pytest.fixture()
def bash(repo_path):
    return BashTool(repo_path)


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

class TestBashExecution:

    @pytest.mark.asyncio
    async def test_simple_echo(self, bash):
        result = await bash.execute({
            "command": "echo hello",
            "description": "Echo hello",
        })
        assert result.output.strip() == "hello"
        assert result.metadata["exit"] == 0

    @pytest.mark.asyncio
    async def test_exit_code_propagated(self, bash):
        result = await bash.execute({
            "command": "exit 42",
            "description": "Exit with code 42",
        })
        assert result.metadata["exit"] == 42

    @pytest.mark.asyncio
    async def test_stderr_merged_into_stdout(self, bash):
        result = await bash.execute({
            "command": "echo err >&2",
            "description": "Write to stderr",
        })
        assert "err" in result.output

    @pytest.mark.asyncio
    async def test_default_cwd_is_repo_root(self, bash, repo_path):
        result = await bash.execute({
            "command": "pwd",
            "description": "Print working directory",
        })
        # On macOS /tmp -> /private/tmp, so compare resolved paths
        from pathlib import Path
        assert Path(result.output.strip()).resolve() == Path(repo_path).resolve()


# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------

class TestBashWorkdir:

    @pytest.mark.asyncio
    async def test_relative_workdir(self, bash, fake_repo):
        result = await bash.execute({
            "command": "ls main.py",
            "workdir": "src",
            "description": "List main.py in src/",
        })
        assert result.metadata["exit"] == 0
        assert "main.py" in result.output

    @pytest.mark.asyncio
    async def test_absolute_workdir_within_repo(self, bash, fake_repo):
        abs_src = str(fake_repo / "src")
        result = await bash.execute({
            "command": "ls main.py",
            "workdir": abs_src,
            "description": "List main.py via absolute workdir",
        })
        assert result.metadata["exit"] == 0

    @pytest.mark.asyncio
    async def test_workdir_path_traversal_blocked(self, bash, fake_repo):
        result = await bash.execute({
            "command": "pwd",
            "workdir": str(fake_repo / ".." / "other"),
            "description": "Attempt path traversal",
        })
        assert "outside" in result.output.lower() or "error" in result.output.lower()
        assert result.metadata.get("error") == "path_traversal"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestBashTimeout:

    @pytest.mark.asyncio
    async def test_timeout_kills_long_command(self, bash):
        result = await bash.execute({
            "command": "sleep 60",
            "timeout": 500,  # 500ms
            "description": "Sleep that should be killed",
        })
        assert result.metadata["timed_out"] is True
        assert "timeout" in result.output.lower()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

class TestBashTruncation:

    @pytest.mark.asyncio
    async def test_large_output_truncated(self, bash):
        # Generate output exceeding MAX_LINES (2000)
        result = await bash.execute({
            "command": "seq 1 3000",
            "description": "Generate 3000 lines",
        })
        assert "truncated" in result.output.lower()


# ---------------------------------------------------------------------------
# Description and schema
# ---------------------------------------------------------------------------

class TestBashSchema:

    def test_description_loaded(self, bash):
        assert len(bash.description) > 50
        assert "bash" in bash.description.lower() or "command" in bash.description.lower()

    def test_function_schema_structure(self, bash):
        schema = bash.to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "bash"
        assert "command" in schema["function"]["parameters"]["properties"]
        assert "command" in schema["function"]["parameters"]["required"]
