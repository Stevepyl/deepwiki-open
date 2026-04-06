"""Tests for api.tools.todo -- TodoTool."""

import pytest

from api.tools.todo import TodoTool, _validate_todos


@pytest.fixture()
def todo(repo_path):
    return TodoTool(repo_path)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestTodoBasic:

    @pytest.mark.asyncio
    async def test_create_todos(self, todo):
        result = await todo.execute({"todos": [
            {"content": "Fix bug", "status": "pending", "priority": "high"},
            {"content": "Write docs", "status": "in_progress", "priority": "medium"},
        ]})
        assert result.metadata["pending"] == 2
        assert len(result.metadata["todos"]) == 2

    @pytest.mark.asyncio
    async def test_replace_entire_list(self, todo):
        """Each call replaces the full list (OpenCode semantics)."""
        await todo.execute({"todos": [
            {"content": "Task A", "status": "pending", "priority": "high"},
            {"content": "Task B", "status": "pending", "priority": "low"},
        ]})
        result = await todo.execute({"todos": [
            {"content": "Task C", "status": "completed", "priority": "medium"},
        ]})
        assert len(result.metadata["todos"]) == 1
        assert result.metadata["todos"][0]["content"] == "Task C"
        assert result.metadata["pending"] == 0

    @pytest.mark.asyncio
    async def test_empty_list_clears_todos(self, todo):
        await todo.execute({"todos": [
            {"content": "Task A", "status": "pending", "priority": "high"},
        ]})
        result = await todo.execute({"todos": []})
        assert len(result.metadata["todos"]) == 0
        assert result.metadata["pending"] == 0

    @pytest.mark.asyncio
    async def test_pending_count_excludes_completed_and_cancelled(self, todo):
        result = await todo.execute({"todos": [
            {"content": "Done", "status": "completed", "priority": "low"},
            {"content": "Cancelled", "status": "cancelled", "priority": "low"},
            {"content": "Active", "status": "in_progress", "priority": "high"},
        ]})
        assert result.metadata["pending"] == 1


# ---------------------------------------------------------------------------
# Immutable access
# ---------------------------------------------------------------------------

class TestTodoImmutability:

    @pytest.mark.asyncio
    async def test_todos_property_returns_copy(self, todo):
        await todo.execute({"todos": [
            {"content": "Task A", "status": "pending", "priority": "high"},
        ]})
        copy = todo.todos
        copy[0]["content"] = "MUTATED"
        # Original should be unchanged
        assert todo.todos[0]["content"] == "Task A"

    @pytest.mark.asyncio
    async def test_todos_property_empty_initially(self, todo):
        assert todo.todos == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestTodoValidation:

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="empty content"):
            _validate_todos([{"content": "", "status": "pending", "priority": "high"}])

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValueError, match="empty content"):
            _validate_todos([{"content": "   ", "status": "pending", "priority": "high"}])

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="invalid status"):
            _validate_todos([{"content": "Task", "status": "done", "priority": "high"}])

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="invalid priority"):
            _validate_todos([{"content": "Task", "status": "pending", "priority": "urgent"}])

    def test_defaults_applied(self):
        result = _validate_todos([{"content": "Task"}])
        assert result[0]["status"] == "pending"
        assert result[0]["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_validation_error_in_execute(self, todo):
        result = await todo.execute({"todos": [
            {"content": "", "status": "pending", "priority": "high"},
        ]})
        assert result.metadata.get("error") == "validation"


# ---------------------------------------------------------------------------
# Parameter errors
# ---------------------------------------------------------------------------

class TestTodoParamErrors:

    @pytest.mark.asyncio
    async def test_missing_todos_param(self, todo):
        result = await todo.execute({})
        assert result.metadata.get("error") == "missing_param"

    @pytest.mark.asyncio
    async def test_todos_not_array(self, todo):
        result = await todo.execute({"todos": "not an array"})
        assert result.metadata.get("error") == "invalid_type"


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

class TestTodoOutput:

    @pytest.mark.asyncio
    async def test_output_is_json(self, todo):
        import json
        result = await todo.execute({"todos": [
            {"content": "Task A", "status": "pending", "priority": "high"},
        ]})
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["content"] == "Task A"

    @pytest.mark.asyncio
    async def test_title_shows_pending_count(self, todo):
        result = await todo.execute({"todos": [
            {"content": "A", "status": "pending", "priority": "low"},
            {"content": "B", "status": "completed", "priority": "low"},
        ]})
        assert "1 todos" in result.title


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestTodoSchema:

    def test_function_schema_structure(self, todo):
        schema = todo.to_function_schema()
        assert schema["function"]["name"] == "todowrite"
        props = schema["function"]["parameters"]["properties"]
        assert "todos" in props
        assert "todos" in schema["function"]["parameters"]["required"]
