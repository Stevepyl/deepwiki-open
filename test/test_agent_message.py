"""
Unit tests for api/agent/message.py.

Tests cover:
- All four part types: TextPart, ToolCallPart, ToolResultPart, ErrorPart
- ToolCallPart state machine transitions (happy path and invalid transitions)
- Immutability: state transitions create new instances, originals unchanged
- Dict independence: copied instances do not share dict objects
- AgentMessage convenience constructors
- Part accessor properties
- Immutable update helpers
- messages_to_openai_format conversion (all roles, edge cases)
- from_chat_messages backward compatibility
- tool_call_part_from_openai parsing (normal and malformed input)
- Discriminated union JSON round-trip
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.agent.message import (
    AgentMessage,
    ErrorPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    message_to_openai_format,
    messages_to_openai_format,
    tool_call_part_from_openai,
)
from api.tools.tool import ToolResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# TextPart
# ---------------------------------------------------------------------------


class TestTextPart:
    def test_basic_construction(self):
        part = TextPart(content="hello")
        assert part.type == "text"
        assert part.content == "hello"

    def test_empty_content(self):
        part = TextPart(content="")
        assert part.content == ""

    def test_frozen(self):
        part = TextPart(content="hello")
        with pytest.raises(Exception):
            part.content = "mutated"  # type: ignore[misc]

    def test_json_round_trip(self):
        part = TextPart(content="hello world")
        data = part.model_dump(mode="json")
        assert data == {"type": "text", "content": "hello world"}
        restored = TextPart.model_validate(data)
        assert restored.content == "hello world"


# ---------------------------------------------------------------------------
# ToolCallPart — construction and dict independence
# ---------------------------------------------------------------------------


class TestToolCallPartConstruction:
    def test_default_state(self):
        tc = ToolCallPart(tool_name="grep", tool_args={"pattern": "main"})
        assert tc.state == "pending"
        assert tc.result is None
        assert tc.error is None
        assert tc.started_at is None
        assert tc.completed_at is None

    def test_auto_generated_tool_call_id(self):
        tc = ToolCallPart(tool_name="grep")
        assert tc.tool_call_id.startswith("call_")
        assert len(tc.tool_call_id) > 5

    def test_custom_tool_call_id(self):
        tc = ToolCallPart(tool_name="grep", tool_call_id="call_abc")
        assert tc.tool_call_id == "call_abc"

    def test_frozen(self):
        tc = ToolCallPart(tool_name="grep")
        with pytest.raises(Exception):
            tc.state = "running"  # type: ignore[misc]

    def test_dict_independence_on_construction(self):
        """Instances must own independent copies of their dict fields."""
        args = {"pattern": "main"}
        tc = ToolCallPart(tool_name="grep", tool_args=args)
        # Mutating the original dict after construction must not affect tc
        args["pattern"] = "MUTATED"
        assert tc.tool_args["pattern"] == "main"

    def test_dict_independence_across_copies(self):
        """model_copy() must produce instances with independent dicts."""
        tc = ToolCallPart(tool_name="grep", tool_args={"pattern": "main"})
        running = tc.start()
        # Mutate the dict inside tc (bypass frozen using object.__setattr__)
        tc.tool_args["pattern"] = "MUTATED"
        # The copy must be unaffected
        assert running.tool_args["pattern"] == "main"


# ---------------------------------------------------------------------------
# ToolCallPart — state machine transitions
# ---------------------------------------------------------------------------


class TestToolCallPartStateMachine:
    def _make_tool_result(self, output: str = "found") -> ToolResult:
        return ToolResult(title="grep", output=output, metadata={"count": 1})

    def test_start_transition(self):
        tc = ToolCallPart(tool_name="grep")
        running = tc.start()
        assert running.state == "running"
        assert running.started_at is not None
        # Original unchanged
        assert tc.state == "pending"
        assert tc.started_at is None

    def test_complete_transition(self):
        tc = ToolCallPart(tool_name="grep")
        running = tc.start()
        result = self._make_tool_result("match at line 5")
        completed = running.complete(result)

        assert completed.state == "completed"
        assert completed.result == "match at line 5"
        assert completed.title == "grep"
        assert completed.metadata == {"count": 1}
        assert completed.completed_at is not None
        # Previous state unchanged
        assert running.state == "running"
        assert running.result is None

    def test_fail_transition(self):
        tc = ToolCallPart(tool_name="bash")
        running = tc.start()
        failed = running.fail("command not found")

        assert failed.state == "error"
        assert failed.error == "command not found"
        assert failed.completed_at is not None
        # Previous state unchanged
        assert running.state == "running"
        assert running.error is None

    def test_complete_metadata_is_independent_copy(self):
        """complete() must not share the ToolResult's metadata dict."""
        meta = {"count": 1}
        result = ToolResult(title="grep", output="x", metadata=meta)
        tc = ToolCallPart(tool_name="grep").start()
        completed = tc.complete(result)
        meta["count"] = 999
        assert completed.metadata["count"] == 1

    def test_invalid_start_from_running(self):
        running = ToolCallPart(tool_name="grep").start()
        with pytest.raises(ValueError, match="Expected 'pending'"):
            running.start()

    def test_invalid_complete_from_pending(self):
        tc = ToolCallPart(tool_name="grep")
        with pytest.raises(ValueError, match="Expected 'running'"):
            tc.complete(self._make_tool_result())

    def test_invalid_fail_from_pending(self):
        tc = ToolCallPart(tool_name="grep")
        with pytest.raises(ValueError, match="Expected 'running'"):
            tc.fail("oops")

    def test_invalid_complete_from_completed(self):
        completed = ToolCallPart(tool_name="grep").start().complete(
            self._make_tool_result()
        )
        with pytest.raises(ValueError, match="Expected 'running'"):
            completed.complete(self._make_tool_result())

    def test_json_round_trip_all_states(self):
        tc = ToolCallPart(tool_name="grep", tool_args={"pattern": "x"})
        for part in [tc, tc.start(), tc.start().fail("err")]:
            data = part.model_dump(mode="json")
            restored = ToolCallPart.model_validate(data)
            assert restored.tool_name == "grep"
            assert restored.state == part.state


# ---------------------------------------------------------------------------
# ToolResultPart
# ---------------------------------------------------------------------------


class TestToolResultPart:
    def test_from_tool_result(self):
        result = ToolResult(title="grep", output="match found", metadata={"count": 1})
        part = ToolResultPart.from_tool_result("call_x", "grep", result)
        assert part.tool_call_id == "call_x"
        assert part.tool_name == "grep"
        assert part.content == "match found"
        assert part.is_error is False
        assert part.metadata == {"count": 1}

    def test_from_tool_result_with_error(self):
        result = ToolResult(title="bash", output="Error: not found", metadata={})
        part = ToolResultPart.from_tool_result("call_y", "bash", result, is_error=True)
        assert part.is_error is True

    def test_metadata_is_independent_copy(self):
        meta = {"count": 1}
        result = ToolResult(title="x", output="x", metadata=meta)
        part = ToolResultPart.from_tool_result("call_x", "x", result)
        meta["count"] = 999
        assert part.metadata["count"] == 1

    def test_frozen(self):
        result = ToolResult(title="x", output="x", metadata={})
        part = ToolResultPart.from_tool_result("c", "x", result)
        with pytest.raises(Exception):
            part.content = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ErrorPart
# ---------------------------------------------------------------------------


class TestErrorPart:
    def test_basic(self):
        part = ErrorPart(content="max steps exceeded", code="max_steps_exceeded")
        assert part.type == "error"
        assert part.content == "max steps exceeded"
        assert part.code == "max_steps_exceeded"

    def test_code_optional(self):
        part = ErrorPart(content="something went wrong")
        assert part.code is None


# ---------------------------------------------------------------------------
# AgentMessage construction
# ---------------------------------------------------------------------------


class TestAgentMessageConstructors:
    def test_user(self):
        msg = AgentMessage.user("hello")
        assert msg.role == "user"
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "hello"

    def test_system(self):
        msg = AgentMessage.system("You are a code assistant.")
        assert msg.role == "system"
        assert msg.text == "You are a code assistant."

    def test_assistant_text(self):
        msg = AgentMessage.assistant_text("I found it.", model="gpt-4o")
        assert msg.role == "assistant"
        assert msg.model == "gpt-4o"
        assert msg.text == "I found it."

    def test_assistant_tool_calls_no_text(self):
        tc = ToolCallPart(tool_name="grep")
        msg = AgentMessage.assistant_tool_calls([tc])
        assert msg.role == "assistant"
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], ToolCallPart)
        assert msg.text is None

    def test_assistant_tool_calls_with_text(self):
        tc = ToolCallPart(tool_name="grep")
        msg = AgentMessage.assistant_tool_calls([tc], text="Let me search...")
        assert len(msg.parts) == 2
        assert isinstance(msg.parts[0], TextPart)
        assert isinstance(msg.parts[1], ToolCallPart)
        assert msg.text == "Let me search..."

    def test_assistant_tool_calls_multiple(self):
        tc1 = ToolCallPart(tool_name="grep")
        tc2 = ToolCallPart(tool_name="read")
        msg = AgentMessage.assistant_tool_calls([tc1, tc2])
        assert len(msg.parts) == 2
        assert msg.has_tool_calls

    def test_tool_result_constructor(self):
        result = ToolResult(title="grep", output="found", metadata={})
        msg = AgentMessage.tool_result("call_1", "grep", result)
        assert msg.role == "tool"
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], ToolResultPart)

    def test_empty_message_valid(self):
        msg = AgentMessage(role="user", parts=())
        assert msg.text is None
        assert msg.tool_calls == ()
        assert msg.has_tool_calls is False

    def test_id_is_unique(self):
        m1 = AgentMessage.user("a")
        m2 = AgentMessage.user("b")
        assert m1.id != m2.id


# ---------------------------------------------------------------------------
# AgentMessage properties
# ---------------------------------------------------------------------------


class TestAgentMessageProperties:
    def test_text_single_part(self):
        msg = AgentMessage.user("hello")
        assert msg.text == "hello"

    def test_text_multiple_text_parts(self):
        msg = AgentMessage(role="assistant", parts=(
            TextPart(content="part one"),
            TextPart(content="part two"),
        ))
        assert msg.text == "part one\npart two"

    def test_text_none_when_no_text_parts(self):
        tc = ToolCallPart(tool_name="grep")
        msg = AgentMessage.assistant_tool_calls([tc])
        assert msg.text is None

    def test_tool_calls_property(self):
        tc1 = ToolCallPart(tool_name="grep")
        tc2 = ToolCallPart(tool_name="read")
        msg = AgentMessage.assistant_tool_calls([tc1, tc2], text="ok")
        assert len(msg.tool_calls) == 2
        assert all(isinstance(t, ToolCallPart) for t in msg.tool_calls)

    def test_tool_results_property(self):
        result = ToolResult(title="x", output="x", metadata={})
        msg = AgentMessage.tool_result("c1", "grep", result)
        assert len(msg.tool_results) == 1

    def test_has_tool_calls_true(self):
        tc = ToolCallPart(tool_name="grep")
        msg = AgentMessage.assistant_tool_calls([tc])
        assert msg.has_tool_calls is True

    def test_has_tool_calls_false(self):
        msg = AgentMessage.user("hello")
        assert msg.has_tool_calls is False


# ---------------------------------------------------------------------------
# AgentMessage immutable update helpers
# ---------------------------------------------------------------------------


class TestAgentMessageUpdateHelpers:
    def test_with_updated_part(self):
        tc = ToolCallPart(tool_name="grep")
        msg = AgentMessage.assistant_tool_calls([tc], text="Searching...")
        running_tc = tc.start()
        updated = msg.with_updated_part(1, running_tc)
        assert updated.parts[1].state == "running"  # type: ignore[union-attr]
        assert msg.parts[1].state == "pending"  # original unchanged

    def test_with_updated_part_out_of_bounds(self):
        msg = AgentMessage.user("hello")
        with pytest.raises(IndexError):
            msg.with_updated_part(5, TextPart(content="nope"))

    def test_with_appended_part(self):
        msg = AgentMessage.user("hello")
        updated = msg.with_appended_part(ErrorPart(content="oops"))
        assert len(updated.parts) == 2
        assert isinstance(updated.parts[1], ErrorPart)
        assert len(msg.parts) == 1  # original unchanged

    def test_with_updated_part_preserves_identity(self):
        msg = AgentMessage.user("hello")
        updated = msg.with_updated_part(0, TextPart(content="bye"))
        assert updated.id == msg.id  # same message id
        assert updated.text == "bye"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_full_message_round_trip(self):
        tc = ToolCallPart(tool_name="grep", tool_args={"pattern": "main"})
        msg = AgentMessage.assistant_tool_calls([tc], text="Searching...")
        data = msg.model_dump(mode="json")
        restored = AgentMessage.model_validate(data)
        assert restored.role == "assistant"
        assert restored.text == "Searching..."
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].tool_name == "grep"

    def test_discriminated_union_deserialization(self):
        """Pydantic must correctly dispatch on the 'type' field."""
        data = {
            "id": "test-id",
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "hello"},
                {
                    "type": "tool_call",
                    "tool_call_id": "call_abc",
                    "tool_name": "grep",
                    "tool_args": {},
                    "state": "pending",
                },
                {"type": "error", "content": "oops"},
            ],
            "created_at": 1234567890.0,
        }
        msg = AgentMessage.model_validate(data)
        assert isinstance(msg.parts[0], TextPart)
        assert isinstance(msg.parts[1], ToolCallPart)
        assert isinstance(msg.parts[2], ErrorPart)

    def test_list_to_tuple_coercion(self):
        """field_validator must coerce list -> tuple during deserialization."""
        data = {
            "role": "user",
            "parts": [{"type": "text", "content": "hi"}],
        }
        msg = AgentMessage.model_validate(data)
        assert isinstance(msg.parts, tuple)


# ---------------------------------------------------------------------------
# from_chat_messages backward compatibility
# ---------------------------------------------------------------------------


class TestFromChatMessages:
    def test_basic_conversion(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = AgentMessage.from_chat_messages(messages)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].text == "hello"
        assert result[1].role == "assistant"
        assert result[1].text == "hi there"

    def test_system_role_preserved(self):
        messages = [{"role": "system", "content": "You are an expert."}]
        result = AgentMessage.from_chat_messages(messages)
        assert len(result) == 1
        assert result[0].role == "system"

    def test_unknown_role_skipped(self, caplog):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "robot", "content": "beep"},
        ]
        result = AgentMessage.from_chat_messages(messages)
        assert len(result) == 1
        assert "unknown role 'robot'" in caplog.text

    def test_tool_role_skipped_with_specific_warning(self, caplog):
        messages = [{"role": "tool", "content": "some result"}]
        result = AgentMessage.from_chat_messages(messages)
        assert len(result) == 0
        assert "cannot be converted from legacy format" in caplog.text

    def test_empty_list(self):
        assert AgentMessage.from_chat_messages([]) == []

    def test_preserves_order(self):
        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        result = AgentMessage.from_chat_messages(messages)
        assert [m.text for m in result] == ["msg1", "msg2", "msg3"]


# ---------------------------------------------------------------------------
# messages_to_openai_format
# ---------------------------------------------------------------------------


class TestMessagesToOpenaiFormat:
    def test_user_message(self):
        msgs = [AgentMessage.user("find main")]
        result = messages_to_openai_format(msgs)
        assert result == [{"role": "user", "content": "find main"}]

    def test_system_message(self):
        msgs = [AgentMessage.system("Be helpful.")]
        result = messages_to_openai_format(msgs)
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_assistant_text_only(self):
        msgs = [AgentMessage.assistant_text("I can help.")]
        result = messages_to_openai_format(msgs)
        assert result == [{"role": "assistant", "content": "I can help."}]

    def test_assistant_with_tool_calls(self):
        tc = ToolCallPart(
            tool_call_id="call_1",
            tool_name="grep",
            tool_args={"pattern": "main"},
        )
        msg = AgentMessage.assistant_tool_calls([tc], text="Let me search.")
        result = messages_to_openai_format([msg])
        assert len(result) == 1
        payload = result[0]
        assert payload["role"] == "assistant"
        assert payload["content"] == "Let me search."
        assert len(payload["tool_calls"]) == 1
        assert payload["tool_calls"][0]["id"] == "call_1"
        assert payload["tool_calls"][0]["type"] == "function"
        assert payload["tool_calls"][0]["function"]["name"] == "grep"
        parsed_args = json.loads(payload["tool_calls"][0]["function"]["arguments"])
        assert parsed_args == {"pattern": "main"}

    def test_assistant_tool_calls_without_text(self):
        tc = ToolCallPart(tool_call_id="call_1", tool_name="grep")
        msg = AgentMessage.assistant_tool_calls([tc])
        result = messages_to_openai_format([msg])
        # content=None is valid when tool_calls is present
        assert result[0]["content"] is None

    def test_tool_message_flattened(self):
        r1 = ToolResult(title="grep", output="match1", metadata={})
        r2 = ToolResult(title="read", output="match2", metadata={})
        msg1 = AgentMessage.tool_result("call_1", "grep", r1)
        msg2 = AgentMessage.tool_result("call_2", "read", r2)
        result = messages_to_openai_format([msg1, msg2])
        assert len(result) == 2
        assert result[0] == {"role": "tool", "tool_call_id": "call_1", "content": "match1"}
        assert result[1] == {"role": "tool", "tool_call_id": "call_2", "content": "match2"}

    def test_empty_user_message_sends_empty_string(self):
        msg = AgentMessage(role="user", parts=())
        result = messages_to_openai_format([msg])
        assert result[0]["content"] == ""

    def test_full_conversation(self):
        r = ToolResult(title="grep", output="found main", metadata={})
        msgs = [
            AgentMessage.system("You are an expert."),
            AgentMessage.user("find main"),
            AgentMessage.assistant_tool_calls(
                [ToolCallPart(tool_call_id="c1", tool_name="grep")],
                text="Searching...",
            ),
            AgentMessage.tool_result("c1", "grep", r),
            AgentMessage.assistant_text("Found it."),
        ]
        result = messages_to_openai_format(msgs)
        assert len(result) == 5
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_tool_message_with_no_results_warns(self, caplog):
        msg = AgentMessage(role="tool", parts=())
        result = messages_to_openai_format([msg])
        assert result == []
        assert "will be dropped" in caplog.text


# ---------------------------------------------------------------------------
# tool_call_part_from_openai
# ---------------------------------------------------------------------------


class TestToolCallPartFromOpenai:
    def test_normal_case(self):
        payload = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "grep",
                "arguments": json.dumps({"pattern": "def main", "path": "api/"}),
            },
        }
        part = tool_call_part_from_openai(payload)
        assert part.tool_call_id == "call_abc123"
        assert part.tool_name == "grep"
        assert part.tool_args == {"pattern": "def main", "path": "api/"}
        assert part.state == "pending"

    def test_malformed_arguments_json(self, caplog):
        payload = {
            "id": "call_bad",
            "function": {"name": "bash", "arguments": "NOT_JSON"},
        }
        part = tool_call_part_from_openai(payload)
        assert part.tool_args == {}
        assert "failed to parse arguments JSON" in caplog.text

    def test_non_dict_arguments(self, caplog):
        payload = {
            "id": "call_bad",
            "function": {"name": "bash", "arguments": "42"},
        }
        part = tool_call_part_from_openai(payload)
        assert part.tool_args == {}
        assert "non-dict type" in caplog.text

    def test_missing_tool_name_warns(self, caplog):
        payload = {"id": "call_x", "function": {"name": "", "arguments": "{}"}}
        part = tool_call_part_from_openai(payload)
        assert part.tool_name == "unknown"
        assert "missing or empty function name" in caplog.text

    def test_missing_id_generates_fallback(self):
        payload = {"function": {"name": "grep", "arguments": "{}"}}
        part = tool_call_part_from_openai(payload)
        assert part.tool_call_id.startswith("call_")

    def test_result_is_always_pending(self):
        payload = {
            "id": "call_x",
            "function": {"name": "grep", "arguments": "{}"},
        }
        part = tool_call_part_from_openai(payload)
        assert part.state == "pending"
        assert part.started_at is None
