"""
Unit tests for api/agent/stream_events.py.

Tests cover:
- Serialization / deserialization for each event type
- Discriminated union (StreamEvent) dispatch
- Frozen immutability
- Optional fields
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from api.agent.stream_events import (
    ErrorEvent,
    FinishEvent,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
)


# ---------------------------------------------------------------------------
# TextDelta
# ---------------------------------------------------------------------------


def test_text_delta_serialization():
    td = TextDelta(content="hello")
    data = td.model_dump()
    assert data == {"type": "text_delta", "content": "hello"}


def test_text_delta_roundtrip():
    td = TextDelta(content="world")
    assert TextDelta.model_validate(td.model_dump()) == td


def test_text_delta_is_frozen():
    td = TextDelta(content="hello")
    with pytest.raises((ValidationError, TypeError)):
        td.content = "bye"  # type: ignore[misc]


def test_text_delta_type_field_is_literal():
    td = TextDelta(content="x")
    assert td.type == "text_delta"


# ---------------------------------------------------------------------------
# ToolCallStart
# ---------------------------------------------------------------------------


def test_tool_call_start_fields():
    tcs = ToolCallStart(
        tool_call_id="call_abc123",
        tool_name="grep",
        tool_args={"pattern": "main", "path": "."},
    )
    assert tcs.type == "tool_call_start"
    assert tcs.tool_call_id == "call_abc123"
    assert tcs.tool_name == "grep"
    assert tcs.tool_args == {"pattern": "main", "path": "."}


def test_tool_call_start_default_args():
    tcs = ToolCallStart(tool_call_id="call_x", tool_name="ls")
    assert tcs.tool_args == {}


def test_tool_call_start_serialization():
    tcs = ToolCallStart(tool_call_id="c1", tool_name="read", tool_args={"path": "foo.py"})
    data = tcs.model_dump()
    assert data["type"] == "tool_call_start"
    assert data["tool_name"] == "read"
    assert data["tool_args"] == {"path": "foo.py"}


def test_tool_call_start_roundtrip():
    tcs = ToolCallStart(tool_call_id="c2", tool_name="bash", tool_args={"command": "ls"})
    assert ToolCallStart.model_validate(tcs.model_dump()) == tcs


def test_tool_call_start_is_frozen():
    tcs = ToolCallStart(tool_call_id="c3", tool_name="grep")
    with pytest.raises((ValidationError, TypeError)):
        tcs.tool_name = "read"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolCallDelta
# ---------------------------------------------------------------------------


def test_tool_call_delta_fields():
    tcd = ToolCallDelta(tool_call_id="call_xyz", args_delta='{"pat')
    assert tcd.type == "tool_call_delta"
    assert tcd.tool_call_id == "call_xyz"
    assert tcd.args_delta == '{"pat'


def test_tool_call_delta_serialization():
    tcd = ToolCallDelta(tool_call_id="c4", args_delta="tern")
    data = tcd.model_dump()
    assert data == {"type": "tool_call_delta", "tool_call_id": "c4", "args_delta": "tern"}


# ---------------------------------------------------------------------------
# FinishEvent
# ---------------------------------------------------------------------------


def test_finish_event_stop():
    f = FinishEvent(finish_reason="stop")
    assert f.type == "finish"
    assert f.finish_reason == "stop"
    assert f.usage is None


def test_finish_event_tool_calls():
    f = FinishEvent(finish_reason="tool_calls")
    assert f.finish_reason == "tool_calls"


def test_finish_event_with_usage():
    f = FinishEvent(
        finish_reason="stop",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    assert f.usage == {"prompt_tokens": 100, "completion_tokens": 50}


def test_finish_event_serialization():
    f = FinishEvent(finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5})
    data = f.model_dump()
    assert data["type"] == "finish"
    assert data["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 10


def test_finish_event_roundtrip():
    f = FinishEvent(finish_reason="tool_calls", usage={"prompt_tokens": 20, "completion_tokens": 30})
    assert FinishEvent.model_validate(f.model_dump()) == f


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------


def test_error_event_minimal():
    e = ErrorEvent(error="API key missing")
    assert e.type == "error"
    assert e.error == "API key missing"
    assert e.code is None


def test_error_event_with_code():
    e = ErrorEvent(error="Connection refused", code="provider_error")
    assert e.code == "provider_error"


def test_error_event_serialization():
    e = ErrorEvent(error="timeout", code="api_timeout")
    data = e.model_dump()
    assert data == {"type": "error", "error": "timeout", "code": "api_timeout"}


# ---------------------------------------------------------------------------
# StreamEvent discriminated union
# ---------------------------------------------------------------------------


def test_stream_event_discriminator_text_delta():
    adapter = TypeAdapter(StreamEvent)
    event = adapter.validate_python({"type": "text_delta", "content": "hi"})
    assert isinstance(event, TextDelta)
    assert event.content == "hi"


def test_stream_event_discriminator_tool_call_start():
    adapter = TypeAdapter(StreamEvent)
    event = adapter.validate_python({
        "type": "tool_call_start",
        "tool_call_id": "c1",
        "tool_name": "grep",
        "tool_args": {"pattern": "fn"},
    })
    assert isinstance(event, ToolCallStart)
    assert event.tool_name == "grep"


def test_stream_event_discriminator_tool_call_delta():
    adapter = TypeAdapter(StreamEvent)
    event = adapter.validate_python({
        "type": "tool_call_delta",
        "tool_call_id": "c2",
        "args_delta": '{"p',
    })
    assert isinstance(event, ToolCallDelta)


def test_stream_event_discriminator_finish():
    adapter = TypeAdapter(StreamEvent)
    event = adapter.validate_python({"type": "finish", "finish_reason": "stop"})
    assert isinstance(event, FinishEvent)


def test_stream_event_discriminator_error():
    adapter = TypeAdapter(StreamEvent)
    event = adapter.validate_python({"type": "error", "error": "oops"})
    assert isinstance(event, ErrorEvent)


def test_stream_event_invalid_type():
    adapter = TypeAdapter(StreamEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown_type", "content": "x"})


def test_stream_event_missing_type():
    adapter = TypeAdapter(StreamEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python({"content": "x"})
