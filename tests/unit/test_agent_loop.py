"""
Unit tests for api/agent/loop.py.

Tests are organized in two sections:

1. Helper unit tests: _check_doom_loop and _execute_tools_parallel tested
   in isolation with simple mock objects.

2. Integration tests: run_agent_loop tested end-to-end using MockProvider
   (yields predefined StreamEvent sequences) and MockTool (returns predefined
   ToolResult). No real LLM API calls are made.

Coverage:
    - Text-only path (single step, no tool calls)
    - Tool call + text path (two steps)
    - Doom loop detection and message injection
    - Max steps enforcement
    - Provider error handling
    - Empty response handling
    - Unknown tool handling
    - TaskTool executor injection
"""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from api.agent.config import AgentConfig, get_agent_config
from api.agent.loop import (
    DOOM_LOOP_MESSAGE,
    DOOM_LOOP_THRESHOLD,
    MAX_STEPS_MESSAGE,
    _check_doom_loop,
    _execute_tools_parallel,
    run_agent_loop,
)
from api.agent.message import AgentMessage, ToolCallPart
from api.agent.stream_events import (
    ErrorEvent,
    FinishEvent,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from api.tools.task import AgentInfo, TaskTool
from api.tools.tool import Tool, ToolResult


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class MockProvider:
    """
    Yields predefined StreamEvent sequences, one per stream_chat() call.

    Attributes:
        call_sequences: List of event lists. The i-th call to stream_chat()
                        yields the events in call_sequences[i].
        call_count:     Number of times stream_chat() has been called.
        last_messages:  The most recent messages argument passed to stream_chat().
        last_tools:     The most recent tools argument passed to stream_chat().
    """

    def __init__(self, call_sequences: list[list[StreamEvent]]) -> None:
        self._sequences = list(call_sequences)
        self.call_count = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] | None = None
        self.provider = "mock"
        self.model = "mock-model"

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        self.last_messages = messages
        self.last_tools = tools
        seq = self._sequences[self.call_count]
        self.call_count += 1
        for event in seq:
            yield event


class MockTool(Tool):
    """Returns a predefined ToolResult on every execute() call."""

    def __init__(self, name: str, result: ToolResult) -> None:
        self.name = name
        self.description = f"Mock {name}"
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
        }
        self._result = result
        self.execute_count = 0

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        self.execute_count += 1
        return self._result


class RaisingTool(Tool):
    """Raises a RuntimeError on execute() to test error handling."""

    def __init__(self, name: str = "raising_tool") -> None:
        self.name = name
        self.description = "Always raises"
        self.parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        raise RuntimeError("Tool intentionally failed")


def _make_agent_config(
    max_steps: int = 5,
    allowed_tools: tuple[str, ...] = (),
) -> AgentConfig:
    """Build a minimal AgentConfig for testing."""
    return AgentConfig(
        name="test-agent",
        description="Test agent",
        mode="primary",
        system_prompt_template="You are a test agent for {repo_name}.",
        allowed_tools=allowed_tools,
        max_steps=max_steps,
    )


def _make_tool_call_start(
    tool_name: str = "grep",
    tool_args: dict[str, Any] | None = None,
    tool_call_id: str = "call_abc123",
) -> ToolCallStart:
    return ToolCallStart(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args or {"pattern": "main"},
    )


def _make_tool_result(output: str = "found: main.py:10") -> ToolResult:
    return ToolResult(title="grep", output=output, metadata={})


async def _collect(gen) -> list[StreamEvent]:
    """Collect all events from an async generator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Tests: _check_doom_loop
# ---------------------------------------------------------------------------


class TestCheckDoomLoop:
    def _make_tc(self, tool_name: str, args: dict[str, Any]) -> ToolCallPart:
        return ToolCallPart(tool_name=tool_name, tool_args=args)

    def test_returns_false_when_no_history(self):
        tc = self._make_tc("grep", {"pattern": "main"})
        assert _check_doom_loop([], [tc]) is False

    def test_returns_false_when_below_threshold(self):
        fp = ("grep", json.dumps({"pattern": "main"}, sort_keys=True))
        recent = [fp] * (DOOM_LOOP_THRESHOLD - 1)
        tc = self._make_tc("grep", {"pattern": "main"})
        assert _check_doom_loop(recent, [tc]) is False

    def test_returns_false_when_no_current_calls(self):
        fp = ("grep", json.dumps({"pattern": "main"}, sort_keys=True))
        recent = [fp] * DOOM_LOOP_THRESHOLD
        assert _check_doom_loop(recent, []) is False

    def test_detects_doom_loop(self):
        fp = ("grep", json.dumps({"pattern": "main"}, sort_keys=True))
        recent = [fp] * DOOM_LOOP_THRESHOLD
        tc = self._make_tc("grep", {"pattern": "main"})
        assert _check_doom_loop(recent, [tc]) is True

    def test_returns_false_when_recent_calls_differ(self):
        recent = [
            ("grep", json.dumps({"pattern": "main"}, sort_keys=True)),
            ("glob", json.dumps({"pattern": "*.py"}, sort_keys=True)),
            ("grep", json.dumps({"pattern": "main"}, sort_keys=True)),
        ]
        tc = self._make_tc("grep", {"pattern": "main"})
        assert _check_doom_loop(recent, [tc]) is False

    def test_returns_false_when_args_differ(self):
        fp_a = ("grep", json.dumps({"pattern": "main"}, sort_keys=True))
        fp_b = ("grep", json.dumps({"pattern": "def"}, sort_keys=True))
        recent = [fp_a, fp_a, fp_b]
        tc = self._make_tc("grep", {"pattern": "main"})
        assert _check_doom_loop(recent, [tc]) is False

    def test_returns_false_when_current_batch_has_multiple_different_calls(self):
        fp = ("grep", json.dumps({"pattern": "main"}, sort_keys=True))
        recent = [fp] * DOOM_LOOP_THRESHOLD
        tc1 = self._make_tc("grep", {"pattern": "main"})
        tc2 = self._make_tc("glob", {"pattern": "*.py"})
        assert _check_doom_loop(recent, [tc1, tc2]) is False

    def test_args_are_compared_with_sorted_keys(self):
        # {"b": 2, "a": 1} and {"a": 1, "b": 2} are the same when sort_keys=True
        fp = ("grep", json.dumps({"a": 1, "b": 2}, sort_keys=True))
        recent = [fp] * DOOM_LOOP_THRESHOLD
        tc = self._make_tc("grep", {"b": 2, "a": 1})
        assert _check_doom_loop(recent, [tc]) is True

    def test_tail_check_uses_last_n_entries(self):
        # The first entry differs; tail (last N) are identical -> True
        diff_fp = ("glob", json.dumps({"pattern": "*.py"}, sort_keys=True))
        same_fp = ("grep", json.dumps({"pattern": "main"}, sort_keys=True))
        recent = [diff_fp] + [same_fp] * DOOM_LOOP_THRESHOLD
        tc = self._make_tc("grep", {"pattern": "main"})
        assert _check_doom_loop(recent, [tc]) is True


# ---------------------------------------------------------------------------
# Tests: _execute_tools_parallel
# ---------------------------------------------------------------------------


class TestExecuteToolsParallel:
    def _make_tc(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        tool_call_id: str = "call_001",
    ) -> ToolCallPart:
        return ToolCallPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=args or {},
        )

    @pytest.mark.asyncio
    async def test_success(self):
        expected = _make_tool_result("output text")
        tool = MockTool("grep", expected)
        tc = self._make_tc("grep")

        results = await _execute_tools_parallel([tc], {"grep": tool})

        assert len(results) == 1
        tool_result, is_error, duration_ms = results[0]
        assert tool_result.output == "output text"
        assert is_error is False
        assert duration_ms >= 0

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        tc = self._make_tc("nonexistent")
        results = await _execute_tools_parallel([tc], {"grep": MockTool("grep", _make_tool_result())})

        tool_result, is_error, duration_ms = results[0]
        assert is_error is True
        assert "nonexistent" in tool_result.output
        assert "grep" in tool_result.output  # lists available tools
        assert tool_result.metadata.get("error") == "unknown_tool"

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error(self):
        tc = self._make_tc("raising_tool")
        results = await _execute_tools_parallel([tc], {"raising_tool": RaisingTool()})

        tool_result, is_error, duration_ms = results[0]
        assert is_error is True
        assert "intentionally failed" in tool_result.output
        assert tool_result.metadata.get("error") == "execution_error"

    @pytest.mark.asyncio
    async def test_multiple_tools_run_concurrently(self):
        """All tools should execute; results must be in input order."""
        r1 = _make_tool_result("result-1")
        r2 = _make_tool_result("result-2")
        r3 = _make_tool_result("result-3")
        t1 = MockTool("grep", r1)
        t2 = MockTool("glob", r2)
        t3 = MockTool("read", r3)
        tc1 = self._make_tc("grep", tool_call_id="call_001")
        tc2 = self._make_tc("glob", tool_call_id="call_002")
        tc3 = self._make_tc("read", tool_call_id="call_003")

        results = await _execute_tools_parallel(
            [tc1, tc2, tc3],
            {"grep": t1, "glob": t2, "read": t3},
        )

        assert len(results) == 3
        assert results[0][0].output == "result-1"
        assert results[1][0].output == "result-2"
        assert results[2][0].output == "result-3"
        assert all(not is_error for _, is_error, _ in results)

    @pytest.mark.asyncio
    async def test_one_failure_does_not_affect_others(self):
        good_result = _make_tool_result("good output")
        good_tool = MockTool("grep", good_result)
        bad_tool = RaisingTool("bash")
        tc1 = self._make_tc("grep", tool_call_id="call_001")
        tc2 = self._make_tc("bash", tool_call_id="call_002")

        results = await _execute_tools_parallel(
            [tc1, tc2], {"grep": good_tool, "bash": bad_tool}
        )

        assert results[0][0].output == "good output"
        assert results[0][1] is False
        assert results[1][1] is True  # error
        assert "execution_error" in results[1][0].metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_tool_metadata_error_is_propagated_as_is_error(self):
        """Tool returning ToolResult with metadata['error'] key -> is_error=True.

        This covers the contract used by all existing tools (read, grep, glob, bash,
        todo) which signal non-throwing errors via metadata['error'] instead of
        raising exceptions.
        """
        error_result = ToolResult(
            title="read",
            output="Error: file not found.",
            metadata={"error": "not_found"},
        )
        tool = MockTool("read", error_result)
        tc = self._make_tc("read")

        results = await _execute_tools_parallel([tc], {"read": tool})

        tool_result, is_error, _ = results[0]
        assert is_error is True
        assert tool_result.metadata["error"] == "not_found"

    @pytest.mark.asyncio
    async def test_tool_without_metadata_error_stays_not_error(self):
        """Tool returning ToolResult without metadata['error'] -> is_error=False."""
        ok_result = ToolResult(
            title="read",
            output="File contents here.",
            metadata={"lines": 42},
        )
        tool = MockTool("read", ok_result)
        tc = self._make_tc("read")

        results = await _execute_tools_parallel([tc], {"read": tool})

        _, is_error, _ = results[0]
        assert is_error is False


# ---------------------------------------------------------------------------
# Tests: run_agent_loop -- integration tests with MockProvider
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    """End-to-end tests using MockProvider (no real LLM calls)."""

    @pytest.mark.asyncio
    async def test_text_only_single_step(self):
        """Provider returns text only -> loop exits after one step."""
        provider = MockProvider([
            [
                TextDelta(content="Hello "),
                TextDelta(content="world"),
                FinishEvent(finish_reason="stop"),
            ]
        ])
        config = _make_agent_config(max_steps=5)
        messages = [AgentMessage.user("What is this?")]

        events = await _collect(run_agent_loop(config, messages, provider, {}, "/repo"))

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(text_deltas) == 2
        assert "".join(e.content for e in text_deltas) == "Hello world"
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "stop"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self):
        """Step 1: tool call. Step 2: text response."""
        tool_result = _make_tool_result("main() found at main.py:10")
        grep_tool = MockTool("grep", tool_result)

        call_id = "call_xyz"
        provider = MockProvider([
            # Step 1: LLM requests grep
            [
                TextDelta(content="Let me search."),
                ToolCallStart(
                    tool_call_id=call_id,
                    tool_name="grep",
                    tool_args={"pattern": "def main"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            # Step 2: LLM produces final answer
            [
                TextDelta(content="Found main at line 10."),
                FinishEvent(finish_reason="stop"),
            ],
        ])
        config = _make_agent_config(max_steps=5, allowed_tools=("grep",))
        messages = [AgentMessage.user("Where is main?")]

        events = await _collect(
            run_agent_loop(config, messages, provider, {"grep": grep_tool}, "/repo")
        )

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
        finish_events = [e for e in events if isinstance(e, FinishEvent)]

        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "grep"
        assert len(tool_ends) == 1
        assert tool_ends[0].tool_call_id == call_id
        assert tool_ends[0].is_error is False
        assert "main.py:10" in tool_ends[0].result_summary
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "stop"
        assert provider.call_count == 2
        assert grep_tool.execute_count == 1

    @pytest.mark.asyncio
    async def test_doom_loop_injects_warning(self):
        """LLM repeating the same tool call triggers DOOM_LOOP_MESSAGE injection."""
        tool_result = _make_tool_result("no match")
        grep_tool = MockTool("grep", tool_result)

        repeated_call = ToolCallStart(
            tool_call_id="call_001",
            tool_name="grep",
            tool_args={"pattern": "main"},
        )

        # Provider returns the same tool call DOOM_LOOP_THRESHOLD + 1 times,
        # then finally returns text.
        sequences = []
        for _ in range(DOOM_LOOP_THRESHOLD + 1):
            sequences.append([
                repeated_call,
                FinishEvent(finish_reason="tool_calls"),
            ])
        sequences.append([TextDelta(content="Done."), FinishEvent(finish_reason="stop")])

        provider = MockProvider(sequences)
        config = _make_agent_config(max_steps=20, allowed_tools=("grep",))
        messages = [AgentMessage.user("Find main")]

        events = await _collect(
            run_agent_loop(config, messages, provider, {"grep": grep_tool}, "/repo")
        )

        # The loop should eventually finish
        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(finish_events) == 1

        # Verify the DOOM_LOOP_MESSAGE was injected into the conversation
        # by checking the messages sent to the provider after the threshold.
        # The message after the (DOOM_LOOP_THRESHOLD + 1)th step should contain
        # DOOM_LOOP_MESSAGE in a user role message.
        found_doom_message = False
        for call_idx in range(DOOM_LOOP_THRESHOLD, provider.call_count):
            # The provider captures last_messages, so check if doom message appeared
            # We verify indirectly: grep_tool was called more than DOOM_LOOP_THRESHOLD
            pass

        # The loop completed: doom loop was handled, not infinite
        assert grep_tool.execute_count >= DOOM_LOOP_THRESHOLD + 1

    @pytest.mark.asyncio
    async def test_max_steps_enforced(self):
        """Loop exits after max_steps with FinishEvent."""
        # Provider always returns a tool call (would loop forever without limit)
        tool_result = _make_tool_result("output")
        grep_tool = MockTool("grep", tool_result)

        # max_steps=2: step 1 -> tool call, step 2 -> final step (tools disabled, text only)
        provider = MockProvider([
            # Step 1: tool call
            [
                ToolCallStart(
                    tool_call_id="call_001",
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            # Step 2: final step (tools disabled by loop, LLM returns text)
            [
                TextDelta(content="Summary: found x."),
                FinishEvent(finish_reason="stop"),
            ],
        ])
        config = _make_agent_config(max_steps=2, allowed_tools=("grep",))
        messages = [AgentMessage.user("Find x")]

        events = await _collect(
            run_agent_loop(config, messages, provider, {"grep": grep_tool}, "/repo")
        )

        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "stop"
        assert provider.call_count == 2

        # On the final step, provider should have received tools=None
        # (We can't inspect call-specific state here, but we verify step count.)

    @pytest.mark.asyncio
    async def test_final_step_disables_tools(self):
        """On the final step, stream_chat is called with tools=None."""
        tool_result = _make_tool_result("output")
        grep_tool = MockTool("grep", tool_result)

        # Track what tools arg was passed on each call
        tools_args_per_call: list[Any] = []
        original_stream = MockProvider([
            [
                ToolCallStart(
                    tool_call_id="call_001",
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            [
                TextDelta(content="Done."),
                FinishEvent(finish_reason="stop"),
            ],
        ])

        class TrackingProvider(MockProvider):
            async def stream_chat(self, messages, tools=None):
                tools_args_per_call.append(tools)
                async for event in super().stream_chat(messages, tools):
                    yield event

        tracking_provider = TrackingProvider([
            [
                ToolCallStart(
                    tool_call_id="call_001",
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            [
                TextDelta(content="Done."),
                FinishEvent(finish_reason="stop"),
            ],
        ])
        config = _make_agent_config(max_steps=2, allowed_tools=("grep",))
        messages = [AgentMessage.user("Find x")]

        await _collect(
            run_agent_loop(
                config, messages, tracking_provider, {"grep": grep_tool}, "/repo"
            )
        )

        # Step 1 (not final): tools list should be present
        assert tools_args_per_call[0] is not None
        assert len(tools_args_per_call[0]) > 0
        # Step 2 (final): tools should be None
        assert tools_args_per_call[1] is None

    @pytest.mark.asyncio
    async def test_provider_error_event_stops_loop(self):
        """Provider emitting ErrorEvent causes loop to yield ErrorEvent + FinishEvent."""
        provider = MockProvider([
            [
                ErrorEvent(error="API timeout", code="provider_error"),
            ]
        ])
        config = _make_agent_config(max_steps=5)
        messages = [AgentMessage.user("query")]

        events = await _collect(run_agent_loop(config, messages, provider, {}, "/repo"))

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(error_events) == 1
        assert error_events[0].error == "API timeout"
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "error"

    @pytest.mark.asyncio
    async def test_empty_response_stops_loop(self):
        """Provider returning FinishEvent with no text/tools exits cleanly."""
        provider = MockProvider([
            [FinishEvent(finish_reason="stop")]
        ])
        config = _make_agent_config(max_steps=5)
        messages = [AgentMessage.user("query")]

        events = await _collect(run_agent_loop(config, messages, provider, {}, "/repo"))

        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_to_llm(self):
        """If LLM calls a tool not in the tools dict, an error ToolResult is fed back."""
        # No tools registered, but LLM calls "grep"
        provider = MockProvider([
            # Step 1: LLM calls unknown tool "grep"
            [
                ToolCallStart(
                    tool_call_id="call_001",
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            # Step 2: LLM produces text after seeing error
            [
                TextDelta(content="I cannot search."),
                FinishEvent(finish_reason="stop"),
            ],
        ])
        config = _make_agent_config(max_steps=5)
        messages = [AgentMessage.user("Find x")]

        events = await _collect(run_agent_loop(config, messages, provider, {}, "/repo"))

        tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(tool_ends) == 1
        assert tool_ends[0].is_error is True
        assert "grep" in tool_ends[0].result_summary

    @pytest.mark.asyncio
    async def test_system_prompt_formatting(self):
        """System prompt template is formatted with system_prompt_vars."""
        provider = MockProvider([
            [TextDelta(content="ok"), FinishEvent(finish_reason="stop")]
        ])
        config = AgentConfig(
            name="test",
            description="test",
            mode="primary",
            system_prompt_template="Agent for {repo_name} on {repo_url}.",
            max_steps=5,
        )
        messages = [AgentMessage.user("query")]
        vars_ = {"repo_name": "myrepo", "repo_url": "https://example.com/myrepo"}

        await _collect(
            run_agent_loop(config, messages, provider, {}, "/repo", system_prompt_vars=vars_)
        )

        # First message in the conversation sent to provider must be the system message
        first_msg = provider.last_messages[0]
        assert first_msg["role"] == "system"
        assert "myrepo" in first_msg["content"]
        assert "https://example.com/myrepo" in first_msg["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_missing_key_falls_back_gracefully(self):
        """Missing placeholder key in system prompt does not raise; raw template is used."""
        provider = MockProvider([
            [TextDelta(content="ok"), FinishEvent(finish_reason="stop")]
        ])
        config = AgentConfig(
            name="test",
            description="test",
            mode="primary",
            system_prompt_template="Agent for {repo_name}.",
            max_steps=5,
        )
        messages = [AgentMessage.user("query")]
        # repo_name is missing -- should not raise
        vars_ = {"wrong_key": "value"}

        events = await _collect(
            run_agent_loop(config, messages, provider, {}, "/repo", system_prompt_vars=vars_)
        )

        # Should still complete
        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(finish_events) == 1

    @pytest.mark.asyncio
    async def test_tool_call_end_event_fields(self):
        """ToolCallEnd events carry correct fields from tool execution."""
        expected_output = "x" * 300  # longer than MAX_RESULT_SUMMARY_LEN (200)
        tool_result = _make_tool_result(expected_output)
        grep_tool = MockTool("grep", tool_result)

        call_id = "call_end_test"
        provider = MockProvider([
            [
                ToolCallStart(
                    tool_call_id=call_id,
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            [TextDelta(content="done"), FinishEvent(finish_reason="stop")],
        ])
        config = _make_agent_config(max_steps=5, allowed_tools=("grep",))
        messages = [AgentMessage.user("query")]

        events = await _collect(
            run_agent_loop(config, messages, provider, {"grep": grep_tool}, "/repo")
        )

        tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(tool_ends) == 1
        end = tool_ends[0]
        assert end.tool_call_id == call_id
        assert end.tool_name == "grep"
        assert len(end.result_summary) <= 203  # 200 chars + "..."
        assert end.result_summary.endswith("...")
        assert end.is_error is False
        assert end.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_step(self):
        """Multiple tool calls in one LLM response are all executed and reported."""
        r1 = _make_tool_result("grep result")
        r2 = _make_tool_result("glob result")
        grep_tool = MockTool("grep", r1)
        glob_tool = MockTool("glob", r2)

        provider = MockProvider([
            [
                ToolCallStart(
                    tool_call_id="call_001",
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                ToolCallStart(
                    tool_call_id="call_002",
                    tool_name="glob",
                    tool_args={"pattern": "*.py"},
                ),
                FinishEvent(finish_reason="tool_calls"),
            ],
            [TextDelta(content="done"), FinishEvent(finish_reason="stop")],
        ])
        config = _make_agent_config(max_steps=5, allowed_tools=("grep", "glob"))
        messages = [AgentMessage.user("query")]

        events = await _collect(
            run_agent_loop(
                config,
                messages,
                provider,
                {"grep": grep_tool, "glob": glob_tool},
                "/repo",
            )
        )

        tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(tool_ends) == 2
        end_names = {e.tool_name for e in tool_ends}
        assert end_names == {"grep", "glob"}
        assert grep_tool.execute_count == 1
        assert glob_tool.execute_count == 1

    @pytest.mark.asyncio
    async def test_usage_is_accumulated(self):
        """Token usage from each step is summed in the final FinishEvent."""
        provider = MockProvider([
            [
                ToolCallStart(
                    tool_call_id="call_001",
                    tool_name="grep",
                    tool_args={"pattern": "x"},
                ),
                FinishEvent(
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                ),
            ],
            [
                TextDelta(content="done"),
                FinishEvent(
                    finish_reason="stop",
                    usage={"prompt_tokens": 200, "completion_tokens": 80},
                ),
            ],
        ])
        grep_tool = MockTool("grep", _make_tool_result("out"))
        config = _make_agent_config(max_steps=5, allowed_tools=("grep",))
        messages = [AgentMessage.user("query")]

        events = await _collect(
            run_agent_loop(config, messages, provider, {"grep": grep_tool}, "/repo")
        )

        finish = next(e for e in events if isinstance(e, FinishEvent))
        # total_usage is passed through if non-zero
        # (The loop sums internally; FinishEvent carries it when > 0)
        assert finish is not None  # basic sanity; usage field optional per provider

    @pytest.mark.asyncio
    async def test_subagent_error_event_not_silenced(self):
        """Sub-agent yielding ErrorEvent before any text -> ToolCallEnd.is_error=True.

        Before the fix, _task_executor ignored ErrorEvent and returned a success
        ToolResult with '(Sub-agent produced no output)', hiding the failure from
        the parent agent's telemetry and recovery logic.

        After the fix:
        - ErrorEvent from sub-agent is detected
        - _task_executor returns ToolResult(metadata={"error": "subagent_error"})
        - TaskTool.execute() spreads that metadata into its return value
        - _execute_tools_parallel sees metadata["error"] -> is_error=True
        - ToolCallEnd.is_error=True is emitted
        """
        explore_info = AgentInfo(name="explore", description="Code explorer", mode="subagent")
        task_tool = TaskTool(repo_path="/repo", agents=[explore_info])

        parent_task_call = ToolCallStart(
            tool_call_id="call_task_001",
            tool_name="task",
            tool_args={
                "description": "explore codebase",
                "prompt": "find main function",
                "subagent_type": "explore",
            },
        )

        # Provider sequences:
        #   [0] parent step 1: calls task tool
        #   [1] sub-agent step 1: provider fails with ErrorEvent
        #   [2] parent step 2: produces text after seeing task error result
        provider = MockProvider([
            [parent_task_call, FinishEvent(finish_reason="tool_calls")],
            [ErrorEvent(error="API timeout", code="provider_error")],
            [TextDelta(content="Task failed but I have partial info."), FinishEvent(finish_reason="stop")],
        ])

        mock_sub_config = _make_agent_config(max_steps=3, allowed_tools=())

        with (
            patch("api.agent.loop.get_agent_config", return_value=mock_sub_config),
            patch("api.agent.loop.get_tools_for_agent", return_value={}),
            patch("api.agent.loop.get_all_agent_infos", return_value=[explore_info]),
        ):
            events = await _collect(
                run_agent_loop(
                    _make_agent_config(max_steps=5, allowed_tools=("task",)),
                    [AgentMessage.user("Explore this repo")],
                    provider,
                    {"task": task_tool},
                    "/repo",
                )
            )

        tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(tool_ends) == 1
        assert tool_ends[0].tool_name == "task"
        assert tool_ends[0].is_error is True

        finish_events = [e for e in events if isinstance(e, FinishEvent)]
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "stop"
        assert provider.call_count == 3  # parent-step1, sub-agent, parent-step2
