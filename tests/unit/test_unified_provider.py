"""
Unit tests for api/agent/provider.py.

Tests cover pure / stateless module-level functions that do NOT require
network access or API keys:
  - _build_tools_in_prompt
  - _inject_tools_prompt
  - _parse_tool_calls_from_text
  - _extract_ollama_text
  - _inject_ollama_no_think
  - _messages_to_bedrock_prompt
  - _openai_tools_to_google_tools
  - _messages_to_google_contents (incl. tool_calls and role="tool" conversion)
  - UnifiedProvider.__init__ (with mocked get_model_config)
  - UnifiedProvider._supports_native_tools
  - _stream_openai_compat async_client initialization (regression for Fix 1)

Integration tests (requiring actual API keys) are NOT included here —
see tests/integration/ for those.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.agent.provider import (
    UnifiedProvider,
    _build_tools_in_prompt,
    _extract_ollama_text,
    _inject_ollama_no_think,
    _inject_tools_prompt,
    _messages_to_bedrock_prompt,
    _messages_to_google_contents,
    _openai_tools_to_google_tools,
    _parse_tool_calls_from_text,
)
from api.agent.stream_events import ToolCallStart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search code with ripgrep",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory to search"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# _build_tools_in_prompt
# ---------------------------------------------------------------------------


def test_build_tools_in_prompt_contains_tool_names():
    result = _build_tools_in_prompt(SAMPLE_TOOLS)
    assert "grep" in result
    assert "read" in result


def test_build_tools_in_prompt_contains_tool_call_tag():
    result = _build_tools_in_prompt(SAMPLE_TOOLS)
    assert "<tool_call>" in result
    assert "</tool_call>" in result


def test_build_tools_in_prompt_contains_parameters():
    result = _build_tools_in_prompt(SAMPLE_TOOLS)
    assert "pattern" in result
    assert "path" in result


def test_build_tools_in_prompt_marks_required():
    result = _build_tools_in_prompt(SAMPLE_TOOLS)
    # "pattern" is required for grep
    assert "required" in result


def test_build_tools_in_prompt_empty_tools():
    result = _build_tools_in_prompt([])
    assert "## Available Tools" in result
    assert "<tool_call>" in result


def test_build_tools_in_prompt_tool_without_parameters():
    tools = [{"type": "function", "function": {"name": "ls", "description": "List files"}}]
    result = _build_tools_in_prompt(tools)
    assert "ls" in result
    assert "List files" in result


# ---------------------------------------------------------------------------
# _inject_tools_prompt
# ---------------------------------------------------------------------------


def test_inject_tools_prompt_appends_to_existing_system():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    result = _inject_tools_prompt(messages, "TOOLS HERE")

    assert "TOOLS HERE" in result[0]["content"]
    assert "You are helpful." in result[0]["content"]
    assert len(result) == 2


def test_inject_tools_prompt_does_not_mutate_original():
    original_system = {"role": "system", "content": "Original."}
    messages = [original_system, {"role": "user", "content": "Hi"}]
    _inject_tools_prompt(messages, "TOOLS")

    # Original dict must not be modified
    assert original_system["content"] == "Original."


def test_inject_tools_prompt_prepends_system_when_missing():
    messages = [{"role": "user", "content": "Hi"}]
    result = _inject_tools_prompt(messages, "TOOLS HERE")

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "TOOLS HERE"
    assert result[1] == messages[0]
    assert len(result) == 2


def test_inject_tools_prompt_returns_new_list():
    messages = [{"role": "user", "content": "Hi"}]
    result = _inject_tools_prompt(messages, "TOOLS")
    assert result is not messages


def test_inject_tools_prompt_user_message_unchanged():
    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Question?"},
    ]
    result = _inject_tools_prompt(messages, "TOOLS")
    assert result[1] == messages[1]


# ---------------------------------------------------------------------------
# _parse_tool_calls_from_text
# ---------------------------------------------------------------------------


def test_parse_single_tool_call():
    text = (
        "Let me search.\n"
        "<tool_call>\n"
        '{"name": "grep", "arguments": {"pattern": "main"}}\n'
        "</tool_call>\n"
        "Done."
    )
    clean, calls = _parse_tool_calls_from_text(text)

    assert len(calls) == 1
    assert isinstance(calls[0], ToolCallStart)
    assert calls[0].tool_name == "grep"
    assert calls[0].tool_args == {"pattern": "main"}


def test_parse_tool_calls_removes_blocks_from_text():
    text = (
        "Here is the result.\n"
        "<tool_call>\n"
        '{"name": "ls"}\n'
        "</tool_call>\n"
        "More text."
    )
    clean, _ = _parse_tool_calls_from_text(text)
    assert "<tool_call>" not in clean
    assert "Here is the result." in clean
    assert "More text." in clean


def test_parse_multiple_tool_calls():
    text = (
        "<tool_call>\n"
        '{"name": "grep", "arguments": {"pattern": "foo"}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "read", "arguments": {"path": "bar.py"}}\n'
        "</tool_call>"
    )
    _, calls = _parse_tool_calls_from_text(text)
    assert len(calls) == 2
    assert calls[0].tool_name == "grep"
    assert calls[1].tool_name == "read"


def test_parse_tool_calls_malformed_json_skipped():
    text = (
        "<tool_call>\n"
        "{invalid json here}\n"
        "</tool_call>\n"
        "Some good text."
    )
    clean, calls = _parse_tool_calls_from_text(text)
    assert len(calls) == 0
    assert "Some good text." in clean


def test_parse_tool_calls_no_tool_calls():
    text = "This is a normal response with no tool calls."
    clean, calls = _parse_tool_calls_from_text(text)
    assert len(calls) == 0
    assert clean == text


def test_parse_tool_calls_generates_unique_ids():
    text = (
        "<tool_call>\n"
        '{"name": "grep", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "read", "arguments": {}}\n'
        "</tool_call>"
    )
    _, calls = _parse_tool_calls_from_text(text)
    assert calls[0].tool_call_id != calls[1].tool_call_id


def test_parse_tool_calls_missing_name_defaults_to_unknown():
    text = '<tool_call>\n{"arguments": {"x": 1}}\n</tool_call>'
    _, calls = _parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "unknown"


def test_parse_tool_calls_missing_arguments_defaults_to_empty():
    text = '<tool_call>\n{"name": "ls"}\n</tool_call>'
    _, calls = _parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].tool_args == {}


# ---------------------------------------------------------------------------
# _extract_ollama_text
# ---------------------------------------------------------------------------


def test_extract_ollama_text_dict_path():
    chunk = {"message": {"content": "hello world"}}
    assert _extract_ollama_text(chunk) == "hello world"


def test_extract_ollama_text_attribute_path():
    chunk = MagicMock()
    chunk.message.content = "hello"
    # Ensure isinstance(chunk, dict) is False
    assert not isinstance(chunk, dict)
    result = _extract_ollama_text(chunk)
    assert result == "hello"


def test_extract_ollama_text_response_attribute():
    chunk = MagicMock(spec=["response"])
    chunk.response = "fallback text"
    result = _extract_ollama_text(chunk)
    assert result == "fallback text"


def test_extract_ollama_text_strips_think_tags():
    chunk = {"message": {"content": "<think>reasoning</think>actual answer"}}
    result = _extract_ollama_text(chunk)
    assert result == "reasoningactual answer"


def test_extract_ollama_text_filters_model_metadata():
    chunk = {"message": {"content": "model=qwen3:1.7b"}}
    assert _extract_ollama_text(chunk) is None


def test_extract_ollama_text_filters_created_at_metadata():
    chunk = {"message": {"content": "created_at=2024-01-01"}}
    assert _extract_ollama_text(chunk) is None


def test_extract_ollama_text_none_content():
    chunk = {"message": {"content": None}}
    assert _extract_ollama_text(chunk) is None


def test_extract_ollama_text_empty_content():
    chunk = {"message": {"content": ""}}
    assert _extract_ollama_text(chunk) is None


# ---------------------------------------------------------------------------
# _inject_ollama_no_think
# ---------------------------------------------------------------------------


def test_inject_ollama_no_think_appends_to_last_user():
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is this?"},
    ]
    result = _inject_ollama_no_think(messages)

    assert result[-1]["content"] == "What is this? /no_think"
    # Original unchanged
    assert messages[-1]["content"] == "What is this?"


def test_inject_ollama_no_think_returns_new_list():
    messages = [{"role": "user", "content": "hi"}]
    result = _inject_ollama_no_think(messages)
    assert result is not messages


def test_inject_ollama_no_think_multi_turn():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Follow-up"},
    ]
    result = _inject_ollama_no_think(messages)

    # Only the last user message should be modified
    assert result[0]["content"] == "First"
    assert result[1]["content"] == "Response"
    assert result[2]["content"] == "Follow-up /no_think"


def test_inject_ollama_no_think_empty_messages():
    assert _inject_ollama_no_think([]) == []


# ---------------------------------------------------------------------------
# _messages_to_bedrock_prompt
# ---------------------------------------------------------------------------


def test_bedrock_prompt_basic():
    messages = [{"role": "user", "content": "Hello"}]
    result = _messages_to_bedrock_prompt(messages)
    assert "Human: Hello" in result
    assert result.endswith("Assistant: ")


def test_bedrock_prompt_with_system():
    messages = [
        {"role": "system", "content": "You are a code expert."},
        {"role": "user", "content": "Explain this"},
    ]
    result = _messages_to_bedrock_prompt(messages)
    assert "You are a code expert." in result
    assert "Human: Explain this" in result
    assert result.endswith("Assistant: ")


def test_bedrock_prompt_multi_turn():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    result = _messages_to_bedrock_prompt(messages)
    assert "Human: Hi" in result
    assert "Assistant: Hello!" in result
    assert "Human: How are you?" in result
    assert result.endswith("Assistant: ")


def test_bedrock_prompt_always_ends_with_assistant_marker():
    messages = [{"role": "user", "content": "Query"}]
    result = _messages_to_bedrock_prompt(messages)
    assert result.endswith("Assistant: ")


# ---------------------------------------------------------------------------
# _openai_tools_to_google_tools
# ---------------------------------------------------------------------------


def test_openai_to_google_tools_structure():
    result = _openai_tools_to_google_tools(SAMPLE_TOOLS)
    assert isinstance(result, list)
    assert len(result) == 1  # All declarations wrapped in one object
    assert "function_declarations" in result[0]
    assert len(result[0]["function_declarations"]) == 2


def test_openai_to_google_tools_preserves_name():
    result = _openai_tools_to_google_tools(SAMPLE_TOOLS)
    names = [d["name"] for d in result[0]["function_declarations"]]
    assert "grep" in names
    assert "read" in names


def test_openai_to_google_tools_preserves_description():
    result = _openai_tools_to_google_tools(SAMPLE_TOOLS)
    descs = [d["description"] for d in result[0]["function_declarations"]]
    assert "Search code with ripgrep" in descs


def test_openai_to_google_tools_preserves_parameters():
    result = _openai_tools_to_google_tools(SAMPLE_TOOLS)
    grep_decl = next(
        d for d in result[0]["function_declarations"] if d["name"] == "grep"
    )
    assert grep_decl["parameters"]["type"] == "object"
    assert "pattern" in grep_decl["parameters"]["properties"]


def test_openai_to_google_tools_empty():
    result = _openai_tools_to_google_tools([])
    assert result == [{"function_declarations": []}]


# ---------------------------------------------------------------------------
# _messages_to_google_contents
# ---------------------------------------------------------------------------


def test_google_contents_user_message():
    messages = [{"role": "user", "content": "Hello"}]
    result = _messages_to_google_contents(messages)
    assert result == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_google_contents_assistant_becomes_model():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = _messages_to_google_contents(messages)
    assert result[1]["role"] == "model"
    assert result[1]["parts"][0]["text"] == "Hello!"


def test_google_contents_system_merged_into_first_user():
    messages = [
        {"role": "system", "content": "Be an expert."},
        {"role": "user", "content": "Explain this"},
    ]
    result = _messages_to_google_contents(messages)
    # System content should be merged into the first user message
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "Be an expert." in result[0]["parts"][0]["text"]
    assert "Explain this" in result[0]["parts"][0]["text"]


def test_google_contents_only_system_message():
    messages = [{"role": "system", "content": "You are helpful."}]
    result = _messages_to_google_contents(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "You are helpful." in result[0]["parts"][0]["text"]


def test_google_contents_tool_message_becomes_function_response():
    # role="tool" must be converted, not skipped.
    # Without a matching assistant tool_call_id the name falls back to tool_call_id.
    messages = [
        {"role": "user", "content": "Query"},
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "42",
        },
    ]
    result = _messages_to_google_contents(messages)
    assert len(result) == 2
    tool_turn = result[1]
    assert tool_turn["role"] == "user"
    fr = tool_turn["parts"][0]["function_response"]
    assert fr["response"]["result"] == "42"
    # name falls back to tool_call_id when no preceding assistant maps it
    assert fr["name"] == "call_abc"


def test_google_contents_assistant_with_tool_calls():
    # role="assistant" with tool_calls should emit function_call parts.
    messages = [
        {"role": "user", "content": "Search for main"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "grep", "arguments": '{"pattern": "main"}'},
                }
            ],
        },
    ]
    result = _messages_to_google_contents(messages)
    assert len(result) == 2
    model_turn = result[1]
    assert model_turn["role"] == "model"
    fc = model_turn["parts"][0]["function_call"]
    assert fc["name"] == "grep"
    assert fc["args"] == {"pattern": "main"}


def test_google_contents_multi_turn_tool_use():
    # Full agent-loop round-trip: user → assistant(tool_call) → tool → assistant(text)
    messages = [
        {"role": "user", "content": "Find main"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": '{"pattern": "main"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "src/main.py:10"},
        {"role": "assistant", "content": "Found it at src/main.py line 10."},
    ]
    result = _messages_to_google_contents(messages)
    assert len(result) == 4

    # user turn
    assert result[0]["role"] == "user"
    # model turn with function_call
    assert result[1]["role"] == "model"
    assert result[1]["parts"][0]["function_call"]["name"] == "grep"
    # user turn with function_response; name resolved via tool_call_id lookup
    assert result[2]["role"] == "user"
    fr = result[2]["parts"][0]["function_response"]
    assert fr["name"] == "grep"
    assert fr["response"]["result"] == "src/main.py:10"
    # final model text turn
    assert result[3]["role"] == "model"
    assert result[3]["parts"][0]["text"] == "Found it at src/main.py line 10."


def test_google_contents_tool_call_id_resolved_to_name():
    # When the preceding assistant message maps call_id → name, the
    # function_response must use the resolved name (not the raw call_id).
    messages = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c99",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c99", "content": "file contents"},
    ]
    result = _messages_to_google_contents(messages)
    fr = result[2]["parts"][0]["function_response"]
    assert fr["name"] == "read"  # resolved, not "c99"


# ---------------------------------------------------------------------------
# UnifiedProvider — constructor and metadata (no network calls)
# ---------------------------------------------------------------------------


def _make_mock_config(provider: str = "openai") -> dict:
    """Return a minimal get_model_config() return value."""
    return {
        "model_client": MagicMock,
        "model_kwargs": {
            "model": "test-model",
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }


def test_unified_provider_init_stores_provider_and_model():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("openai", "gpt-5-nano")
    assert p.provider == "openai"
    assert p.model == "gpt-5-nano"


def test_unified_provider_init_raises_on_unknown_provider():
    with patch("api.agent.provider.get_model_config", side_effect=ValueError("unknown")):
        with pytest.raises(ValueError, match="unknown"):
            UnifiedProvider("not_a_real_provider", "some-model")


def test_unified_provider_supports_native_tools_openai():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("openai", "gpt-5-nano")
    assert p._supports_native_tools() is True


def test_unified_provider_supports_native_tools_azure():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("azure", "gpt-4o")
    assert p._supports_native_tools() is True


def test_unified_provider_supports_native_tools_dashscope():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("dashscope", "qwen-plus")
    assert p._supports_native_tools() is True


def test_unified_provider_supports_native_tools_google():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("google", "gemini-2.5-flash")
    assert p._supports_native_tools() is True


def test_unified_provider_no_native_tools_openrouter():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("openrouter", "openai/gpt-5-nano")
    assert p._supports_native_tools() is False


def test_unified_provider_no_native_tools_ollama():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("ollama", "qwen3:1.7b")
    assert p._supports_native_tools() is False


def test_unified_provider_no_native_tools_bedrock():
    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("bedrock", "anthropic.claude-3-sonnet")
    assert p._supports_native_tools() is False


def test_unified_provider_get_generation_params():
    config = {
        "model_client": MagicMock,
        "model_kwargs": {
            "model": "test",
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 30,
        },
    }
    with patch("api.agent.provider.get_model_config", return_value=config):
        p = UnifiedProvider("google", "gemini-2.5-flash")
    params = p._get_generation_params()
    assert params == {"temperature": 0.5, "top_p": 0.8, "top_k": 30}


def test_unified_provider_get_generation_params_excludes_model():
    config = {
        "model_client": MagicMock,
        "model_kwargs": {"model": "test", "temperature": 0.7},
    }
    with patch("api.agent.provider.get_model_config", return_value=config):
        p = UnifiedProvider("openai", "test")
    params = p._get_generation_params()
    assert "model" not in params


# ---------------------------------------------------------------------------
# _stream_openai_compat — async client initialization regression (Fix 1)
# ---------------------------------------------------------------------------


class _AsyncEmptyIter:
    """Minimal async iterator that yields no chunks (simulates empty stream)."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def test_stream_openai_compat_assigns_async_client():
    """
    Regression test for Fix 1: init_async_client() return value must be
    assigned to self._client.async_client.

    Before the fix, provider.py called self._client.init_async_client() but
    discarded the return value, so async_client stayed None and the subsequent
    .chat.completions.create() call raised AttributeError.
    """
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_AsyncEmptyIter()
    )

    with patch("api.agent.provider.get_model_config", return_value=_make_mock_config()):
        p = UnifiedProvider("openai", "test")

    # Simulate a provider client whose async_client starts uninitialized
    p._client = MagicMock()
    p._client.async_client = None
    p._client.init_async_client.return_value = mock_async_client
    p._client_initialized = True

    async def drive():
        events = []
        async for event in p._stream_openai_compat(
            [{"role": "user", "content": "hi"}], None
        ):
            events.append(event)
        return events

    asyncio.run(drive())

    # After streaming, async_client must be the object returned by init_async_client()
    assert p._client.async_client is mock_async_client
    p._client.init_async_client.assert_called_once()
