import asyncio

from api.agent import chat_handler
from api.agent.filtered_tools import FilteredToolWrapper
from api.agent.stream_events import FinishEvent, ToolCallStart
from api.tools.tool import ToolResult
from api.utils.filters import ParsedFilters


def _request(**overrides):
    payload = {
        "repo_url": "https://github.com/owner/repo",
        "type": "github",
        "messages": [{"role": "user", "content": "Where is cloning handled?"}],
        "provider": "google",
        "agent_name": "explore",
    }
    payload.update(overrides)
    return chat_handler.AgentChatRequest.model_validate(payload)


def test_agent_chat_passes_rag_search_tool_to_loop(monkeypatch):
    captured = {}
    events = []

    monkeypatch.setattr(chat_handler, "download_repo", lambda *args: None)
    monkeypatch.setattr(chat_handler, "UnifiedProvider", lambda *args: object())

    async def fake_run_agent_loop(agent_config, messages, provider, tools, repo_path, system_prompt_vars):
        captured["tools"] = tools
        yield ToolCallStart(
            tool_call_id="call_1",
            tool_name="rag_search",
            tool_args={"query": "repository cloning"},
        )
        yield FinishEvent(finish_reason="stop")

    async def on_event(event):
        events.append(event)

    monkeypatch.setattr(chat_handler, "run_agent_loop", fake_run_agent_loop)

    asyncio.run(
        chat_handler._run_agent_chat(
            _request(excluded_dirs="secrets"),
            on_event,
        )
    )

    assert "rag_search" in captured["tools"]
    assert isinstance(captured["tools"]["rag_search"], FilteredToolWrapper)
    assert captured["tools"]["rag_search"].name == "rag_search"
    assert any(getattr(event, "tool_name", None) == "rag_search" for event in events)


def test_agent_info_lists_rag_search_for_chat_agents():
    info = chat_handler.get_agent_info()

    assert info
    for agent in info:
        assert "rag_search" in agent["allowed_tools"]


def test_rag_search_filter_drops_excluded_chunks():
    class FakeRagTool:
        name = "rag_search"
        description = "fake"
        parameters_schema = {}

        def to_function_schema(self):
            return {}

        async def execute(self, params):
            return ToolResult(
                title="rag_search: auth",
                output=(
                    "### 1. src/auth.py:1-4\n"
                    "def auth(): pass\n\n"
                    "### 2. secrets/key.py:1-2\n"
                    "API_KEY = 'secret'"
                ),
                metadata={"matches": 2},
            )

    wrapper = FilteredToolWrapper(
        FakeRagTool(),
        ParsedFilters.from_strings(excluded_dirs="secrets"),
        "/repo",
    )

    result = asyncio.run(wrapper.execute({"query": "auth"}))

    assert "src/auth.py" in result.output
    assert "secrets/key.py" not in result.output
    assert "1 chunk hidden by filter" in result.output
    assert result.metadata["matches"] == 1
    assert result.metadata["filtered_count"] == 1
