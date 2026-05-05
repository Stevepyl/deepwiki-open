import json

from fastapi.testclient import TestClient

from api.agent import chat_handler
from api.agent.stream_events import FinishEvent, TextDelta, ToolCallEnd, ToolCallStart
from api.api import app


client = TestClient(app)


def _payload(**overrides):
    payload = {
        "repo_url": "https://github.com/owner/repo",
        "type": "github",
        "messages": [{"role": "user", "content": "Where is auth?"}],
        "provider": "google",
        "agent_name": "explore",
    }
    payload.update(overrides)
    return payload


def _patch_happy_path(monkeypatch):
    monkeypatch.setattr(chat_handler, "download_repo", lambda *args: None)
    monkeypatch.setattr(chat_handler, "UnifiedProvider", lambda *args: object())

    async def fake_run_agent_loop(*args, **kwargs):
        yield TextDelta(content="I will search.")
        yield ToolCallStart(
            tool_call_id="call_1",
            tool_name="grep",
            tool_args={"pattern": "auth", "path": "."},
        )
        yield ToolCallEnd(
            tool_call_id="call_1",
            tool_name="grep",
            result_summary="src/auth.ts:1",
        )
        yield FinishEvent(finish_reason="stop", usage={"prompt_tokens": 3, "completion_tokens": 2})

    monkeypatch.setattr(chat_handler, "run_agent_loop", fake_run_agent_loop)


def _post_events(payload):
    response = client.post("/chat/agent-stream", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def test_agent_chat_stream_happy_path(monkeypatch):
    _patch_happy_path(monkeypatch)

    events = _post_events(_payload())

    assert [event["type"] for event in events] == [
        "text_delta",
        "tool_call_start",
        "tool_call_end",
        "finish",
    ]
    assert events[1]["tool_call_id"] == events[2]["tool_call_id"]
    assert events[-1]["finish_reason"] == "stop"


def test_agent_chat_stream_unknown_agent():
    events = _post_events(_payload(agent_name="bogus"))

    assert [event["type"] for event in events] == ["error", "finish"]
    assert events[0]["code"] == "unknown_agent"
    assert events[-1]["finish_reason"] == "error"


def test_agent_chat_stream_clone_failure(monkeypatch):
    def fail_download(*args):
        raise RuntimeError("clone failed")

    monkeypatch.setattr(chat_handler, "download_repo", fail_download)

    events = _post_events(_payload())

    assert [event["type"] for event in events] == ["error", "finish"]
    assert events[0]["code"] == "clone_failed"
    assert events[-1]["finish_reason"] == "error"


def test_agent_chat_stream_provider_init_failure(monkeypatch):
    monkeypatch.setattr(chat_handler, "download_repo", lambda *args: None)

    def fail_provider(*args):
        raise RuntimeError("no provider")

    monkeypatch.setattr(chat_handler, "UnifiedProvider", fail_provider)

    events = _post_events(_payload())

    assert [event["type"] for event in events] == ["error", "finish"]
    assert events[0]["code"] == "provider_error"
    assert events[-1]["finish_reason"] == "error"


def test_agent_chat_stream_empty_messages():
    events = _post_events(_payload(messages=[]))

    assert [event["type"] for event in events] == ["error", "finish"]
    assert events[0]["code"] == "empty_messages"
    assert events[-1]["finish_reason"] == "error"


def test_agent_chat_stream_last_message_not_user():
    events = _post_events(_payload(messages=[{"role": "assistant", "content": "hello"}]))

    assert [event["type"] for event in events] == ["error", "finish"]
    assert events[0]["code"] == "last_message_not_user"
    assert events[-1]["finish_reason"] == "error"


def test_agent_info_filters_to_chat_agents():
    response = client.get("/agent/info")

    assert response.status_code == 200
    names = {agent["name"] for agent in response.json()}
    assert names == {"wiki", "explore", "deep-research"}
