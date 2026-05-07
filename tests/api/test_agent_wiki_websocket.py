import json

from fastapi.testclient import TestClient

from api.agent import wiki_generator
from api.agent.stream_events import FinishEvent, TextDelta
from api.api import app


client = TestClient(app)


def _payload(**overrides):
    payload = {
        "repo_url": "https://github.com/owner/repo",
        "type": "github",
        "provider": "google",
        "language": "en",
        "comprehensive": False,
    }
    payload.update(overrides)
    return payload


def _structure():
    return {
        "id": "wiki-root",
        "title": "Repo Wiki",
        "description": "Generated repo wiki.",
        "pages": [
            {
                "id": "overview",
                "title": "Overview",
                "content": "",
                "filePaths": ["README.md"],
                "importance": "high",
                "relatedPages": [],
            }
        ],
        "sections": None,
        "rootSections": None,
    }


def _collect_until_finish(payload):
    with client.websocket_connect("/ws/agent-wiki") as ws:
        ws.send_json(payload)
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "finish":
                return events


def test_agent_wiki_builds_embeddings_before_planning_and_sends_one_terminal_finish(monkeypatch):
    calls = []

    def fake_download(*args):
        calls.append("clone")

    async def fake_get_or_build_retriever(*args, **kwargs):
        calls.append("embedding")
        assert kwargs["repo_type"] == "local"

    async def fake_consume_agent_loop(
        agent_config,
        messages,
        provider,
        tools,
        repo_path,
        system_prompt_vars,
        on_event,
    ):
        calls.append(agent_config.name)
        assert "rag_search" in tools

        if agent_config.name == "wiki-planner":
            await on_event(TextDelta(content=json.dumps(_structure())))
            await on_event(FinishEvent(finish_reason="stop"))
            return

        if agent_config.name == "wiki-writer":
            await on_event(TextDelta(content="# Overview\n\nGenerated page."))
            await on_event(FinishEvent(finish_reason="stop"))
            return

        raise AssertionError(f"unexpected agent: {agent_config.name}")

    monkeypatch.setattr(wiki_generator, "download_repo", fake_download)
    monkeypatch.setattr(wiki_generator, "_get_or_build_retriever", fake_get_or_build_retriever)
    monkeypatch.setattr(wiki_generator, "build_file_tree", lambda *args, **kwargs: "README.md")
    monkeypatch.setattr(wiki_generator, "read_repo_readme", lambda *args, **kwargs: "# Repo")
    monkeypatch.setattr(wiki_generator, "UnifiedProvider", lambda *args: object())
    monkeypatch.setattr(wiki_generator, "consume_agent_loop", fake_consume_agent_loop)

    events = _collect_until_finish(_payload())

    assert calls.index("embedding") < calls.index("wiki-planner")
    assert [event["type"] for event in events] == [
        "text_delta",
        "wiki_structure_ready",
        "text_delta",
        "wiki_page_done",
        "finish",
    ]
    assert events[0]["phase"] == "planning"
    assert events[2]["phase"] == "writing"
    assert events[-1].get("phase") is None
