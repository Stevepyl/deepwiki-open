import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from api.chat_history import format_conversation_history


class FakeRAG:
    instances = []

    def __init__(self, provider="google", model=None):
        self.provider = provider
        self.model = model
        self.prepare_calls = []
        self.transformed_docs = []
        FakeRAG.instances.append(self)

    def prepare_retriever(
        self,
        repo_url,
        repo_type="github",
        access_token=None,
        excluded_dirs=None,
        excluded_files=None,
        included_dirs=None,
        included_files=None,
    ):
        self.prepare_calls.append(
            {
                "repo_url": repo_url,
                "repo_type": repo_type,
                "access_token": access_token,
                "excluded_dirs": excluded_dirs,
                "excluded_files": excluded_files,
                "included_dirs": included_dirs,
                "included_files": included_files,
            }
        )
        self.transformed_docs = [object(), object()]


@pytest.fixture
def rag_cache_module(monkeypatch):
    fake_config = types.ModuleType("api.config")
    fake_config.get_embedder_type = lambda: "dashscope"
    fake_rag = types.ModuleType("api.rag")
    fake_rag.RAG = object

    monkeypatch.setitem(sys.modules, "api.config", fake_config)
    monkeypatch.setitem(sys.modules, "api.rag", fake_rag)
    sys.modules.pop("api.rag_cache", None)

    module = importlib.import_module("api.rag_cache")
    module.clear_rag_cache()
    yield module
    module.clear_rag_cache()
    sys.modules.pop("api.rag_cache", None)


def _prepare(rag_cache, monkeypatch, **kwargs):
    monkeypatch.setattr(rag_cache, "RAG", FakeRAG)
    monkeypatch.setattr(rag_cache, "get_embedder_type", lambda: "dashscope")
    defaults = {
        "repo_url": "/repo",
        "repo_type": "local",
        "token": None,
        "provider": "dashscope",
        "model": "qwen-plus",
        "excluded_dirs": None,
        "excluded_files": None,
        "included_dirs": None,
        "included_files": None,
    }
    defaults.update(kwargs)
    return rag_cache.get_prepared_rag(**defaults)


def setup_function():
    FakeRAG.instances = []


def test_get_prepared_rag_reuses_same_cache_key(rag_cache_module, monkeypatch):
    first = _prepare(rag_cache_module, monkeypatch)
    second = _prepare(rag_cache_module, monkeypatch)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.rag is second.rag
    assert len(FakeRAG.instances) == 1
    assert len(FakeRAG.instances[0].prepare_calls) == 1


def test_get_prepared_rag_separates_model_cache_keys(rag_cache_module, monkeypatch):
    first = _prepare(rag_cache_module, monkeypatch, model="qwen-plus")
    second = _prepare(rag_cache_module, monkeypatch, model="qwen-turbo")

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first.rag is not second.rag
    assert len(FakeRAG.instances) == 2


def test_get_prepared_rag_rebuilds_after_ttl(rag_cache_module, monkeypatch):
    clock = {"value": 100.0}
    monkeypatch.setenv("DEEPWIKI_RAG_CACHE_TTL_SECONDS", "1")
    monkeypatch.setattr(rag_cache_module.time, "monotonic", lambda: clock["value"])

    first = _prepare(rag_cache_module, monkeypatch)
    clock["value"] = 102.0
    second = _prepare(rag_cache_module, monkeypatch)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first.rag is not second.rag
    assert len(FakeRAG.instances) == 2


def test_format_conversation_history_uses_only_request_messages():
    messages = [
        SimpleNamespace(role="user", content="first question"),
        SimpleNamespace(role="assistant", content="first answer"),
        SimpleNamespace(role="user", content="current question"),
    ]

    history = format_conversation_history(messages)

    assert "first question" in history
    assert "first answer" in history
    assert "current question" not in history
