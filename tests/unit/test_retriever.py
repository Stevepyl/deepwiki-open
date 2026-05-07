import asyncio
import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from adalflow.core.types import Document

from api import data_pipeline
from api import retriever as retriever_module
from api.data_pipeline import DatabaseManager
from api.retriever import CodeRetriever, get_or_build_retriever


def _doc(
    text: str,
    file_path: str,
    vector: list[float],
    *,
    doc_type: str = "py",
) -> Document:
    return Document(
        text=text,
        vector=vector,
        meta_data={
            "file_path": file_path,
            "type": doc_type,
            "start_line": 1,
            "end_line": 3,
            "ast_chunk_index": 0,
            "ast_chunk_count": 1,
        },
    )


@pytest.fixture(autouse=True)
def _clear_retriever_cache():
    retriever_module._CACHE.clear()
    retriever_module._LOCKS.clear()
    yield
    retriever_module._CACHE.clear()
    retriever_module._LOCKS.clear()


def test_prepare_delegates_to_database_manager_and_filters_embeddings(monkeypatch):
    docs = [
        _doc("first", "src/a.py", [0.1, 0.2, 0.3]),
        _doc("second", "src/b.py", [0.4, 0.5, 0.6]),
        _doc("wrong", "src/c.py", [0.7, 0.8]),
        _doc("empty", "src/d.py", []),
    ]

    class FakeDatabaseManager:
        def __init__(self):
            self.calls = []

        def prepare_database(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return docs

    class FakeFAISSRetriever:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

    manager = FakeDatabaseManager()
    monkeypatch.setattr(retriever_module, "_new_database_manager", lambda: manager)
    monkeypatch.setattr(retriever_module, "FAISSRetriever", FakeFAISSRetriever)
    monkeypatch.setattr(retriever_module, "_new_embedder", lambda embedder_type: object())

    retriever = CodeRetriever(embedder_type="openai")
    retriever.prepare(str("/tmp/repo//"), repo_type="local")

    assert manager.calls
    assert manager.calls[0][1]["embedder_type"] == "openai"
    assert [doc.meta_data["file_path"] for doc in retriever.transformed_docs] == [
        "src/a.py",
        "src/b.py",
    ]
    assert FakeFAISSRetriever.instances[0].kwargs["documents"] == retriever.transformed_docs


def test_retrieve_maps_faiss_indices_to_documents():
    documents = [
        _doc("first", "src/a.py", [0.1]),
        _doc("second", "src/b.py", [0.2]),
    ]

    class FakeFAISSRetriever:
        def __call__(self, query, top_k=None):
            assert query == "where is setup?"
            assert top_k == 2
            return [SimpleNamespace(doc_indices=[1, 0, 99])]

    retriever = object.__new__(CodeRetriever)
    retriever.transformed_docs = documents
    retriever.retriever = FakeFAISSRetriever()

    results = retriever.retrieve("where is setup?", top_k=2)

    assert results == [documents[1], documents[0]]


def test_database_manager_reuses_existing_pickle_without_reembedding(monkeypatch, tmp_path):
    cached_docs = [_doc("cached", "src/cached.py", [0.1, 0.2, 0.3])]
    db_file = tmp_path / "repo.pkl"
    db_file.write_bytes(b"cache")

    class FakeLocalDB:
        def get_transformed_data(self, key):
            assert key == "split_and_embed"
            return cached_docs

    monkeypatch.setattr(
        data_pipeline.LocalDB,
        "load_state",
        staticmethod(lambda path: FakeLocalDB()),
    )
    transform = Mock()
    monkeypatch.setattr(data_pipeline, "transform_documents_and_save_to_db", transform)

    manager = DatabaseManager()
    manager.repo_paths = {
        "save_repo_dir": str(tmp_path),
        "save_db_file": str(db_file),
    }

    assert manager.prepare_db_index(embedder_type="openai") == cached_docs
    transform.assert_not_called()


def test_database_manager_reuses_legacy_python_pickle_without_reembedding(monkeypatch, tmp_path):
    cached_docs = [
        Document(
            text="cached",
            vector=[0.1, 0.2, 0.3],
            meta_data={"file_path": "src/legacy.py", "type": "py"},
        )
    ]
    db_file = tmp_path / "repo.pkl"
    db_file.write_bytes(b"cache")

    class FakeLocalDB:
        def get_transformed_data(self, key):
            assert key == "split_and_embed"
            return cached_docs

    monkeypatch.setattr(
        data_pipeline.LocalDB,
        "load_state",
        staticmethod(lambda path: FakeLocalDB()),
    )
    transform = Mock()
    monkeypatch.setattr(data_pipeline, "transform_documents_and_save_to_db", transform)

    manager = DatabaseManager()
    manager.repo_paths = {
        "save_repo_dir": str(tmp_path),
        "save_db_file": str(db_file),
    }

    assert manager.prepare_db_index(embedder_type="openai") == cached_docs
    transform.assert_not_called()


def test_transform_documents_suppresses_text_splitter_chunk_logs(monkeypatch, tmp_path, caplog):
    text_splitter_logger = logging.getLogger(data_pipeline.TEXT_SPLITTER_LOGGER_NAME)
    previous_level = text_splitter_logger.level

    class FakeTransformer:
        def __call__(self, documents):
            text_splitter_logger.info("Text merged into 2 chunks.")
            return documents

    monkeypatch.setattr(
        data_pipeline,
        "prepare_data_pipeline",
        lambda *args, **kwargs: FakeTransformer(),
    )
    monkeypatch.setattr(data_pipeline.LocalDB, "save_state", lambda self, filepath: None)

    try:
        text_splitter_logger.setLevel(logging.INFO)
        caplog.set_level(logging.INFO, logger=data_pipeline.TEXT_SPLITTER_LOGGER_NAME)

        data_pipeline.transform_documents_and_save_to_db(
            [_doc("cached", "src/cached.py", [0.1, 0.2, 0.3])],
            str(tmp_path / "repo.pkl"),
            embedder_type="openai",
        )

        assert "Text merged into 2 chunks." not in caplog.messages
        assert text_splitter_logger.level == logging.INFO
    finally:
        text_splitter_logger.setLevel(previous_level)


def test_get_or_build_retriever_returns_cached_instance(monkeypatch):
    class FakeCodeRetriever:
        prepare_calls = 0

        def __init__(self, embedder_type=None):
            self.embedder_type = embedder_type

        def prepare(self, *args, **kwargs):
            type(self).prepare_calls += 1

    monkeypatch.setattr(retriever_module, "get_embedder_type", lambda: "openai")
    monkeypatch.setattr(retriever_module, "CodeRetriever", FakeCodeRetriever)

    async def scenario():
        first = await get_or_build_retriever("/tmp/repo")
        second = await get_or_build_retriever("/tmp/repo")
        return first, second

    first, second = asyncio.run(scenario())

    assert first is second
    assert FakeCodeRetriever.prepare_calls == 1


def test_get_or_build_retriever_separates_embedder_types(monkeypatch):
    class FakeCodeRetriever:
        def __init__(self, embedder_type=None):
            self.embedder_type = embedder_type

        def prepare(self, *args, **kwargs):
            return None

    current_embedder = {"type": "openai"}
    monkeypatch.setattr(retriever_module, "get_embedder_type", lambda: current_embedder["type"])
    monkeypatch.setattr(retriever_module, "CodeRetriever", FakeCodeRetriever)

    async def scenario():
        first = await get_or_build_retriever("/tmp/repo")
        current_embedder["type"] = "google"
        second = await get_or_build_retriever("/tmp/repo")
        return first, second

    first, second = asyncio.run(scenario())

    assert first is not second
    assert {first.embedder_type, second.embedder_type} == {"openai", "google"}


def test_get_or_build_retriever_evicts_oldest_entry(monkeypatch):
    class FakeCodeRetriever:
        def __init__(self, embedder_type=None):
            self.embedder_type = embedder_type

        def prepare(self, *args, **kwargs):
            return None

    monkeypatch.setattr(retriever_module, "_CACHE_MAX", 4)
    monkeypatch.setattr(retriever_module, "get_embedder_type", lambda: "openai")
    monkeypatch.setattr(retriever_module, "CodeRetriever", FakeCodeRetriever)

    async def scenario():
        for index in range(5):
            await get_or_build_retriever(f"/tmp/repo-{index}")

    asyncio.run(scenario())

    assert (os.path.normpath("/tmp/repo-0"), "openai") not in retriever_module._CACHE
    assert len(retriever_module._CACHE) == 4


def test_get_or_build_retriever_serializes_concurrent_builds(monkeypatch):
    class FakeCodeRetriever:
        prepare_calls = 0

        def __init__(self, embedder_type=None):
            self.embedder_type = embedder_type

        def prepare(self, *args, **kwargs):
            time.sleep(0.05)
            type(self).prepare_calls += 1

    monkeypatch.setattr(retriever_module, "get_embedder_type", lambda: "openai")
    monkeypatch.setattr(retriever_module, "CodeRetriever", FakeCodeRetriever)

    async def scenario():
        return await asyncio.gather(
            get_or_build_retriever("/tmp/repo"),
            get_or_build_retriever("/tmp/repo"),
            get_or_build_retriever("/tmp/repo"),
        )

    results = asyncio.run(scenario())

    assert results[0] is results[1] is results[2]
    assert FakeCodeRetriever.prepare_calls == 1
