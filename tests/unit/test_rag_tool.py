import asyncio

from adalflow.core.types import Document

from api.tools import get_tool
from api.tools import rag as rag_module
from api.tools.rag import RagTool


def _doc(text: str, file_path: str, start_line=10, end_line=12) -> Document:
    return Document(
        text=text,
        meta_data={
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
        },
        vector=[0.1, 0.2],
    )


def test_empty_query_returns_error(tmp_path):
    tool = RagTool(str(tmp_path))

    result = asyncio.run(tool.execute({"query": "   "}))

    assert result.metadata["error"] == "missing_query"
    assert "query is required" in result.output


def test_top_k_is_clamped(monkeypatch, tmp_path):
    calls = []

    class FakeRetriever:
        def retrieve(self, query, top_k):
            calls.append((query, top_k))
            return []

    async def fake_get_or_build_retriever(*args, **kwargs):
        return FakeRetriever()

    monkeypatch.setattr(rag_module, "_get_or_build_retriever", fake_get_or_build_retriever)
    tool = RagTool(str(tmp_path))

    asyncio.run(tool.execute({"query": "auth", "top_k": 99}))
    asyncio.run(tool.execute({"query": "auth", "top_k": 0}))

    assert calls == [("auth", 20), ("auth", 1)]


def test_output_format_includes_location_and_snippet(monkeypatch, tmp_path):
    docs = [_doc("def clone_repo():\n    pass", "api/data_pipeline.py", 20, 24)]

    class FakeRetriever:
        def retrieve(self, query, top_k):
            return docs

    async def fake_get_or_build_retriever(*args, **kwargs):
        return FakeRetriever()

    monkeypatch.setattr(rag_module, "_get_or_build_retriever", fake_get_or_build_retriever)
    tool = RagTool(str(tmp_path))

    result = asyncio.run(tool.execute({"query": "repository cloning"}))

    assert result.title == "rag_search: repository cloning"
    assert "### 1. api/data_pipeline.py:20-24" in result.output
    assert "def clone_repo()" in result.output
    assert result.metadata["matches"] == 1


def test_retriever_exception_returns_error(monkeypatch, tmp_path):
    async def fake_get_or_build_retriever(*args, **kwargs):
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(rag_module, "_get_or_build_retriever", fake_get_or_build_retriever)
    tool = RagTool(str(tmp_path))

    result = asyncio.run(tool.execute({"query": "auth"}))

    assert result.metadata["error"] == "retrieval_failed"
    assert "cache unavailable" in result.output


def test_tool_registers_via_registry(tmp_path):
    tool = get_tool("rag_search", str(tmp_path))

    assert tool.name == "rag_search"
    assert tool.to_function_schema()["function"]["name"] == "rag_search"
