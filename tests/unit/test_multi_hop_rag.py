from types import SimpleNamespace

from adalflow.core.types import Document

from api.data_pipeline import read_all_documents
from api import rag as rag_module
from api.rag import RAG, build_citations_from_retrieved_documents


def _make_code_doc(
    *,
    text: str,
    file_path: str,
    symbol_full_name: str | None = None,
    symbol_name: str | None = None,
    parent_symbol: str | None = None,
    ast_chunk_index: int | None = None,
    ast_chunk_count: int | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    doc_type: str = "py",
) -> Document:
    meta_data = {
        "file_path": file_path,
        "type": doc_type,
        "is_code": True,
        "symbol_full_name": symbol_full_name,
        "symbol_name": symbol_name,
        "parent_symbol": parent_symbol,
    }
    if ast_chunk_index is not None:
        meta_data["ast_chunk_index"] = ast_chunk_index
    if ast_chunk_count is not None:
        meta_data["ast_chunk_count"] = ast_chunk_count
    if start_line is not None:
        meta_data["start_line"] = start_line
    if end_line is not None:
        meta_data["end_line"] = end_line
    return Document(text=text, meta_data=meta_data)


def _build_test_rag(documents, retriever):
    rag = object.__new__(RAG)
    rag.transformed_docs = documents
    rag.retriever = retriever
    rag.repo_id = ""
    rag.knowledge_store = None
    rag.doc_index_map = {}
    rag.docs_by_file = {}
    rag.docs_by_symbol_full_name = {}
    rag.docs_by_file_symbol_name = {}
    rag.docs_by_file_parent_symbol = {}
    rag.doc_position_in_file = {}
    rag._build_document_indices()
    return rag


class FakeKnowledgeStore:
    def __init__(self, refs):
        self.refs = refs

    def get_reference_neighbors(self, repo_id, target, limit=10):
        return self.refs.get(target, [])[:limit]


def test_read_all_documents_adds_python_ast_metadata(tmp_path):
    python_file = tmp_path / "sample.py"
    python_file.write_text(
        "\n".join(
            [
                "class Foo:",
                "    def bar(self):",
                "        return 1",
                "",
                "def baz():",
                "    return Foo().bar()",
            ]
        ),
        encoding="utf-8",
    )
    markdown_file = tmp_path / "README.md"
    markdown_file.write_text("# Hello\n", encoding="utf-8")

    documents = read_all_documents(str(tmp_path), embedder_type="google")

    python_docs = [doc for doc in documents if doc.meta_data.get("type") == "py"]
    assert python_docs
    for doc in python_docs:
        for field_name in ("start_line", "end_line", "ast_chunk_index", "ast_chunk_count"):
            assert field_name in doc.meta_data
            assert doc.meta_data[field_name] is not None

    non_python_docs = [doc for doc in documents if doc.meta_data.get("type") != "py"]
    assert non_python_docs
    assert "start_line" not in non_python_docs[0].meta_data


def test_expand_hop2_keeps_max_anchor_score_for_shared_candidates(monkeypatch):
    documents = [
        _make_code_doc(
            text="def alpha(): pass",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.Handler.alpha",
            symbol_name="alpha",
            parent_symbol="Handler",
            ast_chunk_index=1,
            ast_chunk_count=4,
            start_line=20,
            end_line=24,
        ),
        _make_code_doc(
            text="def alpha(): pass",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.Handler.alpha",
            symbol_name="alpha",
            parent_symbol="Handler",
            ast_chunk_index=0,
            ast_chunk_count=4,
            start_line=10,
            end_line=14,
        ),
        _make_code_doc(
            text="def gamma(): pass",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.Handler.gamma",
            symbol_name="gamma",
            parent_symbol="Handler",
            ast_chunk_index=2,
            ast_chunk_count=4,
            start_line=30,
            end_line=34,
        ),
        _make_code_doc(
            text="def gamma(): pass",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.Handler.gamma",
            symbol_name="gamma",
            parent_symbol="Handler",
            ast_chunk_index=3,
            ast_chunk_count=4,
            start_line=40,
            end_line=44,
        ),
    ]
    rag = _build_test_rag(documents, retriever=lambda _: [])
    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.7,
            "multi_hop": {
                "enabled": True,
                "seed_k": 2,
                "hop2_max_per_seed": 3,
                "neighbor_window": 1,
                "final_top_k": 10,
                "final_semantic_weight": 0.55,
                "final_anchor_weight": 0.30,
                "final_seed_weight": 0.15,
                "anchor_weights": {
                    "symbol_full_name": 1.0,
                    "symbol_name": 0.85,
                    "parent_symbol": 0.65,
                    "same_file_neighbor": 0.5,
                },
            },
        },
    )

    anchor_scores = rag._expand_hop2([0, 3], rag._get_multi_hop_config())

    assert anchor_scores[1] == 1.0
    assert anchor_scores[2] == 1.0


def test_call_multi_hop_merges_and_reranks_expanded_docs(monkeypatch):
    documents = [
        _make_code_doc(
            text="def alpha(): return helper()",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.alpha",
            symbol_name="alpha",
            parent_symbol=None,
            ast_chunk_index=0,
            ast_chunk_count=3,
            start_line=10,
            end_line=14,
        ),
        _make_code_doc(
            text="def alpha(): return helper()",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.alpha",
            symbol_name="alpha",
            parent_symbol=None,
            ast_chunk_index=1,
            ast_chunk_count=3,
            start_line=20,
            end_line=24,
        ),
        _make_code_doc(
            text="def helper(): return 1",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.helper",
            symbol_name="helper",
            parent_symbol=None,
            ast_chunk_index=2,
            ast_chunk_count=3,
            start_line=30,
            end_line=34,
        ),
    ]
    retriever_result = SimpleNamespace(doc_indices=[0], documents=[])
    rag = _build_test_rag(documents, retriever=lambda _: [retriever_result])

    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.7,
            "hybrid": {
                "enabled": False,
            },
            "multi_hop": {
                "enabled": True,
                "seed_k": 1,
                "hop2_max_per_seed": 3,
                "neighbor_window": 1,
                "final_top_k": 10,
                "final_semantic_weight": 0.55,
                "final_anchor_weight": 0.30,
                "final_seed_weight": 0.15,
                "anchor_weights": {
                    "symbol_full_name": 1.0,
                    "symbol_name": 0.85,
                    "parent_symbol": 0.65,
                    "same_file_neighbor": 0.5,
                },
            },
        },
    )

    retrieved_documents = rag.call("Where is alpha defined?")

    assert retrieved_documents[0].doc_indices == [0, 1]
    assert [doc.meta_data["file_path"] for doc in retrieved_documents[0].documents] == [
        "pkg/foo.py",
        "pkg/foo.py",
    ]


def test_call_graph_expand_adds_reverse_reference_consumer(monkeypatch):
    documents = [
        _make_code_doc(
            text="def changed(): return 1",
            file_path="pkg/settings.py",
            symbol_full_name="pkg.settings.changed",
            symbol_name="changed",
            ast_chunk_index=0,
            ast_chunk_count=1,
            start_line=1,
            end_line=2,
        ),
        _make_code_doc(
            text="def start_task(): return settings.changed()",
            file_path="pkg/worker.py",
            symbol_full_name="pkg.worker.start_task",
            symbol_name="start_task",
            ast_chunk_index=0,
            ast_chunk_count=1,
            start_line=10,
            end_line=12,
        ),
    ]
    retriever_result = SimpleNamespace(doc_indices=[0], documents=[])
    rag = _build_test_rag(documents, retriever=lambda _: [retriever_result])
    rag.repo_id = "fixture"
    rag.knowledge_store = FakeKnowledgeStore(
        {
            "pkg.settings.changed": [
                {
                    "source_scope": "pkg.worker.start_task",
                    "ref_type": "call",
                    "target": "pkg.settings.changed",
                    "file_path": "pkg/worker.py",
                    "line": 11,
                    "snippet": "settings.changed()",
                }
            ]
        }
    )

    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.7,
            "hybrid": {
                "enabled": False,
            },
            "multi_hop": {
                "enabled": True,
                "seed_k": 1,
                "hop2_max_per_seed": 3,
                "neighbor_window": 1,
                "final_top_k": 10,
                "final_semantic_weight": 0.55,
                "final_anchor_weight": 0.30,
                "final_seed_weight": 0.15,
                "anchor_weights": {
                    "symbol_full_name": 1.0,
                    "symbol_name": 0.85,
                    "parent_symbol": 0.65,
                    "same_file_neighbor": 0.5,
                },
            },
            "graph": {
                "enabled": True,
                "top_seed_k": 1,
                "max_refs_per_seed": 5,
                "max_candidates_per_seed": 2,
                "anchor_weight": 0.9,
            },
        },
    )

    retrieved_documents = rag.call("Where is changed used?")

    assert retrieved_documents[0].doc_indices == [0, 1]
    assert retrieved_documents[0].documents[1].meta_data["symbol_full_name"] == "pkg.worker.start_task"


def test_call_without_multi_hop_keeps_hop1_order(monkeypatch):
    documents = [
        _make_code_doc(
            text="def alpha(): pass",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.alpha",
            symbol_name="alpha",
            ast_chunk_index=0,
            ast_chunk_count=2,
            start_line=10,
            end_line=14,
        ),
        _make_code_doc(
            text="def alpha(): pass",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.alpha",
            symbol_name="alpha",
            ast_chunk_index=1,
            ast_chunk_count=2,
            start_line=20,
            end_line=24,
        ),
    ]
    retriever_result = SimpleNamespace(doc_indices=[0], documents=[])
    rag = _build_test_rag(documents, retriever=lambda _: [retriever_result])

    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.7,
            "hybrid": {
                "enabled": False,
            },
            "multi_hop": {
                "enabled": False,
            },
        },
    )

    retrieved_documents = rag.call("Where is alpha defined?")

    assert retrieved_documents[0].doc_indices == [0]
    assert len(retrieved_documents[0].documents) == 1


def test_call_hybrid_exact_retrieves_symbol_outside_faiss(monkeypatch):
    documents = [
        _make_code_doc(
            text="def unrelated(): pass",
            file_path="pkg/other.py",
            symbol_full_name="pkg.other.unrelated",
            symbol_name="unrelated",
            ast_chunk_index=0,
            ast_chunk_count=1,
            start_line=1,
            end_line=2,
        ),
        _make_code_doc(
            text="def parse_python(path): return path",
            file_path="app/ingest.py",
            symbol_full_name="app.ingest.parse_python",
            symbol_name="parse_python",
            ast_chunk_index=2,
            ast_chunk_count=4,
            start_line=42,
            end_line=55,
        ),
    ]
    retriever_result = SimpleNamespace(doc_indices=[0], documents=[])
    rag = _build_test_rag(documents, retriever=lambda _: [retriever_result])

    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.7,
            "hybrid": {
                "enabled": True,
                "exact_enabled": True,
                "sparse_enabled": False,
                "exact_top_k": 5,
                "rrf_k": 60,
                "max_candidates": 10,
                "max_query_tokens": 8,
            },
            "multi_hop": {
                "enabled": False,
            },
        },
    )

    retrieved_documents = rag.call("`parse_python` 在哪里定义？")

    assert retrieved_documents[0].doc_indices[0] == 1
    assert retrieved_documents[0].documents[0].meta_data["symbol_name"] == "parse_python"


def test_call_hybrid_sparse_retrieves_keyword_match_outside_faiss(monkeypatch):
    documents = [
        _make_code_doc(
            text="def unrelated(): return None",
            file_path="pkg/other.py",
            symbol_full_name="pkg.other.unrelated",
            symbol_name="unrelated",
            ast_chunk_index=0,
            ast_chunk_count=1,
            start_line=1,
            end_line=2,
        ),
        _make_code_doc(
            text="def validate_token(token): return token.startswith('ghp_')",
            file_path="api/auth.py",
            symbol_full_name="api.auth.validate_token",
            symbol_name="validate_token",
            ast_chunk_index=0,
            ast_chunk_count=1,
            start_line=10,
            end_line=12,
        ),
    ]
    retriever_result = SimpleNamespace(doc_indices=[0], documents=[])
    rag = _build_test_rag(documents, retriever=lambda _: [retriever_result])

    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.5,
            "hybrid": {
                "enabled": True,
                "exact_enabled": False,
                "sparse_enabled": True,
                "sparse_top_k": 5,
                "rrf_k": 60,
                "max_candidates": 10,
                "max_query_tokens": 8,
            },
            "multi_hop": {
                "enabled": False,
            },
        },
    )

    retrieved_documents = rag.call("validate token 鉴权逻辑在哪里？")

    assert 1 in retrieved_documents[0].doc_indices
    assert retrieved_documents[0].documents[0].meta_data["file_path"] == "api/auth.py"


def test_call_with_missing_metadata_keeps_hop1_results(monkeypatch):
    documents = [
        Document(
            text="def alpha(): pass",
            meta_data={
                "file_path": "pkg/foo.py",
                "type": "py",
                "is_code": True,
            },
        ),
        Document(
            text="def helper(): pass",
            meta_data={
                "file_path": "pkg/foo.py",
                "type": "py",
                "is_code": True,
            },
        ),
    ]
    retriever_result = SimpleNamespace(doc_indices=[0], documents=[])
    rag = _build_test_rag(documents, retriever=lambda _: [retriever_result])

    monkeypatch.setitem(
        rag_module.configs,
        "retriever",
        {
            "top_k": 20,
            "symbol_alpha": 0.7,
            "hybrid": {
                "enabled": False,
            },
            "multi_hop": {
                "enabled": True,
                "seed_k": 1,
                "hop2_max_per_seed": 3,
                "neighbor_window": 1,
                "final_top_k": 10,
                "final_semantic_weight": 0.55,
                "final_anchor_weight": 0.30,
                "final_seed_weight": 0.15,
                "anchor_weights": {
                    "symbol_full_name": 1.0,
                    "symbol_name": 0.85,
                    "parent_symbol": 0.65,
                    "same_file_neighbor": 0.5,
                },
            },
        },
    )

    retrieved_documents = rag.call("Where is alpha defined?")

    assert retrieved_documents[0].doc_indices == [0]


def test_build_citations_from_retrieved_documents_extracts_metadata():
    documents = [
        _make_code_doc(
            text="def alpha():\n    return 1",
            file_path="pkg/foo.py",
            symbol_full_name="pkg.foo.alpha",
            start_line=10,
            end_line=12,
            doc_type="py",
        ),
        _make_code_doc(
            text="class Beta:\n    pass",
            file_path="pkg/bar.py",
            symbol_name="Beta",
            start_line=20,
            end_line=21,
            doc_type="py",
        ),
    ]

    citations = build_citations_from_retrieved_documents(documents)

    assert citations[0] == {
        "index": 1,
        "file_path": "pkg/foo.py",
        "start_line": 10,
        "end_line": 12,
        "symbol": "pkg.foo.alpha",
        "chunk_type": "py",
        "score": 1.0,
        "snippet": "def alpha():\n    return 1",
    }
    assert citations[1]["index"] == 2
    assert citations[1]["symbol"] == "Beta"
    assert citations[1]["score"] == 0.0


def test_build_citations_from_retrieved_documents_truncates_snippet():
    document = _make_code_doc(
        text="x" * 1200,
        file_path="pkg/long.py",
        start_line=1,
        end_line=100,
    )

    citations = build_citations_from_retrieved_documents(
        [document],
        max_snippet_chars=32,
    )

    assert citations[0]["snippet"] == "x" * 32 + "\n...[truncated]"


def test_build_citations_from_retrieved_documents_handles_missing_metadata():
    document = Document(text="plain text", meta_data={})

    citations = build_citations_from_retrieved_documents([document])

    assert citations == [
        {
            "index": 1,
            "file_path": "unknown",
            "start_line": 0,
            "end_line": 0,
            "symbol": "",
            "chunk_type": "",
            "score": 1.0,
            "snippet": "plain text",
        }
    ]
