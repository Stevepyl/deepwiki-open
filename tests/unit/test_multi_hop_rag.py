from types import SimpleNamespace

from adalflow.core.types import Document

from api.data_pipeline import read_all_documents
from api import rag as rag_module
from api.rag import RAG


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
    rag.doc_index_map = {}
    rag.docs_by_file = {}
    rag.docs_by_symbol_full_name = {}
    rag.docs_by_file_symbol_name = {}
    rag.docs_by_file_parent_symbol = {}
    rag.doc_position_in_file = {}
    rag._build_document_indices()
    return rag


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
            "multi_hop": {
                "enabled": False,
            },
        },
    )

    retrieved_documents = rag.call("Where is alpha defined?")

    assert retrieved_documents[0].doc_indices == [0]
    assert len(retrieved_documents[0].documents) == 1


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
