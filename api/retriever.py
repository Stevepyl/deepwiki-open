"""FAISS-backed code retrieval for agent tools.

This module intentionally contains only the retrieval half of ``api.rag``:
embedder setup, database preparation, embedding validation, and FAISS lookup.
It does not own conversation memory, prompt generation, or model providers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document

from api.config import configs, get_embedder_type

logger = logging.getLogger(__name__)

_CACHE_MAX = 4


def _normalize_repo_input(repo_url_or_path: str) -> str:
    value = repo_url_or_path.strip()
    if value.startswith(("http://", "https://")):
        return value.rstrip("/")
    return os.path.normpath(value)


def _embedding_vector_length(doc: Document) -> int:
    vector = getattr(doc, "vector", None)
    if vector is None:
        return 0
    try:
        if hasattr(vector, "shape"):
            if len(vector.shape) == 0:
                return 0
            return int(vector.shape[-1])
        if hasattr(vector, "__len__"):
            return int(len(vector))
    except Exception:
        return 0
    return 0


def _validate_and_filter_embeddings(documents: list[Document]) -> list[Document]:
    if not documents:
        logger.warning("No documents provided for embedding validation")
        return []

    sizes: dict[int, int] = {}
    for doc in documents:
        size = _embedding_vector_length(doc)
        if size > 0:
            sizes[size] = sizes.get(size, 0) + 1

    if not sizes:
        logger.error("No valid embeddings found in any documents")
        return []

    target_size = max(sizes, key=sizes.get)
    valid_documents = [
        doc
        for doc in documents
        if _embedding_vector_length(doc) == target_size
    ]
    if len(valid_documents) < len(documents):
        logger.warning(
            "Filtered %s documents with missing or mismatched embeddings",
            len(documents) - len(valid_documents),
        )
    return valid_documents


def _get_faiss_retriever_config() -> dict[str, Any]:
    retriever_cfg = dict(configs.get("retriever", {}) or {})
    retriever_cfg.pop("symbol_alpha", None)
    retriever_cfg.pop("multi_hop", None)
    return retriever_cfg


def _new_database_manager():
    from api.data_pipeline import DatabaseManager  # noqa: PLC0415

    return DatabaseManager()


def _new_embedder(embedder_type: str):
    from api.tools.embedder import get_embedder  # noqa: PLC0415

    return get_embedder(embedder_type=embedder_type)


class CodeRetriever:
    """Pure FAISS-backed retriever for one repository."""

    def __init__(self, embedder_type: Optional[str] = None) -> None:
        self.embedder_type = embedder_type or get_embedder_type()
        self.embedder = _new_embedder(self.embedder_type)
        self.query_embedder = self._build_query_embedder()
        self.db_manager = _new_database_manager()
        self.transformed_docs: list[Document] = []
        self.retriever: Optional[FAISSRetriever] = None

    def _build_query_embedder(self):
        if self.embedder_type != "ollama":
            return self.embedder

        def single_string_embedder(query):
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ollama embedder only supports a single string")
                query = query[0]
            return self.embedder(input=query)

        return single_string_embedder

    def prepare(
        self,
        repo_url_or_path: str,
        repo_type: str = "local",
        access_token: Optional[str] = None,
        excluded_dirs: Optional[list[str]] = None,
        excluded_files: Optional[list[str]] = None,
        included_dirs: Optional[list[str]] = None,
        included_files: Optional[list[str]] = None,
    ) -> None:
        normalized_repo = _normalize_repo_input(repo_url_or_path)
        docs = self.db_manager.prepare_database(
            normalized_repo,
            repo_type,
            access_token,
            embedder_type=self.embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )
        self.transformed_docs = _validate_and_filter_embeddings(docs)
        if not self.transformed_docs:
            raise ValueError(f"No usable embeddings for {normalized_repo}")

        self.retriever = FAISSRetriever(
            **_get_faiss_retriever_config(),
            embedder=self.query_embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        if self.retriever is None:
            raise RuntimeError("CodeRetriever.prepare() must be called before retrieve()")

        result = self.retriever(query, top_k=top_k) if top_k else self.retriever(query)
        outputs = result if isinstance(result, list) else [result]
        if not outputs:
            return []

        doc_indices = getattr(outputs[0], "doc_indices", []) or []
        return [
            self.transformed_docs[index]
            for index in doc_indices
            if 0 <= index < len(self.transformed_docs)
        ]


_CACHE: OrderedDict[tuple[str, str], CodeRetriever] = OrderedDict()
_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}
_CACHE_GUARD = asyncio.Lock()


def _key(repo_url_or_path: str, embedder_type: str) -> tuple[str, str]:
    return (_normalize_repo_input(repo_url_or_path), embedder_type)


async def get_or_build_retriever(
    repo_url_or_path: str,
    repo_type: str = "local",
    access_token: Optional[str] = None,
) -> CodeRetriever:
    embedder_type = get_embedder_type()
    cache_key = _key(repo_url_or_path, embedder_type)

    async with _CACHE_GUARD:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            return cached
        lock = _LOCKS.setdefault(cache_key, asyncio.Lock())

    async with lock:
        async with _CACHE_GUARD:
            cached = _CACHE.get(cache_key)
            if cached is not None:
                _CACHE.move_to_end(cache_key)
                return cached

        retriever = CodeRetriever(embedder_type=embedder_type)
        await asyncio.to_thread(
            retriever.prepare,
            repo_url_or_path,
            repo_type,
            access_token,
        )

        async with _CACHE_GUARD:
            _CACHE[cache_key] = retriever
            _CACHE.move_to_end(cache_key)
            while len(_CACHE) > _CACHE_MAX:
                evicted_key, _ = _CACHE.popitem(last=False)
                if evicted_key != cache_key:
                    _LOCKS.pop(evicted_key, None)
                logger.info("CodeRetriever cache evicted %s", evicted_key)
        return retriever
