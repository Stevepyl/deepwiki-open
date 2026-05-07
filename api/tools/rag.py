"""Semantic RAG search tool for repository agents."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from adalflow.core.types import Document

from api.tools.tool import Tool, ToolResult, load_description, truncate_output

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 10
_MAX_TOP_K = 20


async def _get_or_build_retriever(*args, **kwargs):
    from api.retriever import get_or_build_retriever  # noqa: PLC0415

    return await get_or_build_retriever(*args, **kwargs)


class RagTool(Tool):
    """Run semantic search over the repository's cached code embeddings."""

    name = "rag_search"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of the code or concept to find.",
            },
            "top_k": {
                "type": "integer",
                "description": (
                    f"Number of chunks to return (1-{_MAX_TOP_K}). "
                    f"Defaults to {_DEFAULT_TOP_K}."
                ),
                "minimum": 1,
                "maximum": _MAX_TOP_K,
            },
        },
        "required": ["query"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        txt_path = Path(__file__).parent / "rag.txt"
        self.description = load_description(str(txt_path))

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        query = str(params.get("query") or "").strip()
        if not query:
            return ToolResult(
                title="rag_search",
                output="Error: query is required.",
                metadata={"error": "missing_query"},
            )

        top_k = _coerce_top_k(params.get("top_k"))
        try:
            retriever = await _get_or_build_retriever(self.repo_path, repo_type="local")
            docs = await asyncio.to_thread(retriever.retrieve, query, top_k)
        except Exception as exc:
            logger.exception("rag_search failed")
            return ToolResult(
                title="rag_search",
                output=f"Error: {exc}",
                metadata={"error": "retrieval_failed"},
            )

        output = truncate_output(_format_docs(docs, self.repo_path))
        return ToolResult(
            title=f"rag_search: {query}",
            output=output,
            metadata={"matches": len(docs), "query": query, "top_k": top_k},
        )


def _coerce_top_k(value: Any) -> int:
    try:
        parsed = int(_DEFAULT_TOP_K if value is None or value == "" else value)
    except (TypeError, ValueError):
        parsed = _DEFAULT_TOP_K
    return min(max(parsed, 1), _MAX_TOP_K)


def _format_docs(docs: list[Document], repo_path: str) -> str:
    if not docs:
        return "No matches."

    lines: list[str] = []
    for index, doc in enumerate(docs, 1):
        meta = getattr(doc, "meta_data", {}) or {}
        path = _repo_relative_path(str(meta.get("file_path") or "<unknown>"), repo_path)
        loc = _format_location(path, meta.get("start_line"), meta.get("end_line"))
        snippet = (getattr(doc, "text", "") or "").strip()
        lines.append(f"### {index}. {loc}\n{snippet}\n")
    return "\n".join(lines).rstrip()


def _repo_relative_path(raw_path: str, repo_path: str) -> str:
    if not os.path.isabs(raw_path):
        return raw_path
    try:
        return str(Path(raw_path).resolve().relative_to(Path(repo_path).resolve()))
    except ValueError:
        return raw_path


def _format_location(path: str, start_line: Any, end_line: Any) -> str:
    if start_line is None:
        return path
    if end_line is None:
        return f"{path}:{start_line}"
    return f"{path}:{start_line}-{end_line}"
