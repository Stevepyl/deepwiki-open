from __future__ import annotations

from api.contracts import EdgeListResponse, ReferenceListResponse, SymbolListResponse
from api.knowledge_store import RepoKnowledgeStore


class GraphService:
    """Read API for symbols, references and graph relations."""

    def __init__(self, store: RepoKnowledgeStore | None = None):
        self.store = store or RepoKnowledgeStore()

    def list_symbols(self, repo_id: str, query: str | None = None, limit: int = 100) -> SymbolListResponse:
        return SymbolListResponse(
            repo_id=repo_id,
            symbols=self.store.list_symbols(repo_id, query=query, limit=limit),
        )

    def list_references(
        self,
        repo_id: str,
        *,
        source_scope: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> ReferenceListResponse:
        return ReferenceListResponse(
            repo_id=repo_id,
            references=self.store.list_references(
                repo_id,
                source_scope=source_scope,
                target=target,
                limit=limit,
            ),
        )

    def list_edges(self, repo_id: str, symbol: str | None = None, limit: int = 100) -> EdgeListResponse:
        return EdgeListResponse(
            repo_id=repo_id,
            edges=self.store.list_edges(repo_id, symbol=symbol, limit=limit),
        )
