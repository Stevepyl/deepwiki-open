from __future__ import annotations

from api.contracts import IndexBuildRequest, IndexBuildResponse, IndexStatusResponse
from api.knowledge_store import RepoKnowledgeStore, default_repo_id, repo_display_name
from api.rag_cache import get_prepared_rag


class IndexingService:
    """Repository knowledge-building boundary.

    Current implementation wraps the existing RAG prepare path. Future
    refactors can move clone/filter/chunk/embed/index/graph steps behind this
    service without changing the stable API.
    """

    def __init__(self, store: RepoKnowledgeStore | None = None):
        self.store = store or RepoKnowledgeStore()

    def build_index(
        self,
        *,
        repo_id: str,
        repo_location: str,
        repo_type: str,
        token: str | None,
        request: IndexBuildRequest,
    ) -> IndexBuildResponse:
        prepared = get_prepared_rag(
            repo_url=repo_location,
            repo_type=repo_type,
            token=token,
            provider=request.provider,
            model=request.model,
            excluded_dirs=request.excluded_dirs,
            excluded_files=request.excluded_files,
            included_dirs=request.included_dirs,
            included_files=request.included_files,
        )
        self.store.save_repo(
            repo_id,
            repo_display_name(repo_location),
            repo_location,
            prepared.documents_count,
            "indexed",
        )
        return IndexBuildResponse(
            repo_id=repo_id,
            status="indexed",
            cache_hit=prepared.cache_hit,
            documents_count=prepared.documents_count,
            prepare_latency_sec=prepared.prepare_latency_sec,
        )

    def get_status(self, repo_id: str) -> IndexStatusResponse:
        row = self.store.get_repo(repo_id)
        if row is None:
            return IndexStatusResponse(repo_id=repo_id, status="not_found", chunk_count=0)
        return IndexStatusResponse(
            repo_id=repo_id,
            status=row["status"],
            chunk_count=row["chunk_count"],
        )

    @staticmethod
    def repo_id_for_location(location: str) -> str:
        return default_repo_id(location)
