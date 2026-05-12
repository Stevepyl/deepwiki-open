from __future__ import annotations

from urllib.parse import urlparse

from api.contracts import RepoCreateRequest, RepoResponse
from api.knowledge_store import RepoKnowledgeStore, default_repo_id, repo_display_name


class RepoService:
    """Repository registration and status boundary.

    This service intentionally does not clone or index repositories. Those are
    separate capabilities owned by the indexing service.
    """

    def __init__(self, store: RepoKnowledgeStore | None = None):
        self.store = store or RepoKnowledgeStore()

    def create_repo(self, request: RepoCreateRequest) -> RepoResponse:
        location = self._repo_location(request)
        repo_id = default_repo_id(location)
        name = self._display_name(location)
        self.store.save_repo(repo_id, name, location, 0, "registered")
        return RepoResponse(
            repo_id=repo_id,
            name=name,
            source_type=request.source_type,
            location=location,
            status="registered",
            chunk_count=0,
        )

    def get_repo(self, repo_id: str) -> RepoResponse:
        row = self.store.get_repo(repo_id)
        if row is None:
            return RepoResponse(
                repo_id=repo_id,
                name="",
                source_type="unknown",
                location="",
                status="not_found",
                chunk_count=0,
            )
        return RepoResponse(
            repo_id=row["repo_id"],
            name=row["name"],
            source_type="registered",
            location=row["path"],
            status=row["status"],
            chunk_count=row["chunk_count"],
        )

    @staticmethod
    def _repo_location(request: RepoCreateRequest) -> str:
        if request.source_type == "local":
            if not request.local_path:
                raise ValueError("local_path is required for local repositories")
            return request.local_path
        if not request.repo_url:
            raise ValueError("repo_url is required for remote repositories")
        return request.repo_url

    @staticmethod
    def _display_name(location: str) -> str:
        parsed = urlparse(location)
        if parsed.scheme and parsed.netloc:
            path = parsed.path.strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return path or parsed.netloc
        return repo_display_name(location)
