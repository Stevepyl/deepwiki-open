from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query

from api.contracts import IndexBuildRequest, IndexBuildResponse, IndexStatusResponse
from api.services.indexing_service import IndexingService


router = APIRouter(prefix="/api/repos/{repo_id}/index", tags=["indexing"])


@router.post("", response_model=IndexBuildResponse)
async def build_index(
    repo_id: str,
    request: IndexBuildRequest,
    repo_location: str = Query(..., description="Repository URL or local path"),
    repo_type: str = Query("github", description="Repository type"),
    authorization: str | None = Header(None),
) -> IndexBuildResponse:
    expected_repo_id = IndexingService.repo_id_for_location(repo_location)
    if repo_id != expected_repo_id:
        raise HTTPException(
            status_code=400,
            detail=f"repo_id does not match repo_location. Expected {expected_repo_id}.",
        )
    token = authorization.removeprefix("Bearer ").strip() if authorization else None
    return IndexingService().build_index(
        repo_id=repo_id,
        repo_location=repo_location,
        repo_type=repo_type,
        token=token,
        request=request,
    )


@router.get("/status", response_model=IndexStatusResponse)
async def get_index_status(repo_id: str) -> IndexStatusResponse:
    status = IndexingService().get_status(repo_id)
    if status.status == "not_found":
        raise HTTPException(status_code=404, detail="Repository index not found")
    return status
