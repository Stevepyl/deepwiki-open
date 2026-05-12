from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.contracts import RepoCreateRequest, RepoResponse
from api.services.repo_service import RepoService


router = APIRouter(prefix="/api/repos", tags=["repos"])


@router.post("", response_model=RepoResponse)
async def create_repo(request: RepoCreateRequest) -> RepoResponse:
    try:
        return RepoService().create_repo(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repo(repo_id: str) -> RepoResponse:
    repo = RepoService().get_repo(repo_id)
    if repo.status == "not_found":
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.get("/{repo_id}/status", response_model=RepoResponse)
async def get_repo_status(repo_id: str) -> RepoResponse:
    repo = RepoService().get_repo(repo_id)
    if repo.status == "not_found":
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo
