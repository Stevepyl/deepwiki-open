from __future__ import annotations

from fastapi import APIRouter, Query

from api.contracts import EdgeListResponse, ReferenceListResponse, SymbolListResponse
from api.services.graph_service import GraphService


router = APIRouter(prefix="/api/repos/{repo_id}", tags=["graph"])


@router.get("/symbols", response_model=SymbolListResponse)
async def list_symbols(
    repo_id: str,
    query: str | None = Query(None, description="Optional symbol/path search term"),
    limit: int = Query(100, ge=1, le=500),
) -> SymbolListResponse:
    return GraphService().list_symbols(repo_id, query=query, limit=limit)


@router.get("/graph/references", response_model=ReferenceListResponse)
async def list_references(
    repo_id: str,
    source_scope: str | None = Query(None, description="Filter by referencing scope"),
    target: str | None = Query(None, description="Filter by referenced target"),
    limit: int = Query(100, ge=1, le=500),
) -> ReferenceListResponse:
    return GraphService().list_references(
        repo_id,
        source_scope=source_scope,
        target=target,
        limit=limit,
    )


@router.get("/graph/edges", response_model=EdgeListResponse)
async def list_edges(
    repo_id: str,
    symbol: str | None = Query(None, description="Filter edges related to this symbol"),
    limit: int = Query(100, ge=1, le=500),
) -> EdgeListResponse:
    return GraphService().list_edges(repo_id, symbol=symbol, limit=limit)
