from __future__ import annotations

from fastapi import APIRouter

from api.contracts import CapabilityResponse


router = APIRouter(prefix="/api/capabilities", tags=["capabilities"])


@router.get("", response_model=list[CapabilityResponse])
async def list_capabilities() -> list[CapabilityResponse]:
    return [
        CapabilityResponse(
            name="repos",
            status="available",
            description="Repository registration and status boundary.",
        ),
        CapabilityResponse(
            name="indexing",
            status="available",
            description="Repository knowledge-building boundary wrapping the current RAG prepare path.",
        ),
        CapabilityResponse(
            name="graph",
            status="available",
            description="Read API for symbols, references and graph edges.",
        ),
        CapabilityResponse(
            name="rag_qa",
            status="planned",
            description="Stable RAG question-answering API; existing WebSocket chat remains available.",
        ),
        CapabilityResponse(
            name="wiki",
            status="planned",
            description="Stable project overview and wiki generation API; existing wiki cache APIs remain available.",
        ),
        CapabilityResponse(
            name="agentic_search",
            status="planned",
            description="Task-based multi-step search API for future frontend integration.",
        ),
    ]
