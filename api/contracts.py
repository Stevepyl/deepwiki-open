"""Stable API contracts for the refactored backend capability boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RepoSourceType = Literal["github", "gitlab", "bitbucket", "local"]


class RepoCreateRequest(BaseModel):
    source_type: RepoSourceType = Field("github", description="Repository source type")
    repo_url: str | None = Field(None, description="Remote repository URL")
    local_path: str | None = Field(None, description="Local repository path")
    token: str | None = Field(None, description="Optional access token")


class RepoResponse(BaseModel):
    repo_id: str
    name: str
    source_type: RepoSourceType | str
    location: str
    status: str
    chunk_count: int = 0


class IndexBuildRequest(BaseModel):
    provider: str | None = Field("google", description="Model provider")
    model: str | None = Field(None, description="Model name")
    excluded_dirs: list[str] | None = None
    excluded_files: list[str] | None = None
    included_dirs: list[str] | None = None
    included_files: list[str] | None = None


class IndexBuildResponse(BaseModel):
    repo_id: str
    status: str
    cache_hit: bool
    documents_count: int
    prepare_latency_sec: float


class IndexStatusResponse(BaseModel):
    repo_id: str
    status: str
    chunk_count: int = 0


class SymbolResponse(BaseModel):
    symbol_id: str
    name: str
    symbol_type: str
    file_path: str
    module: str | None = None
    start_line: int = 0
    end_line: int = 0
    signature: str = ""
    runtime_tags: list[str] = Field(default_factory=list)


class SymbolListResponse(BaseModel):
    repo_id: str
    symbols: list[SymbolResponse]


class ReferenceResponse(BaseModel):
    file_path: str
    source_scope: str
    ref_type: str
    target: str
    ref_name: str
    line: int
    snippet: str = ""


class ReferenceListResponse(BaseModel):
    repo_id: str
    references: list[ReferenceResponse]


class EdgeResponse(BaseModel):
    source: str
    target: str
    relation: str


class EdgeListResponse(BaseModel):
    repo_id: str
    edges: list[EdgeResponse]


class CapabilityResponse(BaseModel):
    name: str
    status: Literal["available", "planned"]
    description: str
