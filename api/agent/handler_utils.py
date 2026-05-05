"""Shared helpers for agent-backed API handlers."""

from __future__ import annotations

import os
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel

from api.agent.loop import run_agent_loop
from api.agent.message import AgentMessage
from api.agent.provider import UnifiedProvider
from api.agent.stream_events import StreamEvent
from api.utils.filters import ParsedFilters

_ADALFLOW_ROOT = os.path.expanduser(os.path.join("~", ".adalflow"))

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}


def compute_repo_name(repo_url: str, repo_type: str) -> str:
    """Mirrors DatabaseManager._extract_repo_name_from_url logic."""
    url_parts = repo_url.rstrip("/").split("/")
    if repo_type in ("github", "gitlab", "bitbucket") and len(url_parts) >= 5:
        owner = url_parts[-2]
        repo = url_parts[-1].replace(".git", "")
        return f"{owner}_{repo}"
    return url_parts[-1].replace(".git", "")


def compute_repo_path(repo_url: str, repo_type: str) -> str:
    repo_name = compute_repo_name(repo_url, repo_type)
    return os.path.join(_ADALFLOW_ROOT, "repos", repo_name)


def parse_request_filters(
    excluded_dirs: str | None,
    excluded_files: str | None,
    included_dirs: str | None,
    included_files: str | None,
) -> ParsedFilters:
    """Build ParsedFilters from request filter string fields."""
    return ParsedFilters.from_strings(
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files,
    )


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


async def send_event(ws: WebSocket, evt: BaseModel) -> None:
    await ws.send_json(evt.model_dump())


async def send_tagged_event(ws: WebSocket, evt: StreamEvent, **tags: Any) -> None:
    """Send a StreamEvent with extra phase/page metadata."""
    payload = evt.model_dump()
    payload.update(tags)
    await ws.send_json(payload)


async def safe_send(ws: WebSocket, evt: BaseModel) -> None:
    try:
        await ws.send_json(evt.model_dump())
    except Exception:
        pass


async def safe_close(ws: WebSocket) -> None:
    try:
        await ws.close()
    except Exception:
        pass


async def consume_agent_loop(
    agent_config,
    messages: list[AgentMessage],
    provider: UnifiedProvider,
    tools: dict,
    repo_path: str,
    system_prompt_vars: dict[str, str],
    on_event,
) -> None:
    async for evt in run_agent_loop(
        agent_config, messages, provider, tools, repo_path, system_prompt_vars
    ):
        await on_event(evt)
