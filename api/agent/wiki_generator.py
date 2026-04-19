"""
Agent Wiki generation backend (subtask 12, Path B).

Exposes a single WebSocket handler: handle_agent_wiki_websocket.

Protocol (server → client, discriminated by "type" field):
  Phase 1 (planning):
    text_delta | tool_call_start | tool_call_end (+ phase="planning")
    wiki_structure_ready  -- Phase 1 complete, structure JSON ready
    wiki_structure_error  -- Phase 1 fatal; WebSocket closed after this

  Phase 2 (writing, one page at a time):
    text_delta | tool_call_start | tool_call_end (+ phase="writing", page_index, page_id)
    wiki_page_done   -- one page complete
    wiki_page_error  -- single page timed out; session continues with next page

  Final:
    finish
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel

from api.agent.config import get_agent_config, get_tools_for_agent
from api.agent.loop import run_agent_loop
from api.agent.message import AgentMessage
from api.agent.provider import UnifiedProvider
from api.agent.stream_events import (
    FinishEvent,
    StreamEvent,
    TextDelta,
    WikiPageDone,
    WikiPageError,
    WikiStructureError,
    WikiStructureReady,
)
from api.agent.filtered_tools import wrap_tools_with_filters
from api.data_pipeline import download_repo
from api.utils.filters import ParsedFilters
from api.utils.repo_tree import build_file_tree, read_repo_readme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE1_TIMEOUT_S = 300   # 5 min planner ceiling
_PHASE2_TIMEOUT_S = 300   # 5 min per-page writer ceiling
_ADALFLOW_ROOT = os.path.expanduser(os.path.join("~", ".adalflow"))

_COMPREHENSIVE_INSTRUCTION = (
    "Generate 8-12 pages. Include a 'sections' array that groups pages into "
    "logical categories (max 2 nesting levels). Include a 'rootSections' array "
    "with the IDs of top-level sections."
)
_CONCISE_INSTRUCTION = (
    "Generate 4-6 pages. Omit 'sections' and 'rootSections' fields (set them to null). "
    "Keep the page list flat."
)

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class AgentWikiRequest(BaseModel):
    repo_url: str
    type: str = "github"
    token: Optional[str] = None
    provider: str = "google"
    model: Optional[str] = None
    language: str = "en"
    comprehensive: bool = True
    file_tree_hint: Optional[str] = None
    readme_hint: Optional[str] = None
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None

# ---------------------------------------------------------------------------
# Repo-path helpers
# ---------------------------------------------------------------------------


def _compute_repo_name(req: AgentWikiRequest) -> str:
    """Mirrors DatabaseManager._extract_repo_name_from_url logic."""
    url_parts = req.repo_url.rstrip("/").split("/")
    if req.type in ("github", "gitlab", "bitbucket") and len(url_parts) >= 5:
        owner = url_parts[-2]
        repo = url_parts[-1].replace(".git", "")
        return f"{owner}_{repo}"
    return url_parts[-1].replace(".git", "")


def _compute_repo_path(req: AgentWikiRequest) -> str:
    repo_name = _compute_repo_name(req)
    return os.path.join(_ADALFLOW_ROOT, "repos", repo_name)


def _parse_request_filters(req: AgentWikiRequest) -> ParsedFilters:
    """Build ParsedFilters from the request's filter string fields."""
    return ParsedFilters.from_strings(
        excluded_dirs=req.excluded_dirs,
        excluded_files=req.excluded_files,
        included_dirs=req.included_dirs,
        included_files=req.included_files,
    )


# ---------------------------------------------------------------------------
# Language name lookup (mirrors existing convention in prompts)
# ---------------------------------------------------------------------------

_LANGUAGE_NAMES: dict[str, str] = {
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


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


# ---------------------------------------------------------------------------
# system_prompt_vars builders
# ---------------------------------------------------------------------------


def _planner_prompt_vars(req: AgentWikiRequest) -> dict[str, str]:
    return {
        "repo_type": req.type,
        "repo_url": req.repo_url,
        "repo_name": _compute_repo_name(req),
        "language_name": _language_name(req.language),
        "comprehensive_instruction": (
            _COMPREHENSIVE_INSTRUCTION if req.comprehensive else _CONCISE_INSTRUCTION
        ),
    }


def _writer_prompt_vars(req: AgentWikiRequest) -> dict[str, str]:
    return {
        "repo_type": req.type,
        "repo_url": req.repo_url,
        "repo_name": _compute_repo_name(req),
        "language_name": _language_name(req.language),
    }


# ---------------------------------------------------------------------------
# User prompt formatters
# ---------------------------------------------------------------------------


def _format_planner_user_prompt(
    file_tree: str,
    readme: str,
    comprehensive: bool,
    language: str,
) -> str:
    mode = "comprehensive (8-12 pages with sections)" if comprehensive else "concise (4-6 pages, flat)"
    readme_section = f"<readme>\n{readme[:4000]}\n</readme>" if readme else "<readme>(not found)</readme>"
    return (
        f"Plan a {mode} wiki for this repository.\n\n"
        f"File tree hint (verify paths before using in filePaths):\n"
        f"<file_tree>\n{file_tree[:8000]}\n</file_tree>\n\n"
        f"{readme_section}\n\n"
        "Use glob/ls/read to explore the repository structure and verify file paths. "
        "Then output the complete wiki plan as a single JSON object."
    )


def _format_writer_user_prompt(page: dict[str, Any], language: str) -> str:
    hints = "\n".join(f"- {p}" for p in page.get("filePaths", []))
    related = page.get("relatedPages", [])
    related_str = ", ".join(related) if related else "none"
    return (
        f"Write a wiki page titled: **{page['title']}**\n\n"
        f"Description: {page.get('description', '')}\n\n"
        f"Relevant file paths (hints — verify each before citing):\n{hints or '(none provided)'}\n\n"
        f"Related pages: {related_str}\n\n"
        "Follow the explore-then-write workflow: verify hints, explore related files, then write. "
        f"Write all prose in {_language_name(language)}."
    )


# ---------------------------------------------------------------------------
# WebSocket send helpers
# ---------------------------------------------------------------------------


async def _send_event(ws: WebSocket, evt: BaseModel) -> None:
    await ws.send_json(evt.model_dump())


async def _send_tagged_event(ws: WebSocket, evt: StreamEvent, **tags: Any) -> None:
    """Send a StreamEvent with extra phase/page metadata (envelope pattern)."""
    payload = evt.model_dump()
    payload.update(tags)
    await ws.send_json(payload)


async def _safe_send(ws: WebSocket, evt: BaseModel) -> None:
    try:
        await ws.send_json(evt.model_dump())
    except Exception:
        pass


async def _safe_close(ws: WebSocket) -> None:
    try:
        await ws.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# JSON structure parser
# ---------------------------------------------------------------------------


def _validate_wiki_structure(data: dict[str, Any]) -> bool:
    """Light-weight schema check — avoids circular import with api.api."""
    if not isinstance(data, dict):
        return False
    if not all(k in data for k in ("id", "title", "description", "pages")):
        return False
    if not isinstance(data.get("pages"), list):
        return False
    for page in data["pages"]:
        if not isinstance(page, dict):
            return False
        if not all(k in page for k in ("id", "title")):
            return False
        if "content" not in page:
            page["content"] = ""
        if "filePaths" not in page:
            page["filePaths"] = []
        if "importance" not in page:
            page["importance"] = "medium"
        if "relatedPages" not in page:
            page["relatedPages"] = []
    return True


def parse_wiki_structure(raw: str) -> dict[str, Any] | None:
    """Multi-layer JSON extraction from planner LLM output.

    Layer 1: strip markdown code fence
    Layer 2: find outermost {...} in the text
    Layer 3: json.loads
    Layer 4: light-weight schema validation
    """
    text = raw.strip()

    # Layer 1 — strip markdown fence
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Layer 2 — find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        logger.error("parse_wiki_structure: no JSON object found in planner output")
        return None

    # Layer 3 — parse
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.error("parse_wiki_structure: JSON decode failed: %s", exc)
        return None

    # Layer 4 — validate
    if not _validate_wiki_structure(data):
        logger.error("parse_wiki_structure: schema validation failed")
        return None

    return data


# ---------------------------------------------------------------------------
# Page flattening (section-order traversal)
# ---------------------------------------------------------------------------


def _collect_section_pages(section_id: str, structure: dict[str, Any], seen: set[str]) -> list[str]:
    """Recursively collect page IDs from a section and its subsections."""
    section = next(
        (s for s in (structure.get("sections") or []) if s["id"] == section_id),
        None,
    )
    if section is None:
        return []
    page_ids: list[str] = []
    for page_id in section.get("pages", []):
        if page_id not in seen:
            seen.add(page_id)
            page_ids.append(page_id)
    for sub_id in section.get("subsections") or []:
        page_ids.extend(_collect_section_pages(sub_id, structure, seen))
    return page_ids


def _flatten_pages_in_section_order(structure: dict[str, Any]) -> list[dict[str, Any]]:
    """Return pages in visual order: root sections first, orphan pages last."""
    page_by_id = {p["id"]: p for p in structure.get("pages", [])}
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []

    for section_id in structure.get("rootSections") or []:
        for page_id in _collect_section_pages(section_id, structure, seen):
            if page_id in page_by_id:
                ordered.append(page_by_id[page_id])

    for page in structure.get("pages", []):
        if page["id"] not in seen:
            ordered.append(page)

    return ordered


# ---------------------------------------------------------------------------
# Phase coroutine wrappers (needed for asyncio.wait_for)
# ---------------------------------------------------------------------------


async def _consume_agent_loop(
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


# ---------------------------------------------------------------------------
# Phase 1 — planner
# ---------------------------------------------------------------------------


async def _run_planner_phase(
    ws: WebSocket,
    req: AgentWikiRequest,
    repo_path: str,
    file_tree: str,
    readme: str,
    filters: ParsedFilters,
) -> dict[str, Any] | None:
    config = get_agent_config("wiki-planner")
    tools = wrap_tools_with_filters(get_tools_for_agent(config, repo_path), filters, repo_path)
    try:
        provider = UnifiedProvider(req.provider, req.model or "")
    except (ValueError, Exception) as exc:
        await _send_event(ws, WikiStructureError(
            code="internal_error",
            message=f"Provider initialisation failed: {exc}",
        ))
        return None

    user_prompt = _format_planner_user_prompt(
        file_tree, readme, req.comprehensive, req.language,
    )
    collected: list[str] = []

    async def on_event(evt: StreamEvent) -> None:
        if isinstance(evt, TextDelta):
            collected.append(evt.content)
        await _send_tagged_event(ws, evt, phase="planning")

    try:
        await asyncio.wait_for(
            _consume_agent_loop(
                config,
                [AgentMessage.user(user_prompt)],
                provider,
                tools,
                repo_path,
                _planner_prompt_vars(req),
                on_event,
            ),
            timeout=_PHASE1_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        await _send_event(ws, WikiStructureError(
            code="planner_timeout",
            message=f"Wiki planner exceeded {_PHASE1_TIMEOUT_S}s",
        ))
        return None
    except Exception as exc:
        logger.exception("Planner phase raised unexpected error")
        await _send_event(ws, WikiStructureError(
            code="internal_error", message=str(exc),
        ))
        return None

    raw = "".join(collected)
    structure = parse_wiki_structure(raw)
    if structure is None:
        await _send_event(ws, WikiStructureError(
            code="json_parse_error",
            message="Planner output could not be parsed as a valid wiki structure",
        ))
        return None

    await _send_event(ws, WikiStructureReady(structure=structure))
    return structure


# ---------------------------------------------------------------------------
# Phase 2 — writer (per page)
# ---------------------------------------------------------------------------


async def _run_writer_phase(
    ws: WebSocket,
    req: AgentWikiRequest,
    repo_path: str,
    page: dict[str, Any],
    page_index: int,
    total_pages: int,
    filters: ParsedFilters,
) -> None:
    config = get_agent_config("wiki-writer")
    tools = wrap_tools_with_filters(get_tools_for_agent(config, repo_path), filters, repo_path)
    try:
        provider = UnifiedProvider(req.provider, req.model or "")
    except Exception as exc:
        await _send_event(ws, WikiPageError(
            page_id=page["id"],
            page_index=page_index,
            code="writer_failed",
            message=f"Provider initialisation failed: {exc}",
        ))
        return

    user_prompt = _format_writer_user_prompt(page, req.language)
    content: list[str] = []

    async def on_event(evt: StreamEvent) -> None:
        if isinstance(evt, TextDelta):
            content.append(evt.content)
        await _send_tagged_event(
            ws, evt,
            phase="writing",
            page_index=page_index,
            page_id=page["id"],
        )

    try:
        await asyncio.wait_for(
            _consume_agent_loop(
                config,
                [AgentMessage.user(user_prompt)],
                provider,
                tools,
                repo_path,
                _writer_prompt_vars(req),
                on_event,
            ),
            timeout=_PHASE2_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        await _send_event(ws, WikiPageError(
            page_id=page["id"],
            page_index=page_index,
            code="writer_timeout",
            message=f"Writer exceeded {_PHASE2_TIMEOUT_S}s for page '{page['title']}'",
        ))
        return
    except Exception as exc:
        logger.exception("Writer phase raised unexpected error for page '%s'", page.get("title"))
        await _send_event(ws, WikiPageError(
            page_id=page["id"],
            page_index=page_index,
            code="writer_failed",
            message=str(exc),
        ))
        return

    await _send_event(ws, WikiPageDone(
        page_id=page["id"],
        page_title=page["title"],
        page_index=page_index,
        total_pages=total_pages,
        content="".join(content),
    ))


# ---------------------------------------------------------------------------
# Main WebSocket handler
# ---------------------------------------------------------------------------


async def handle_agent_wiki_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
        try:
            req = AgentWikiRequest.model_validate(raw)
        except Exception as exc:
            await _safe_send(websocket, WikiStructureError(
                code="internal_error",
                message=f"Invalid request: {exc}",
            ))
            return

        # Step 1: ensure repo is cloned locally (idempotent)
        repo_path = _compute_repo_path(req)
        try:
            await asyncio.to_thread(
                download_repo, req.repo_url, repo_path, req.type, req.token,
            )
        except Exception as exc:
            logger.exception("download_repo failed for %s", req.repo_url)
            await _safe_send(websocket, WikiStructureError(
                code="clone_failed",
                message=f"Failed to clone repository: {exc}",
            ))
            return

        # Step 2: parse filter rules from request
        filters = _parse_request_filters(req)

        # Step 3: build file context (prefer front-end hints to avoid redundant walk;
        #          apply user filters so the planner hint is already scoped)
        file_tree = req.file_tree_hint or build_file_tree(repo_path, filters=filters)
        readme = req.readme_hint or read_repo_readme(repo_path)

        # Step 4: Phase 1 — planner produces wiki structure
        structure = await _run_planner_phase(
            websocket, req, repo_path, file_tree, readme, filters,
        )
        if structure is None:
            return

        # Step 5: Phase 2 — writer generates content for each page (sequential)
        pages = _flatten_pages_in_section_order(structure)
        total = len(pages)
        for idx, page in enumerate(pages):
            await _run_writer_phase(websocket, req, repo_path, page, idx, total, filters)

        await _send_event(websocket, FinishEvent(finish_reason="stop"))

    except WebSocketDisconnect:
        logger.info("agent-wiki: client disconnected")
    except Exception as exc:
        logger.exception("agent-wiki: unhandled exception")
        await _safe_send(websocket, WikiStructureError(
            code="internal_error", message=str(exc),
        ))
    finally:
        await _safe_close(websocket)
