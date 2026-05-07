"""Agent chat API handlers."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel

from api.agent.config import get_agent_config, get_all_agent_configs, get_tools_for_agent
from api.agent.filtered_tools import wrap_tools_with_filters
from api.agent.handler_utils import (
    compute_repo_name,
    compute_repo_path,
    language_name,
    parse_request_filters,
    safe_close,
    safe_send,
    send_event,
)
from api.agent.loop import run_agent_loop
from api.agent.message import AgentMessage
from api.agent.provider import UnifiedProvider
from api.agent.stream_events import ErrorEvent, FinishEvent, StreamEvent
from api.config import configs
from api.data_pipeline import download_repo
from api.simple_chat import ChatMessage

logger = logging.getLogger(__name__)

_ALLOWED_CHAT_AGENTS = frozenset({"wiki", "explore", "deep-research"})

OnEvent = Callable[[StreamEvent], Awaitable[None]]


class AgentChatRequest(BaseModel):
    repo_url: str
    type: str = "github"
    token: Optional[str] = None
    provider: str = configs.get("default_provider", "openai")
    model: Optional[str] = None
    language: str = "en"
    messages: list[ChatMessage]
    agent_name: str = "explore"
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None


async def _emit_error(on_event: OnEvent, error: str, code: str) -> None:
    await on_event(ErrorEvent(error=error, code=code))
    await on_event(FinishEvent(finish_reason="error"))


async def _run_agent_chat(req: AgentChatRequest, on_event: OnEvent) -> None:
    if req.agent_name not in _ALLOWED_CHAT_AGENTS:
        allowed = sorted(_ALLOWED_CHAT_AGENTS)
        await _emit_error(
            on_event,
            f"Unknown agent '{req.agent_name}'. Allowed: {allowed}",
            "unknown_agent",
        )
        return

    if not req.messages:
        await _emit_error(on_event, "messages must contain at least one item", "empty_messages")
        return

    messages = AgentMessage.from_chat_messages(
        [message.model_dump() for message in req.messages]
    )
    if not messages:
        await _emit_error(on_event, "messages did not contain any usable chat messages", "empty_messages")
        return

    if messages[-1].role != "user":
        await _emit_error(on_event, "last message must have role 'user'", "last_message_not_user")
        return

    repo_path = compute_repo_path(req.repo_url, req.type)
    try:
        await asyncio.to_thread(download_repo, req.repo_url, repo_path, req.type, req.token)
    except Exception as exc:
        logger.exception("agent-chat: download_repo failed for %s", req.repo_url)
        await _emit_error(on_event, f"Failed to clone repository: {exc}", "clone_failed")
        return

    filters = parse_request_filters(
        req.excluded_dirs,
        req.excluded_files,
        req.included_dirs,
        req.included_files,
    )
    agent_config = get_agent_config(req.agent_name)
    tools = wrap_tools_with_filters(
        get_tools_for_agent(agent_config, repo_path),
        filters,
        repo_path,
    )

    try:
        provider = UnifiedProvider(req.provider, req.model or "")
    except Exception as exc:
        await _emit_error(on_event, f"Provider initialisation failed: {exc}", "provider_error")
        return

    system_prompt_vars = {
        "repo_type": req.type,
        "repo_url": req.repo_url,
        "repo_name": compute_repo_name(req.repo_url, req.type),
        "language_name": language_name(req.language),
    }

    try:
        async for evt in run_agent_loop(
            agent_config,
            messages,
            provider,
            tools,
            repo_path,
            system_prompt_vars,
        ):
            await on_event(evt)
    except Exception as exc:
        logger.exception("agent-chat: run_agent_loop failed")
        await _emit_error(on_event, str(exc), "internal_error")


async def handle_agent_chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
        try:
            req = AgentChatRequest.model_validate(raw)
        except Exception as exc:
            await safe_send(websocket, ErrorEvent(error=f"Invalid request: {exc}", code="invalid_request"))
            await safe_send(websocket, FinishEvent(finish_reason="error"))
            return

        await _run_agent_chat(req, lambda evt: send_event(websocket, evt))
    except WebSocketDisconnect:
        logger.info("agent-chat: client disconnected")
    except Exception as exc:
        logger.exception("agent-chat: unhandled websocket exception")
        await safe_send(websocket, ErrorEvent(error=str(exc), code="internal_error"))
        await safe_send(websocket, FinishEvent(finish_reason="error"))
    finally:
        await safe_close(websocket)


async def agent_chat_stream(request: AgentChatRequest) -> StreamingResponse:
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=64)

    async def on_event(evt: StreamEvent) -> None:
        await queue.put(evt)

    async def produce() -> None:
        try:
            await _run_agent_chat(request, on_event)
        finally:
            await queue.put(None)

    async def body():
        producer = asyncio.create_task(produce())
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield evt.model_dump_json() + "\n"
        finally:
            if not producer.done():
                producer.cancel()

    return StreamingResponse(body(), media_type="application/x-ndjson")


def get_agent_info() -> list[dict]:
    configs = get_all_agent_configs()
    return [
        {
            "name": config.name,
            "description": config.description,
            "mode": config.mode,
            "max_steps": config.max_steps,
            "allowed_tools": list(config.allowed_tools),
        }
        for name, config in sorted(configs.items())
        if name in _ALLOWED_CHAT_AGENTS
    ]
