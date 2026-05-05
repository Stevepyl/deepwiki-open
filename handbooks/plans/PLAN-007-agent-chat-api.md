---
number: PLAN-007
name: Agent Chat Backend API
description: Expose the existing agent infrastructure as a chat API parallel to /ws/chat and /chat/completions/stream, with WebSocket + HTTP transports and minimal frontend connectors.
status: implemented
update_at: 2026-05-06
category: plan
language: en
audience: developers-and-agents
---

# PLAN-007 — Agent Chat Backend API

## Context

The repo has two parallel chat surfaces today:
- `POST /chat/completions/stream` (`api/api.py:398` → `api/simple_chat.py:76`) — RAG-backed text stream, raw bytes.
- `WS /ws/chat` (`api/api.py:401` → `api/websocket_wiki.py:63`) — RAG-backed text stream, WebSocket.

Neither exposes the agent's reasoning or tool-use. Meanwhile a complete agent stack is already operating in production for wiki generation:

- `api/agent/loop.py:207` — `run_agent_loop(...)` async generator (ReAct, parallel tool calls, doom-loop detection, step limits, TaskTool subagents).
- `api/agent/provider.py` — `UnifiedProvider` for all 7 providers.
- `api/agent/stream_events.py` — typed event union (`TextDelta`, `ToolCallStart`, `ToolCallEnd`, `FinishEvent`, `ErrorEvent`, plus wiki-specific variants).
- `api/agent/wiki_generator.py:522` — proven WebSocket binding driving the loop.

What's missing is a chat-style entry point — multi-turn conversation in, real-time agent reasoning + tool events out. This plan adds it with two transports parallel to the existing chat, plus the minimum frontend connectors needed to use it.

## Backend changes

### B1. Extract shared helpers — `api/agent/handler_utils.py` (NEW)

`api/agent/wiki_generator.py:97-234` has private helpers that the new chat handler needs identically. Move them to a shared module to avoid duplication.

Functions to extract (currently in `wiki_generator.py`):

| Current (private, in `wiki_generator.py`) | New (public, in `handler_utils.py`) |
|---|---|
| `_compute_repo_name(req)` (line 97) | `compute_repo_name(repo_url, repo_type)` |
| `_compute_repo_path(req)` (line 107) | `compute_repo_path(repo_url, repo_type)` |
| `_parse_request_filters(req)` (line 112) | `parse_request_filters(excluded_dirs, excluded_files, included_dirs, included_files)` |
| `_LANGUAGE_NAMES` dict (line 126) | `LANGUAGE_NAMES` |
| `_language_name(code)` (line 140) | `language_name(code)` |
| `_send_event(ws, evt)` (line 212) | `send_event(ws, evt)` |
| `_send_tagged_event(ws, evt, **tags)` (line 216) | `send_tagged_event(ws, evt, **tags)` |
| `_safe_send(ws, evt)` (line 223) | `safe_send(ws, evt)` |
| `_safe_close(ws)` (line 230) | `safe_close(ws)` |
| `_consume_agent_loop(...)` (line 349) | `consume_agent_loop(...)` |

Refactor `wiki_generator.py` to import from `handler_utils` and delete the moved private helpers. Pure mechanical change — wiki tests must still pass with zero new failures.

### B2. Chat handler — `api/agent/chat_handler.py` (NEW, ~200 lines)

**Request model** (defined in this module, mirrors `wiki_generator.py`'s pattern of co-locating model + handler):

```python
class AgentChatRequest(BaseModel):
    repo_url: str
    type: str = "github"
    token: Optional[str] = None
    provider: str = "google"
    model: Optional[str] = None
    language: str = "en"
    messages: list[ChatMessage]      # reuse api.simple_chat.ChatMessage
    agent_name: str = "explore"      # default: read-only, 15-step budget
    excluded_dirs: Optional[str] = None
    excluded_files: Optional[str] = None
    included_dirs: Optional[str] = None
    included_files: Optional[str] = None
```

**Allow-list**: `_ALLOWED_CHAT_AGENTS = {"wiki", "explore", "deep-research"}`. Excludes `wiki-planner` / `wiki-writer` (they expect specific user-prompt shapes — file-tree dump for the planner, a page record for the writer).

**Core**: `_run_agent_chat(req, on_event)` — validates `agent_name`, clones repo via `download_repo` in `asyncio.to_thread`, parses filters, wraps tools, instantiates `UnifiedProvider`, converts legacy `{role, content}` messages to `AgentMessage` via `AgentMessage.from_chat_messages` (`api/agent/message.py:333`), validates last message is from user, then forwards every event from `run_agent_loop` to `on_event`. Defensive `FinishEvent` on exception paths.

**WebSocket handler**: `handle_agent_chat_websocket(ws)` — accepts, parses JSON, calls `_run_agent_chat` with `on_event = lambda evt: send_event(ws, evt)`. Untagged — chat is single-loop, no `phase` envelope needed (the wiki handler tags events with `phase=planning|writing` only because it runs two loops back-to-back).

**HTTP streaming handler**: `agent_chat_stream(request: AgentChatRequest)` — uses `asyncio.Queue(maxsize=64)` + producer task pattern. Yields `evt.model_dump_json() + "\n"`. Returns `StreamingResponse(..., media_type="application/x-ndjson")`.

**Optional `get_agent_info()`** — `GET /agent/info` returns `[{name, description, mode, max_steps, allowed_tools}, ...]`, filtered to `_ALLOWED_CHAT_AGENTS`. ~5 lines; unblocks future agent-picker UI.

### B3. Routes — `api/api.py`

Append after the existing agent-wiki registration (currently ends around line 405):

```python
from api.agent.chat_handler import (
    handle_agent_chat_websocket, agent_chat_stream, get_agent_info,
)
app.add_api_websocket_route("/ws/agent-chat", handle_agent_chat_websocket)
app.add_api_route("/chat/agent-stream", agent_chat_stream, methods=["POST"])
app.add_api_route("/agent/info", get_agent_info, methods=["GET"])
```

## Frontend changes (connectors only)

PLAN-005 owns the chat UI. This plan stops at "events arrive correctly typed."

### F1. Event types — `src/types/agentChat.ts` (NEW)

Discriminated union mirroring `api/agent/stream_events.py`. Skip wiki-only variants. Includes:
- `AgentChatRequest`
- `TextDeltaEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `FinishAgentEvent`, `ErrorAgentEvent`
- `AgentChatEvent` union

### F2. WebSocket client — extend `src/utils/websocketClient.ts`

Add `createAgentChatWebSocket(request, onEvent, onError, onClose)`. Note `onEvent` (not `onMessage`) — payload is structured JSON, not raw text. URL: `${SERVER_BASE_URL.replace(/^http/, "ws")}/ws/agent-chat`. `ws.onmessage` parses `event.data` as JSON and dispatches to `onEvent`.

### F3. HTTP client — `src/utils/agentChatStream.ts` (NEW)

`streamAgentChatHttp(request, onEvent, signal?)` — `fetch("/api/chat/agent-stream", {method: "POST", body: JSON.stringify(request)})`, reads body via `getReader()` + `TextDecoder`, line-buffered split on `\n`, parses each line, calls `onEvent`. Defensive partial-line drain on stream end.

### F4. Next.js proxy — `src/app/api/chat/agent-stream/route.ts` (NEW)

Clone of `src/app/api/chat/stream/route.ts:1-113`. Only changes:
- Target URL: `${TARGET_SERVER_BASE_URL}/chat/agent-stream`
- Forward `Accept: application/x-ndjson` (content-type pass-through preserves it for the client)

### F5. Out of scope (explicit)

- No edits to `src/app/[owner]/[repo]/ask/page.tsx`
- No new page route
- No new hooks, context providers, or UI components
- PLAN-005 wires up the chat view

## Streaming protocol

Wire is the JSON serialization of `api/agent/stream_events.py` events. No new event types. No envelope tags.

Example session for `"Where is the auth code?"`:

```ndjson
{"type":"text_delta","content":"I'll search for "}
{"type":"text_delta","content":"authentication-related files."}
{"type":"tool_call_start","tool_call_id":"call_a1b2c3","tool_name":"grep","tool_args":{"pattern":"authenticate","path":"."}}
{"type":"tool_call_end","tool_call_id":"call_a1b2c3","tool_name":"grep","result_summary":"src/auth/login.ts:12: function authenticate(...)\n... +14 matches","is_error":false,"duration_ms":47,"metadata":{"matches":15}}
{"type":"text_delta","content":"\n\nThe authentication entry point is "}
{"type":"text_delta","content":"`src/auth/login.ts`."}
{"type":"finish","finish_reason":"stop","usage":{"prompt_tokens":3201,"completion_tokens":48}}
```

Error path:

```ndjson
{"type":"error","error":"Unknown agent 'bogus'. Allowed: ['deep-research','explore','wiki']","code":"unknown_agent"}
{"type":"finish","finish_reason":"error"}
```

Server invariants (test-checkable):
1. Exactly one terminal `finish` event per session.
2. `error` may precede `finish`; on error, `finish.finish_reason === "error"`.
3. Every `tool_call_end` has a matching prior `tool_call_start` with the same `tool_call_id`.
4. `text_delta.content` may be `""` but never null.

Transport differences:
- WebSocket: each event is one `websocket.send_json(...)` call → one `MessageEvent` on the client.
- HTTP NDJSON: each event is `event.model_dump_json() + "\n"` → one line in the response body.

The shape of the events is identical across transports. A single `onEvent` handler can be shared between `createAgentChatWebSocket` and `streamAgentChatHttp`.

## Critical files to read before implementation

- `api/agent/wiki_generator.py` — handler template (lines 97-235 helpers, 522-580 main).
- `api/agent/loop.py` — confirm `run_agent_loop` signature at `:207` and termination semantics.
- `api/agent/stream_events.py` — confirm event union has not drifted.
- `api/agent/message.py:333` — `AgentMessage.from_chat_messages` legacy converter.
- `api/simple_chat.py:52` (`ChatMessage`) and `:732` (raw-text precedent — no SSE).
- `api/api.py:393-405` — route registration block.
- `src/app/api/chat/stream/route.ts` — proxy template to clone.

## Tests

| File | Cases |
|---|---|
| `tests/api/test_agent_chat_websocket.py` (NEW) | happy path; unknown `agent_name`; invalid payload; clone failure; provider init failure; empty messages; last-message-not-user |
| `tests/api/test_agent_chat_stream.py` (NEW) | same matrix, asserting NDJSON body parses to expected event sequence |

Mock pattern: monkeypatch `api.agent.chat_handler.run_agent_loop` to yield a synthetic event sequence (`TextDelta → ToolCallStart → ToolCallEnd → FinishEvent`). Use `TestClient.websocket_connect("/ws/agent-chat")` for the WS test and `client.post("/chat/agent-stream", ...)` for the HTTP test.

## Implementation steps

1. **Extract shared helpers (no behavior change).** Create `api/agent/handler_utils.py`. Refactor `wiki_generator.py` to import from it. Run `pytest tests/api/ -q --tb=short 2>&1 | tail -50` — must be green before continuing.
2. **Chat handler core.** Create `api/agent/chat_handler.py` with `AgentChatRequest`, `_ALLOWED_CHAT_AGENTS`, `_run_agent_chat`. Add a unit test driving `_run_agent_chat` with mocked `run_agent_loop`.
3. **WebSocket handler + route.** Add `handle_agent_chat_websocket`. Register at `api/api.py`. Add `tests/api/test_agent_chat_websocket.py`.
4. **HTTP streaming handler + route.** Add `agent_chat_stream` with the queue+producer pattern. Register. Add `tests/api/test_agent_chat_stream.py`.
5. **Agent info endpoint.** Add `get_agent_info`. Register.
6. **Frontend types.** `src/types/agentChat.ts`.
7. **Frontend WebSocket connector.** Extend `src/utils/websocketClient.ts`.
8. **Frontend HTTP connector.** `src/utils/agentChatStream.ts`.
9. **Frontend proxy.** `src/app/api/chat/agent-stream/route.ts`.
10. **End-to-end smoke.** Dev server up; Python `websockets` client + curl POST against both endpoints; `bun run lint`.

Steps 3 and 4 are independent (could parallelize). Frontend (6-9) waits on step 4 being green so the wire format is final.

## Verification

| Stage | Command |
|---|---|
| Refactor regression (step 1) | `pytest tests/api/ -q --tb=short 2>&1 \| tail -50` |
| New backend tests (steps 3-4) | `pytest tests/api/test_agent_chat_websocket.py tests/api/test_agent_chat_stream.py -q --tb=short 2>&1 \| tail -50` |
| Manual WebSocket | Python `websockets` client connecting to `ws://localhost:8001/ws/agent-chat` with a sample request |
| Manual HTTP | `curl -N -X POST http://localhost:8001/chat/agent-stream -H 'Content-Type: application/json' -d '{...}'` — every line must parse as JSON |
| Frontend lint | `bun run lint 2>&1 \| tail -30` |

## Design decisions (locked)

1. **HTTP wire format: NDJSON** (`application/x-ndjson`, one JSON object per `\n`). No SSE precedent in this repo; SSE framing buys nothing for structured events.
2. **`agent_name` default: `"explore"`.** Read-only tools, 15-step budget — safest + cheapest default. Most chat questions are lookup-shaped. `wiki` (full tools, 25 steps) and `deep-research` (full tools, 40 steps) are opt-in.
3. **Frontend scope: connectors only.** TS types + WS client + HTTP client + Next.js proxy. PLAN-005 owns the chat view.
4. **Conversation shape: legacy `{role, content}`.** Reuse `ChatMessage` from `api/simple_chat.py:52`. Richer `AgentMessage` parts structure stays internal.
5. **No `[DEEP RESEARCH]` tag interop.** `agent_name="deep-research"` is the explicit selector. Tag-based mode-switching in `simple_chat.py:152` is left untouched.
6. **No locking, no per-session timeout in v1.** `download_repo` is idempotent. Bound execution via `max_steps` in agent config + client disconnect. Revisit if production sees abuse.

## Sub-tasks

1. Extract shared helpers to `api/agent/handler_utils.py` and refactor `wiki_generator.py`.
2. Implement `AgentChatRequest` + `_run_agent_chat` core in `api/agent/chat_handler.py`.
3. Add `handle_agent_chat_websocket` + register `/ws/agent-chat`.
4. Add `agent_chat_stream` (NDJSON) + register `/chat/agent-stream`.
5. Add `get_agent_info` + register `GET /agent/info`.
6. Add tests `tests/api/test_agent_chat_websocket.py` and `tests/api/test_agent_chat_stream.py`.
7. Frontend types `src/types/agentChat.ts`.
8. Frontend WebSocket connector in `src/utils/websocketClient.ts`.
9. Frontend HTTP connector `src/utils/agentChatStream.ts`.
10. Frontend Next.js proxy `src/app/api/chat/agent-stream/route.ts`.
