---
number: PLAN-INDEX
name: Plans Index
description: Tracks implementation status for plans and their sub-tasks under handbooks/plans.
update_at: 2026-05-06
category: index
language: en
audience: developers-and-agents
---

# Plans Index

Read this index before opening individual plans when you need current implementation status.

## Status Legend

| Status | Meaning |
|---|---|
| `proposed` | Plan is documented, but implementation has not started or is not visible in current code. |
| `in-progress` | Implementation work has started, but the plan is not fully delivered. |
| `implemented` | Implementation appears complete against the plan's stated scope. |
| `blocked` | Implementation is paused on a known dependency or unresolved decision. |

## Plans

| Plan | Plan Status | Implementation Status | Evidence | Sub-task Status |
|---|---|---|---|---|
| [PLAN-001 - AST Code Splitter Improvement Plan](PLAN-001-ast-code-splitter.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `api/data_pipeline.py` still imports `TextSplitter` and `prepare_data_pipeline` still constructs `TextSplitter(**configs["text_splitter"])`. | No explicit sub-tasks recorded. |
| [PLAN-002 - OpsWiki Frontend Refinement Overview](PLAN-002-frontend-refinement-overview.md) | `proposed` | `in-progress` | PLAN-003 is implemented in `src_v2`; PLAN-004 through PLAN-006 are still not started, and the legacy `src/` frontend remains untouched per the overview non-goal. | Cross-cutting plan; sub-tasks live in PLAN-003 through PLAN-006. |
| [PLAN-003 - Foundation Design Tokens and App Shell](PLAN-003-foundation.md) | `implemented` | `implemented` | Plan frontmatter is `status: implemented`; `src_v2/app/globals.css` contains the Paper and Ink tokens; `src_v2/app/layout.tsx` mounts `LanguageProvider` and `SettingsProvider` without `ThemeProvider`; `src_v2/components/shell/` contains AppShell, Sidebar, Topbar, Composer, Wordmark, SettingsPanel, and small chrome primitives; shared renderers/connectors/types are copied into `src_v2`; `src_v2/messages/en.ts` centralizes shell copy; `bun run lint` and `bun run build` pass with existing warnings only. | Blocks PLAN-004, PLAN-005, PLAN-006; owns copying shared chat/agent-chat connector files into `src_v2`. |
| [PLAN-004 - Welcome and Projects Directory](PLAN-004-welcome-and-projects.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/page.tsx` is 627 lines rendering the current home; `src/app/wiki/projects/page.tsx` is the active projects route and `src/app/projects/` does not exist. Now carries a Tailwind-first styling rule shared with PLAN-005 and PLAN-006. | Depends on PLAN-003. |
| [PLAN-005 - Chat View](PLAN-005-chat-view.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/[owner]/[repo]/ask/page.tsx` (583 lines) still uses `AskComposer` and `AskResultView`; PLAN-007 connector files exist for the rewrite. Now carries a Tailwind-first styling rule shared with PLAN-004 and PLAN-006. | Depends on PLAN-003 and PLAN-007's implemented connector contract. |
| [PLAN-006 - Wiki Workshop Slides and Loading](PLAN-006-wiki-family.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/[owner]/[repo]/page.tsx` is 2267 lines and `src/components/WikiTreeView.tsx` is still the TOC; no `src/components/generation/` folder exists. Now carries a Tailwind-first styling rule shared with PLAN-004 and PLAN-005. | Depends on PLAN-003; keeps wiki-family generation on raw-text `/ws/chat`. |
| [PLAN-007 - Agent Chat Backend API](PLAN-007-agent-chat-api.md) | `implemented` | `implemented` | Plan frontmatter is `status: implemented`; `api/agent/chat_handler.py` implements the chat core; `/ws/agent-chat`, `/chat/agent-stream`, and `/agent/info` are registered in `api/api.py`; frontend connector files exist under `src/types/`, `src/utils/`, and `src/app/api/chat/agent-stream/`. | 10 sub-tasks implemented. |
| [PLAN-008 - RAG Retrieval as an Agent Tool](PLAN-008-rag-tool.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `api/retriever.py` and `api/tools/rag.py` do not exist; `_TOOL_CLASSES` in `api/tools/__init__.py` does not register `rag_search`. | 6 sub-tasks, none started. |

## Sub-tasks

| Plan | Sub-task | Status | Notes |
|---|---|---|---|
| PLAN-001 | None recorded | N/A | The plan is organized by solution options, implementation references, benefits, and risks rather than a task checklist. |
| PLAN-002 | None recorded | N/A | Overview plan; real tasks live in sub-plans PLAN-003 through PLAN-006. |
| PLAN-003 | Token rewrite of `globals.css` | `implemented` | `src_v2/app/globals.css` registers Paper and Ink tokens, global element defaults, wordmark styles, shell/composer chrome, SettingsPanel chrome, markdown styles, and the 1100px wiki responsive guard. |
| PLAN-003 | Build shared shell components under `src_v2/components/shell/` | `implemented` | Includes `<AppShell>`, `<Sidebar>`, `<Topbar>`, `<Composer>`, `<Wordmark>`, `<SettingsPanel>`, `<IconButton>`, `<Switcher>`, `<Eyebrow>`, and `<DayDivider>`. |
| PLAN-003 | Copy shared legacy chat and agent-chat connector files into `src_v2` | `implemented` | Preserves `createChatWebSocket`, `createAgentChatWebSocket`, `streamAgentChatHttp`, and `AgentChatEvent`/`AgentChatRequest` types with self-contained `src_v2` imports. |
| PLAN-003 | Introduce `SettingsContext` and `useConversationHistory` | `implemented` | `SettingsContext` persists `opswiki.*` settings; `useConversationHistory` powers the sidebar project tree and removes stale `deepwiki.*` localStorage keys on read. |
| PLAN-003 | Preserve legacy `src/` while adding `src_v2` foundation | `implemented` | Existing `src/` pages and components are untouched, matching PLAN-002 and PLAN-003 scope. |
| PLAN-004 | Rewrite `src/app/page.tsx` welcome route | `not-started` | Under 120 lines target. |
| PLAN-004 | Create `src/app/projects/page.tsx` and delete `src/app/wiki/projects/` | `not-started` | New `/projects` URL per PLAN-002 D8. |
| PLAN-005 | Rewrite `src/app/[owner]/[repo]/ask/page.tsx` | `not-started` | Under 200 lines target; use `AgentChatRequest` with default `agent_name: "explore"`. |
| PLAN-005 | Build `<ChatStream>`, `<Message>`, `<Citation>`, and `<ToolEvent>` under `src_v2/components/chat/` | `not-started` | Reuses `<Markdown>` and renders structured `tool_call_start` / `tool_call_end` events. |
| PLAN-005 | Wire WebSocket + HTTP agent-chat transports | `not-started` | Primary `/ws/agent-chat`; fallback `/api/chat/agent-stream`; share one `AgentChatEvent` reducer. |
| PLAN-006 | Rewrite `src/app/[owner]/[repo]/page.tsx` wiki route | `not-started` | Under 250 lines target. |
| PLAN-006 | Rewrite `src/app/[owner]/[repo]/workshop/page.tsx` | `not-started` | Under 200 lines target. |
| PLAN-006 | Rewrite `src/app/[owner]/[repo]/slides/page.tsx` | `not-started` | Under 200 lines target. |
| PLAN-006 | Build `<GenerationLoader>` and `useGenerationPhases` hook | `not-started` | Implements the loading screen with best-effort phase parsing over raw `/ws/chat` text, not `AgentChatEvent`. |
| PLAN-007 | Extract shared helpers to `api/agent/handler_utils.py` | `implemented` | `api/agent/wiki_generator.py` now imports shared repo/filter/language/send/loop helpers. |
| PLAN-007 | Implement `AgentChatRequest` + `_run_agent_chat` core | `implemented` | `api/agent/chat_handler.py` validates agent/messages, clones the repo, wraps tools, and streams `run_agent_loop` events. |
| PLAN-007 | Add `handle_agent_chat_websocket` + register `/ws/agent-chat` | `implemented` | Untagged events are sent through `send_event`; route registered in `api/api.py`. |
| PLAN-007 | Add `agent_chat_stream` (NDJSON) + register `/chat/agent-stream` | `implemented` | Streaming response uses `application/x-ndjson`; route registered in `api/api.py`. |
| PLAN-007 | Add `get_agent_info` + register `GET /agent/info` | `implemented` | Returns only `wiki`, `explore`, and `deep-research` agent metadata. |
| PLAN-007 | Add WebSocket + HTTP test files | `implemented` | `tests/api/test_agent_chat_websocket.py` and `tests/api/test_agent_chat_stream.py` cover happy and error paths. |
| PLAN-007 | Frontend types `src/types/agentChat.ts` | `implemented` | Discriminated union covers chat-relevant agent events and request payloads. |
| PLAN-007 | Extend `src/utils/websocketClient.ts` with `createAgentChatWebSocket` | `implemented` | Uses structured `onEvent` callback against `/ws/agent-chat`. |
| PLAN-007 | Frontend HTTP connector `src/utils/agentChatStream.ts` | `implemented` | Reads line-buffered NDJSON from `/api/chat/agent-stream`. |
| PLAN-007 | Frontend Next.js proxy `src/app/api/chat/agent-stream/route.ts` | `implemented` | Proxies to backend `/chat/agent-stream` with `Accept: application/x-ndjson`. |
| PLAN-008 | Implement `api/retriever.py` (`CodeRetriever`, LRU cache with per-key lock) | `not-started` | Pure retrieval component; reuses `DatabaseManager` for the `.pkl` disk cache. |
| PLAN-008 | Implement `api/tools/rag.py` and `api/tools/rag.txt` | `not-started` | Thin `Tool`-ABC adapter calling `get_or_build_retriever(repo_path)`. |
| PLAN-008 | Register `rag_search` in tool registry and agent configs | `not-started` | Add to `_TOOL_CLASSES`, `_ALL_TOOLS`, and `_READ_ONLY_TOOLS`. |
| PLAN-008 | Post-filter `rag_search` results in `FilteredToolWrapper` | `not-started` | Drop chunks whose `meta_data.file_path` is excluded. |
| PLAN-008 | Add unit tests for retriever and tool | `not-started` | `tests/unit/test_retriever.py`, `tests/unit/test_rag_tool.py`, `tests/api/test_agent_chat_rag_tool.py`. |
| PLAN-008 | Update handbooks indexes to list PLAN-008 | `implemented` | This entry in `handbooks/plans/index.md`/`index.json` and `handbooks/index.md`. |
