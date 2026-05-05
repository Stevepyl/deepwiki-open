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
| [PLAN-002 - OpsWiki Frontend Refinement Overview](PLAN-002-frontend-refinement-overview.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/layout.tsx` still loads `ThemeProvider`; `src/app/page.tsx` still renders the DeepWiki welcome; no `src/components/shell/` folder exists. | Cross-cutting plan; sub-tasks live in PLAN-003 through PLAN-006. |
| [PLAN-003 - Foundation Design Tokens and App Shell](PLAN-003-foundation.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/globals.css` still carries dark-mode tokens; `src/components/ConfigurationModal.tsx` and `src/components/theme-toggle.tsx` are still present. | Blocks PLAN-004, PLAN-005, PLAN-006. |
| [PLAN-004 - Welcome and Projects Directory](PLAN-004-welcome-and-projects.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/page.tsx` is 627 lines rendering the current home; `src/app/wiki/projects/page.tsx` is the active projects route and `src/app/projects/` does not exist. | Depends on PLAN-003. |
| [PLAN-005 - Chat View](PLAN-005-chat-view.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/[owner]/[repo]/ask/page.tsx` (583 lines) still uses `AskComposer` and `AskResultView`. | Depends on PLAN-003. |
| [PLAN-006 - Wiki Workshop Slides and Loading](PLAN-006-wiki-family.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `src/app/[owner]/[repo]/page.tsx` is 2267 lines and `src/components/WikiTreeView.tsx` is still the TOC; no `src/components/generation/` folder exists. | Depends on PLAN-003. |
| [PLAN-007 - Agent Chat Backend API](PLAN-007-agent-chat-api.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; no `api/agent/chat_handler.py` exists; `/ws/agent-chat` route is not registered in `api/api.py`. | 10 sub-tasks (helper extraction, chat handler, WS+HTTP routes, agent info, tests, 4 frontend files). |

## Sub-tasks

| Plan | Sub-task | Status | Notes |
|---|---|---|---|
| PLAN-001 | None recorded | N/A | The plan is organized by solution options, implementation references, benefits, and risks rather than a task checklist. |
| PLAN-002 | None recorded | N/A | Overview plan; real tasks live in sub-plans PLAN-003 through PLAN-006. |
| PLAN-003 | Token rewrite of `globals.css` | `not-started` | Plan step 1; blocks every other shell task. |
| PLAN-003 | Build shared shell components under `src/components/shell/` | `not-started` | Plan step 2; includes `<AppShell>`, `<Sidebar>`, `<Topbar>`, `<Composer>`, `<Wordmark>`, `<SettingsPanel>`. |
| PLAN-003 | Introduce `SettingsContext` and `useConversationHistory` | `not-started` | Plan step 3; feeds PLAN-005 and PLAN-006. |
| PLAN-003 | Delete orphaned components (theme-toggle, ConfigurationModal, etc.) | `not-started` | Final commit of the plan. |
| PLAN-004 | Rewrite `src/app/page.tsx` welcome route | `not-started` | Under 120 lines target. |
| PLAN-004 | Create `src/app/projects/page.tsx` and delete `src/app/wiki/projects/` | `not-started` | New `/projects` URL per PLAN-002 D8. |
| PLAN-005 | Rewrite `src/app/[owner]/[repo]/ask/page.tsx` | `not-started` | Under 200 lines target. |
| PLAN-005 | Build `<ChatStream>`, `<Message>`, `<Citation>` under `src/components/chat/` | `not-started` | Reuses `<Markdown>`. |
| PLAN-006 | Rewrite `src/app/[owner]/[repo]/page.tsx` wiki route | `not-started` | Under 250 lines target. |
| PLAN-006 | Rewrite `src/app/[owner]/[repo]/workshop/page.tsx` | `not-started` | Under 200 lines target. |
| PLAN-006 | Rewrite `src/app/[owner]/[repo]/slides/page.tsx` | `not-started` | Under 200 lines target. |
| PLAN-006 | Build `<GenerationLoader>` and `useGenerationPhases` hook | `not-started` | Implements the loading screen with best-effort phase parsing. |
| PLAN-007 | Extract shared helpers to `api/agent/handler_utils.py` | `not-started` | Mechanical refactor of `api/agent/wiki_generator.py:97-234`. Blocks all subsequent steps. |
| PLAN-007 | Implement `AgentChatRequest` + `_run_agent_chat` core | `not-started` | New module `api/agent/chat_handler.py`. |
| PLAN-007 | Add `handle_agent_chat_websocket` + register `/ws/agent-chat` | `not-started` | Untagged events. |
| PLAN-007 | Add `agent_chat_stream` (NDJSON) + register `/chat/agent-stream` | `not-started` | `application/x-ndjson` media type. |
| PLAN-007 | Add `get_agent_info` + register `GET /agent/info` | `not-started` | Filtered to `_ALLOWED_CHAT_AGENTS`. |
| PLAN-007 | Add WebSocket + HTTP test files | `not-started` | `tests/api/test_agent_chat_websocket.py`, `tests/api/test_agent_chat_stream.py`. |
| PLAN-007 | Frontend types `src/types/agentChat.ts` | `not-started` | Discriminated union for all 5 chat-relevant events. |
| PLAN-007 | Extend `src/utils/websocketClient.ts` with `createAgentChatWebSocket` | `not-started` | `onEvent` callback (not `onMessage`). |
| PLAN-007 | Frontend HTTP connector `src/utils/agentChatStream.ts` | `not-started` | Line-buffered NDJSON parsing. |
| PLAN-007 | Frontend Next.js proxy `src/app/api/chat/agent-stream/route.ts` | `not-started` | Clone of existing chat proxy with target URL change. |
