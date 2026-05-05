---
number: PLAN-002
name: OpsWiki Frontend Refinement Overview
description: Cross-cutting plan that ties together the four sub-plans rebranding the frontend to the OpsWiki Paper and Ink prototype while preserving the FastAPI chat, agent-chat, and wiki contracts.
update_at: 2026-05-06
category: improvement-plan
language: en
status: proposed
---

# OpsWiki Frontend Refinement — Overview

## Context

The current frontend in `src_old/` ships a Japanese-dark-aesthetic "DeepWiki-Open" UI, accumulated over multiple feature additions. The result is heavy: `src_old/app/page.tsx` and `src_old/app/[owner]/[repo]/page.tsx` are each over two thousand lines, four configuration modals (`ConfigurationModal`, `ModelSelectionModal`, `WikiTypeSelector`, `UserSelector`) compete for attention on the home page, and the core reading experience is buried under chrome.

The static prototype under `prototype/` proposes a rebrand to **OpsWiki** with a single visual direction — "Paper and Ink", warm cream surfaces with rust orange accent, serif-led hierarchy, light mode only. Seven HTML files (`prototype/index.html`, `prototype/app-{chat,wiki,workshop,projects,slides,loading}.html`) demonstrate the full app surface against `prototype/styles.css` design tokens.

This plan family builds the new frontend in `src/` — a clean directory that is independent of the existing `src_old/`. The existing `src_old/` is left untouched. Production-grade pieces from `src_old/` that are worth reusing (Markdown renderer, Mermaid renderer, WebSocket client, type definitions, utility helpers) are **copied** into `src/` rather than imported across directory boundaries.

Contract update from PLAN-007: the backend now exposes structured agent chat through `WebSocket /ws/agent-chat` and `POST /chat/agent-stream`, and `src_old/` contains connector files for that contract (`src_old/types/agentChat.ts`, `src_old/utils/agentChatStream.ts`, `src_old/app/api/chat/agent-stream/route.ts`, plus `createAgentChatWebSocket` in `src_old/utils/websocketClient.ts`). The current `src_old/app/[owner]/[repo]/ask/page.tsx` still uses the legacy raw-text `/ws/chat` path, but the `src` rewrite must preserve these new connectors and PLAN-005 should wire the redesigned Ask UI to them.

The work is split across four sub-plans (PLAN-003 through PLAN-006). This document is the index, the cross-cutting decision log, and the verification harness.

## Goals

- Match the visual language of `prototype/` pixel-for-pixel on a 1440×900 viewport for the seven mapped routes.
- Reduce the line count of every page module to fit in one screen of context (target: each `page.tsx` under 250 lines).
- Keep the FastAPI request/response contract unchanged. No backend edits in this plan family; consume the already-implemented legacy chat, agent-chat, cache, auth, and project endpoints as documented.
- Reuse shared rendering primitives and connectors (Markdown, Mermaid, legacy WebSocket helper, agent-chat WebSocket/HTTP helpers, type definitions) — copy them into `src/` so the new frontend is self-contained.
- Drop dark mode and the wiki-type toggle (concise vs comprehensive); accept ADR-001's recommendation in code.
- Leave `src_old/` entirely untouched. All new work lives in `src/`.

## Non-goals

- Backend changes: FastAPI routes, WebSocket event shapes, cache file layout. PLAN-007's existing agent-chat endpoints are consumed, not modified.
- A new authentication system. The existing auth-code gate is preserved and wired into the topbar settings panel only.
- A `/api/conversations` endpoint. Sidebar conversation history is `localStorage`-only in v1.
- `next/font` migration. Plans use `@import url(...)` from Google Fonts to match `prototype/styles.css` exactly; a font-loading optimization pass is deferred.
- Removing `LanguageContext`. The plumbing stays; no UI affordance is added.
- Command palette beyond focusing the sidebar search input on `⌘K`.
- Any modification to `src_old/`. The old frontend is preserved as-is.

## Page map

| # | Prototype file | Target route | Backend endpoints | Sub-plan |
|---|---|---|---|---|
| 1 | `prototype/index.html` | `/` | `GET /api/auth/status`, `GET /api/models/config` (lazy) | PLAN-004 |
| 2 | `prototype/app-projects.html` | `/projects` (renamed from `/wiki/projects`) | `GET /api/wiki/projects`, `DELETE /api/wiki/projects` | PLAN-004 |
| 3 | `prototype/app-loading.html` | `/[owner]/[repo]?status=generating` | `WebSocket /ws/chat` (text stream best-effort parsing) | PLAN-006 |
| 4 | `prototype/app-chat.html` | `/[owner]/[repo]/ask` | `WebSocket /ws/agent-chat`, `POST /api/chat/agent-stream` (fallback); legacy `/ws/chat` remains for old `src_old/` | PLAN-005 |
| 5 | `prototype/app-wiki.html` | `/[owner]/[repo]` | `GET /api/wiki_cache`, `POST /api/wiki_cache`, `DELETE /api/wiki_cache`, `WebSocket /ws/chat` | PLAN-006 |
| 6 | `prototype/app-workshop.html` | `/[owner]/[repo]/workshop` | `GET /api/wiki_cache`, `WebSocket /ws/chat` | PLAN-006 |
| 7 | `prototype/app-slides.html` | `/[owner]/[repo]/slides` | `GET /api/wiki_cache`, `WebSocket /ws/chat` | PLAN-006 |

The chat/wiki/workshop topbar exposes a 3-tab switcher and a "Present as slides" icon button. The sidebar and floating composer are pixel-identical across those three views — they live in shared shell components built in PLAN-003.

## Cross-cutting decisions

These apply across every sub-plan; they are not re-stated in each one.

### D1. Brand wordmark
The product is rendered as **OpsWiki** with italic/roman contrast: `<em>Ops</em>Wiki`. A `<Wordmark size="hero" | "sidebar">` component lives in `src/components/shell/Wordmark.tsx`. The `package.json` `name` field, repository name, and backend identifier are not changed — only the user-visible UI string.

### D2. Light mode only
`src/app/globals.css` is written with the prototype's `:root` tokens, Tailwind v4 `@theme` registration, and base element defaults only — no `data-theme="dark"` selectors. There is no `theme-toggle` component in `src/`. Component styling is Tailwind-first in `src` components, and CSS variables match `prototype/styles.css:7-47` exactly so a side-by-side comparison is meaningful.

### D3. Configuration is hidden behind the topbar
There is no on-landing configuration modal. The settings icon (gear) on the topbar opens a slide-in `<SettingsPanel>` that exposes:
- Provider + model selection (reads `GET /api/models/config`)
- Authorization code (reads `GET /api/auth/status`, posts `POST /api/auth/validate`)
- Repository token (for private repos)
- Excluded/included dirs and files
- Language (reuses `LanguageContext`)

The `<SettingsPanel>` is one component, shared by every route that needs configuration. It is built in PLAN-003.

### D4. Wiki type toggle dropped
Per ADR-001, the concise/comprehensive selector is removed. The backend cache key is unaware of it (see `docs/api/frontend-backend-apis.md` §7.4). The wiki always generates in the configured comprehensive mode.

### D5. Conversation history is local
The sidebar's collapsible "project → conversations" tree reads from `localStorage` keyed by `repoKey = ${type}:${owner}/${repo}`. There is no backend store. Items are written when an Ask page submits a question, and a delete affordance is available on hover. A future `/api/conversations` endpoint can replace this without changing the UI.

### D6. Loading screen progress is best-effort
The prototype's phase log (Clone → Chunk → Embed → Generate) does not have a matching backend event stream. The loading screen subscribes to the existing `/ws/chat` text stream via `websocketClient.ts` and runs a permissive line parser:
- Lines starting with known prefixes (`Clone`, `Chunking`, `Embedding`, `Generating page`) advance the phase pipeline.
- Any other text is appended to the log feed verbatim.
- If the stream produces no recognizable phase, the screen shows "Generating…" with the spinner and an indeterminate hairline progress bar.

If reliable phase reporting is needed later, the WebSocket protocol can be extended (out of scope here).

### D7. Wiki composer redirects to Ask
The floating composer on `/[owner]/[repo]` and `/[owner]/[repo]/workshop` redirects to `/[owner]/[repo]/ask?q=<encoded>` on submit. There is no inline-answer drawer. This keeps the wiki/workshop pages reading-focused and avoids a second WebSocket lifecycle per page.

### D8. Projects route URL
The route moves from `/wiki/projects` to `/projects`. The old path is removed. There is no redirect — this is a rebrand, not a migration of a public URL.

### D9. Component reuse vs replacement

**Copy into `src/` and reuse:**
- `src_old/components/Markdown.tsx` → `src/components/Markdown.tsx`
- `src_old/components/Mermaid.tsx` → `src/components/Mermaid.tsx`
- `src_old/utils/websocketClient.ts` → `src/utils/websocketClient.ts`
- `src_old/utils/agentChatStream.ts` → `src/utils/agentChatStream.ts`
- `src_old/utils/getRepoUrl.tsx` → `src/utils/getRepoUrl.tsx`
- `src_old/utils/urlDecoder.tsx` → `src/utils/urlDecoder.tsx`
- `src_old/hooks/useProcessedProjects.ts` → `src/hooks/useProcessedProjects.ts`
- `src_old/contexts/LanguageContext.tsx` → `src/contexts/LanguageContext.tsx`
- `src_old/types/**` → `src/types/**`

All imports within `src/` reference `src/` paths only. No cross-directory imports between `src_old/` and `src/`.

**Build new in `src/` (no equivalent in `src_old/`):**
- `src/app/page.tsx` (home)
- `src/app/[owner]/[repo]/page.tsx` (wiki)
- `src/app/[owner]/[repo]/ask/page.tsx`
- `src/app/[owner]/[repo]/slides/page.tsx`
- `src/app/[owner]/[repo]/workshop/page.tsx`
- `src/app/projects/page.tsx`
- `src/components/shell/*`
- `src/components/chat/*`
- `src/components/wiki/*`
- `src/components/workshop/*`
- `src/components/slides/*`
- `src/components/generation/*`
- `src/components/welcome/*`
- `src/components/projects/*`
- `src/contexts/SettingsContext.tsx`
- `src/hooks/useConversationHistory.ts`

### D10. Agent chat is the Ask rewrite contract

PLAN-005 should treat PLAN-007's structured agent chat as the primary backend for the redesigned Ask route:
- `createAgentChatWebSocket()` opens direct `WebSocket /ws/agent-chat`.
- `streamAgentChatHttp()` calls same-origin `POST /api/chat/agent-stream`, which proxies to backend `POST /chat/agent-stream`.
- `text_delta` fills assistant content; `tool_call_start` / `tool_call_end` feed tool trace UI; `finish` completes the turn; `error` renders a structured failure state.
- `agent_name` defaults to `explore`; deep-research mode maps to `agent_name: "deep-research"`.

The wiki, workshop, slides, and loading routes still use the legacy raw-text `/ws/chat` generation path unless a later plan explicitly migrates them to `/ws/agent-wiki`.

## Sub-plan graph

```
PLAN-003 (foundation: tokens + shared shell)
   |
   +--> PLAN-004 (welcome + projects)
   |
   +--> PLAN-005 (chat)
   |
   +--> PLAN-006 (wiki + workshop + slides + loading)
```

PLAN-003 must merge before any other sub-plan can land — every other plan imports from `src/components/shell/`. PLAN-004, PLAN-005, and PLAN-006 are independent of each other and can be picked up in parallel.

## Critical files

- `prototype/styles.css:7-47` — the canonical token set
- `prototype/*.html` — the seven target screens
- `docs/api/frontend-backend-apis.md` — the unchanged backend contract
- `handbooks/plans/PLAN-007-agent-chat-api.md` — implemented agent-chat connector contract used by PLAN-005
- `src/utils/agentChatStream.ts`, `src/types/agentChat.ts`, `src/app/api/chat/agent-stream/route.ts` — preserve the PLAN-007 frontend connector surface
- `src/app/layout.tsx` — root layout (no `ThemeProvider`, has `LanguageProvider` and `SettingsProvider`)
- `src/app/globals.css` — Tailwind import, Paper and Ink tokens, base defaults only
- `src/components/shell/*` — Tailwind-first shell, composer, sidebar, topbar, settings, and small chrome primitives
- `next.config.ts` — rewrites are unchanged (still proxies to Python backend)
- `tailwind.config.js` — token registration only

## Verification

This block is the harness that PLAN-003 through PLAN-006 reuse. Each sub-plan adds page-specific checks on top.

1. `bun run lint` exits clean.
2. `bun run build` exits clean.
3. Run the backend (`uv run -m api.main`) and frontend (`bun dev`).
4. Walk the seven routes from the page map. For each, open the matching `prototype/*.html` in another tab at the same viewport and visually compare:
   - Type stack, hairline color, paper hierarchy
   - Composer anchor (it must not move between Chat and Wiki)
   - Active-state accent bar on the sidebar
5. In the Network tab, confirm only documented endpoints are hit. No 4xx or 5xx on the golden path.
6. `grep -r "data-theme" src_old/` returns nothing functional. `grep -r "dark:" src_old/` returns nothing functional.
7. `grep -r "DeepWiki" src_old/` returns no UI strings (project identifiers in CLAUDE.md and tests are fine).
8. Resize to 1100px width: `prototype/`'s responsive guard at `prototype/styles.css:1350-1358` collapses the wiki TOC. Confirm the same behavior in the rebuilt wiki page.

## Risks

- **Markdown component coupling**: `src_old/components/Markdown.tsx` may carry assumptions about the dark-mode color palette through CSS variables. PLAN-003 must audit and rewire the syntax-highlighting tokens before PLAN-005/006 land.
- **WebSocket reconnect on route change**: the current `websocketClient.ts` opens a new socket per request. The redirect from wiki → ask in D7 should pass the question via URL state, not via a held-open socket.
- **Agent connector omission**: if `src` copies only the legacy `websocketClient.ts` path and omits `agentChatStream.ts` or `agentChat.ts`, PLAN-005 will silently fall back to raw-text chat and lose tool/reasoning events. Treat the PLAN-007 connector files as part of the shared frontend contract.
- **Cache-invalidation drift**: removing the wiki-type toggle (D4) is safe because the backend cache key already ignores it, but in-flight `localStorage` entries written by the old UI may still reference it. PLAN-003 includes a one-time cleanup on app load.
- **i18n strings**: prototype copy is English-only. If `LanguageContext` is left active, an unchecked language switch could surface untranslated strings. PLAN-003 wraps all new copy in a single `messages/en.ts` and ignores `LanguageContext` for v1; the plumbing remains for a follow-up.
