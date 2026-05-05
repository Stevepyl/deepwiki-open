---
number: PLAN-005
name: Chat View
description: Rewrites the Ask route into a chat-stream UI that matches the prototype, using the structured agent-chat connectors and shared Markdown renderer.
update_at: 2026-05-06
category: improvement-plan
language: en
status: proposed
---

# PLAN-005 Chat View

## Context

The current chat surface is `src_old/app/[owner]/[repo]/ask/page.tsx` (583 lines) plus `src_old/components/AskComposer.tsx` and `src_old/components/AskResultView.tsx`. It implements a single-question-plus-answer model with deep-research toggle and iteration traces.

The prototype reframes this as a conversation: alternating user and AI messages, citations as pill badges, day dividers between sessions, a floating composer that shares the shell with the wiki view. The interaction model moves from "ask once, render once" to "multi-turn with local history".

Backend protocol is already extended by PLAN-007. The current `src_old/` Ask page still uses legacy `WebSocket /ws/chat` plus `POST /api/chat/stream` raw-text fallback, but the `src` Ask rewrite should use structured agent chat:

- Primary transport: `createAgentChatWebSocket()` → direct `WebSocket /ws/agent-chat`.
- HTTP fallback: `streamAgentChatHttp()` → same-origin `POST /api/chat/agent-stream` → backend `POST /chat/agent-stream`.
- Request shape: `AgentChatRequest`, still using legacy `{ role, content }` messages plus `agent_name`.
- Response shape: `AgentChatEvent` JSON objects, where `text_delta` appends assistant text and `tool_call_start` / `tool_call_end` drive the reasoning/tool timeline.

Use `agent_name: "explore"` by default. Map the deep-research mode to `agent_name: "deep-research"` instead of the old `[DEEP RESEARCH]` prompt tag. The `wiki` agent remains available as an explicit advanced option, not the default chat mode.

All new code is written to `src/`. The existing `src_old/` is left untouched.

## Styling rule — Tailwind-first

Write all new markup with Tailwind utility classes referencing the Paper and Ink tokens registered by PLAN-003's `@theme` block in `src/app/globals.css`. Example: `className="bg-[var(--paper)] text-[var(--ink-primary)] max-w-[760px] mx-auto"`.

Fall back to prototype CSS class names (e.g. `.chat-stream`, `.message`, `.citation`, `.tool-event`) only when Tailwind cannot express the rule cleanly — paper-grain pseudo-elements, multi-line citation pill layouts, or the running-tool shimmer animation. When you do fall back, colocate the rule in `globals.css` under a clearly labeled section; do not create per-component CSS files.

Never introduce new dark-mode selectors or `data-theme` branches. Tokens drive the palette.

## Target file

`src/app/[owner]/[repo]/ask/page.tsx` — new file, target under 200 lines.

## Route parameters

- URL path: `/[owner]/[repo]/ask`.
- Optional query params:
  - `?q=<prefilled question>` — consumed once on mount to auto-submit the first turn. This is how the wiki/workshop composer redirects in from PLAN-006.
  - `?convId=<uuid>` — when present, loads the conversation from `useConversationHistory()` and renders its message history. When absent, starts a new conversation (fresh UUID).

## Page structure

- Wrapped in `<AppShell>` from PLAN-003.
- Topbar:
  - Breadcrumb: `<owner>/<repo>` (mono) `/` `<current conv title>` (serif italic).
  - Switcher: Chat (active) · Wiki · Workshop (3-tab).
  - Actions: Slides icon link, Share, More.
- Main content:
  - Centered scrollable column (max-width 760px, per `prototype/styles.css:732`).
  - Renders `<DayDivider>` between messages whose timestamps cross a calendar day.
  - `<Message>` for each turn.
- Composer (shared `<Composer variant="chat">`): submit creates a new user message, fires an agent-chat WS request, then appends an AI message that fills from `text_delta` events.

## Components to build

- `src/components/chat/ChatStream.tsx` (~80 lines) — renders the ordered list of messages with day dividers. Pure view; no state.
- `src/components/chat/Message.tsx` (~60 lines) — one message bubble. Props: `role`, `content`, `timestamp`, `model?`, `citations?`. Uses `<Markdown>` from `src/components/Markdown.tsx` to render `content`.
- `src/components/chat/Citation.tsx` (~20 lines) — pill-shaped citation link. Hover reveals the file path.
- `src/components/chat/ToolEvent.tsx` (~60 lines) — compact running/completed/error row for `tool_call_start` and `tool_call_end` events. Collapsed by default; shows tool name, duration, error state, and result summary.

## State management

State lives in the page component via `useState` + `useEffect`. No Redux/Zustand. The flow per turn:

1. User hits `↵` in `<Composer>`.
2. Page constructs an `AgentChatRequest` from:
   - `repo_url` — computed with `getRepoUrl(repoInfo)` from `src/utils/getRepoUrl.tsx`.
   - `messages` — all prior turns plus the new user message.
   - `provider`, `model`, `token`, `excluded_*`, `included_*` — from `useSettings()` (PLAN-003 `src/contexts/SettingsContext.tsx`).
   - `language` — from `LanguageContext` (not user-toggleable in v1; defaults to "en").
   - `agent_name` — `"explore"` by default; `"deep-research"` when the user enables deep research.
3. Opens a socket via `createAgentChatWebSocket()` from `src/utils/websocketClient.ts` (copied in PLAN-003).
4. Appends an empty AI message with `streaming: true`. As `AgentChatEvent` objects arrive:
   - `text_delta`: append `content` to the assistant message.
   - `tool_call_start`: append or update a running tool event keyed by `tool_call_id`.
   - `tool_call_end`: mark the matching tool event complete and store `result_summary`, `duration_ms`, and `is_error`.
   - `error`: append an error state to the assistant message and stop accepting more content for that turn.
   - `finish`: mark the assistant message complete and close the stream state.
5. On socket close:
   - Parse the accumulated text for a `## Citations` tail (prototype shows inline citation pills; extract them with a regex). Store on the message as `citations: string[]`.
   - Mark message complete.
   - Call `useConversationHistory().addConversation(repoKey, ...)` or update the existing entry.
6. On failure/timeout before a terminal `finish`: fall back to `streamAgentChatHttp()` and continue feeding events through the same reducer.

## Citation parsing

Citations in the prototype look like:

```
<span class="citation"><span class="citation__num">1</span>src_old/hooks/pretool.ts:42</span>
```

The backend does not emit these as structured data. For v1, parse citations from a trailing `\n\nSources:\n- path/to/file:line\n- ...` block if present. If the model does not emit that block, render no citations. A prompt-template change to reliably elicit the block is out of scope here.

## Persistence

`useConversationHistory()` persists to `localStorage`. Write shape:

```ts
type ConvEntry = {
  id: string;
  repoKey: string;       // `${type}:${owner}/${repo}`
  title: string;         // first user message, truncated
  messages: {
    role: "user" | "assistant";
    content: string;
    timestamp: number;
    citations?: string[];
  }[];
  lastMessageAt: number;
  model?: string;
};
```

Quota guard: if `localStorage` is >4MB, evict oldest conversations before writing.

## Components to delete after this plan

None — `src_old/` is left untouched. `src_old/components/AskComposer.tsx` and `src_old/components/AskResultView.tsx` remain in `src_old/` as-is.

## Critical files referenced or modified

- `prototype/app-chat.html` — DOM source of truth
- `prototype/styles.css:727-890` — chat-stream styles
- `src/app/[owner]/[repo]/ask/page.tsx` — new
- `src/components/chat/*` — new folder
- `src/components/Markdown.tsx` — copied from `src_old/` in PLAN-003
- `src/utils/websocketClient.ts` — copied from `src_old/` in PLAN-003
- `src/utils/agentChatStream.ts` — copied from `src_old/` in PLAN-003
- `src/types/agentChat.ts` — copied from `src_old/` in PLAN-003
- `src/utils/getRepoUrl.tsx` — copied from `src_old/` in PLAN-003
- `src/hooks/useConversationHistory.ts` — introduced in PLAN-003
- `src/contexts/SettingsContext.tsx` — introduced in PLAN-003
- `docs/api/frontend-backend-apis.md` §5.3 (`/ws/agent-chat`), §5.4 (`/api/chat/agent-stream` fallback) — primary Ask rewrite contract
- `handbooks/plans/PLAN-007-agent-chat-api.md` — implemented backend and connector plan

## Verification

In addition to the PLAN-002 shared harness:

1. Open `src/` `/[owner]/[repo]/ask` on a cached repository. Ask a question. The AI message populates token-by-token from `text_delta` events.
2. Refresh the page with the same `?convId`. The prior messages re-render from `localStorage`.
3. Open DevTools → Network → WS frame inspector. Confirm the payload matches `AgentChatRequest`, includes `agent_name`, and contains the full `messages` history, not just the latest question.
4. Mock or trigger one `tool_call_start` / `tool_call_end` pair. The tool event appears in the message timeline, transitions from running to complete/error, and does not pollute the assistant markdown body.
5. Simulate WS failure before `finish` (stop the backend mid-request). The client retries over `POST /api/chat/agent-stream` and continues streaming NDJSON events into the same reducer.
6. Ask three questions that span midnight local time. A `<DayDivider>` appears between the messages that crossed the boundary.
7. The sidebar `<ProjectTree>` in `<AppShell>` updates immediately when a new conversation is started (shared hook, not a page-local store).
8. A question with `?q=explain+hooks` in the URL auto-submits on mount.
9. Visual parity with `prototype/app-chat.html` at 1440×900.
10. `grep -r "from.*src_old/" src/` returns no hits.

## Risks

- **Citation parsing fragility.** Agent chat events are structured, but citations are still model-output text. Accept that v1 shows no citations when the model deviates; do not let regex failures crash the render.
- **Mixed protocol confusion.** Legacy `/ws/chat` streams raw text; agent chat streams structured events. Keep separate reducers and do not send an `AgentChatRequest` to `/ws/chat` or parse `/ws/agent-chat` as text.
- **localStorage quota.** A long conversation with embedded code can consume MBs. The eviction guard is best-effort; add a "Clear history" action under Settings as a follow-up.
- **WebSocket reconnect storms.** If a user types quickly after an error, the fallback could double-fire. Debounce the submit and disable the composer while `streaming: true`.
- **Multi-tab interference.** Two tabs writing to `localStorage.opswiki.conversationHistory` can clobber each other. Use a `storage` event listener on the hook to refresh state if another tab writes; accept last-writer-wins for the messages array.
