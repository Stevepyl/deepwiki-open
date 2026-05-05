---
number: PLAN-005
name: Chat View
description: Rewrites the Ask route into a chat-stream UI that matches the prototype, reusing the WebSocket client and Markdown renderer.
update_at: 2026-05-06
category: improvement-plan
language: en
status: proposed
---

# PLAN-005 Chat View

## Context

The current chat surface is `src/app/[owner]/[repo]/ask/page.tsx` (583 lines) plus `src/components/AskComposer.tsx` and `src/components/AskResultView.tsx`. It implements a single-question-plus-answer model with deep-research toggle and iteration traces.

The prototype reframes this as a conversation: alternating user and AI messages, citations as pill badges, day dividers between sessions, a floating composer that shares the shell with the wiki view. The interaction model moves from "ask once, render once" to "multi-turn with local history".

Backend protocol is unchanged. `WebSocket /ws/chat` still takes a `ChatCompletionRequest` with `messages: ChatMessage[]` and streams raw text chunks. The frontend now accumulates those messages across turns and keeps them in `localStorage`.

All new code is written to `src_v2/`. The existing `src/` is left untouched.

## Target file

`src_v2/app/[owner]/[repo]/ask/page.tsx` — new file, target under 200 lines.

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
- Composer (shared `<Composer variant="chat">`): submit creates a new user message, fires a WS request, then appends an AI message that fills from the streaming text.

## Components to build

- `src_v2/components/chat/ChatStream.tsx` (~80 lines) — renders the ordered list of messages with day dividers. Pure view; no state.
- `src_v2/components/chat/Message.tsx` (~60 lines) — one message bubble. Props: `role`, `content`, `timestamp`, `model?`, `citations?`. Uses `<Markdown>` from `src_v2/components/Markdown.tsx` to render `content`.
- `src_v2/components/chat/Citation.tsx` (~20 lines) — pill-shaped citation link. Hover reveals the file path.

## State management

State lives in the page component via `useState` + `useEffect`. No Redux/Zustand. The flow per turn:

1. User hits `↵` in `<Composer>`.
2. Page constructs a `ChatCompletionRequest` from:
   - `repo_url` — computed with `getRepoUrl(repoInfo)` from `src_v2/utils/getRepoUrl.tsx`.
   - `messages` — all prior turns plus the new user message.
   - `provider`, `model`, `token`, `excluded_*`, `included_*` — from `useSettings()` (PLAN-003 `src_v2/contexts/SettingsContext.tsx`).
   - `language` — from `LanguageContext` (not user-toggleable in v1; defaults to "en").
3. Opens a socket via `src_v2/utils/websocketClient.ts` (copied in PLAN-003).
4. Appends an empty AI message with `streaming: true`. As chunks arrive, append to its `content`.
5. On socket close:
   - Parse the accumulated text for a `## Citations` tail (prototype shows inline citation pills; extract them with a regex). Store on the message as `citations: string[]`.
   - Mark message complete.
   - Call `useConversationHistory().addConversation(repoKey, ...)` or update the existing entry.
6. On failure/timeout: fall back to `POST /api/chat/stream` per the existing websocketClient behavior.

## Citation parsing

Citations in the prototype look like:

```
<span class="citation"><span class="citation__num">1</span>src/hooks/pretool.ts:42</span>
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

None — `src/` is left untouched. `src/components/AskComposer.tsx` and `src/components/AskResultView.tsx` remain in `src/` as-is.

## Critical files referenced or modified

- `prototype/app-chat.html` — DOM source of truth
- `prototype/styles.css:727-890` — chat-stream styles
- `src_v2/app/[owner]/[repo]/ask/page.tsx` — new
- `src_v2/components/chat/*` — new folder
- `src_v2/components/Markdown.tsx` — copied from `src/` in PLAN-003
- `src_v2/utils/websocketClient.ts` — copied from `src/` in PLAN-003
- `src_v2/utils/getRepoUrl.tsx` — copied from `src/` in PLAN-003
- `src_v2/hooks/useConversationHistory.ts` — introduced in PLAN-003
- `src_v2/contexts/SettingsContext.tsx` — introduced in PLAN-003
- `docs/api/frontend-backend-apis.md` §5.1 (`/ws/chat`), §5.2 (`/api/chat/stream` fallback) — contract

## Verification

In addition to the PLAN-002 shared harness:

1. Open `src_v2/` `/[owner]/[repo]/ask` on a cached repository. Ask a question. The AI message populates token-by-token as the WebSocket streams.
2. Refresh the page with the same `?convId`. The prior messages re-render from `localStorage`.
3. Open DevTools → Network → WS frame inspector. Confirm the payload matches `ChatCompletionRequest` and contains the full `messages` history, not just the latest question.
4. Simulate WS failure (stop the backend mid-request). The client retries over `POST /api/chat/stream` and continues streaming into the same message.
5. Ask three questions that span midnight local time. A `<DayDivider>` appears between the messages that crossed the boundary.
6. The sidebar `<ProjectTree>` in `<AppShell>` updates immediately when a new conversation is started (shared hook, not a page-local store).
7. A question with `?q=explain+hooks` in the URL auto-submits on mount.
8. Visual parity with `prototype/app-chat.html` at 1440×900.
9. `grep -r "from.*src/" src_v2/` returns no hits.

## Risks

- **Citation parsing fragility.** Without structured events, citations depend on model output formatting. Accept that v1 shows no citations when the model deviates; do not let regex failures crash the render.
- **localStorage quota.** A long conversation with embedded code can consume MBs. The eviction guard is best-effort; add a "Clear history" action under Settings as a follow-up.
- **WebSocket reconnect storms.** If a user types quickly after an error, the fallback could double-fire. Debounce the submit and disable the composer while `streaming: true`.
- **Multi-tab interference.** Two tabs writing to `localStorage.opswiki.conversationHistory` can clobber each other. Use a `storage` event listener on the hook to refresh state if another tab writes; accept last-writer-wins for the messages array.
