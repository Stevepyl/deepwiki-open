---
number: PLAN-003
name: Foundation — Design Tokens and App Shell
description: Replaces the dark-mode token set with the prototype Paper and Ink tokens, rebuilds the shared app shell, and preserves shared chat and agent-chat connectors.
update_at: 2026-05-06
category: improvement-plan
language: en
status: implemented
---

# PLAN-003 Foundation — Design Tokens and App Shell

## Context

PLAN-002 (overview) defers every shared primitive to this file. Without it, the welcome, projects, chat, and wiki sub-plans have nothing to import. Foundation must land before any of PLAN-004 through PLAN-006 can merge.

All work in this plan is written to `src/`. The existing `src_old/` directory is not modified.

Two kinds of work live here:
1. **Tokens and Tailwind base CSS.** Create `src/app/globals.css` with Tailwind imports, the prototype's `:root` block, and a Tailwind v4 `@theme` block. No `data-theme` or `.dark:` selectors. Wire the font imports from `prototype/styles.css:5`.
2. **App shell components.** Build the small, reusable primitives that every app page renders inside: `<AppShell>`, `<Sidebar>`, `<Topbar>`, `<Composer>`, `<Wordmark>`, `<SettingsPanel>`, plus a handful of icon and chrome helpers. These live in `src/components/shell/`.

Scope is deliberately limited to shared infrastructure. Page-level routes and per-page components are owned by the later sub-plans.

PLAN-007 added frontend connector infrastructure for structured agent chat. Foundation owns copying that shared infrastructure into `src/`; PLAN-005 owns using it in the Ask route.

## Token rewrite

**File:** `src/app/globals.css` (new)

- Create with a Tailwind v4 `@theme` block that registers the tokens from `prototype/styles.css:7-47`. Values — paper, ink, hairline, accent, radii, sidebar and composer dimensions — are copied verbatim.
- Keep the Google Fonts `@import` from `prototype/styles.css:5` for v1 parity. A `next/font` migration is a separate follow-up.
- No `data-theme="dark"` selector.
- Register global element defaults (`html, body`, `button`, `input`, `a`, `::selection`) by copying `prototype/styles.css:53-95`.
- Do not keep shell, composer, settings-panel, markdown, or wordmark component selectors in globals. Component styling is Tailwind-first inside the relevant `src` component files; globals owns tokens and base element defaults only.

**File:** `tailwind.config.js`
- Remove any color palette overrides. Tailwind v4 reads from `@theme` in `globals.css`.
- Ensure `content` globs cover both `src_old/**/*.{ts,tsx}` and `src/**/*.{ts,tsx}`.

**File:** `src/app/layout.tsx` (new)
- No `ThemeProvider`.
- Has `LanguageProvider` (see PLAN-002 D9) and `SettingsProvider`.
- Sets `<html lang="en">` with no `data-theme` attribute.

## Wordmark

**File:** `src/components/shell/Wordmark.tsx` (new, ~20 lines)

Props: `size: "hero" | "sidebar"`. Renders `<h1>` or `<span>` depending on size, with content `<em>Ops</em>Wiki`. The two sizes map to Tailwind utility class sets using the Paper and Ink font and color tokens.

Used by the welcome page (hero) and the sidebar (brand).

## App shell

The shell is split into small components so each stays under ~120 lines. They live together in `src/components/shell/`.

### `<AppShell>` (new)

**File:** `src/components/shell/AppShell.tsx` (~60 lines)

- Two-column layout: `<Sidebar>` on the left, Tailwind-styled `<main>` on the right.
- Applies the fade mask in `prototype/styles.css:699-712` with Tailwind `after:*` utilities on the main element.
- Accepts `topbar` and `children` slots plus an optional `composer` slot.
- Nothing else. All business logic belongs in pages.

### `<Sidebar>` (new)

**File:** `src/components/shell/Sidebar.tsx` (~100 lines)

- Fixed-width (`--sidebar-w`) paper panel.
- Children (top to bottom):
  - `<Wordmark size="sidebar">` wrapping a `next/link` to `/`.
  - `<SidebarSearch>` input (focuses on `⌘K`; no results handler in v1 — it scrolls/filters the local list).
  - Scroll area with `<ProjectTree>`.
  - Footer with `<NewRepoButton>` and `<UserChip>`.
- `<ProjectTree>`, `<ProjectGroup>`, `<Conv>` are internal subcomponents (file-colocated as long as total stays under 200 lines; extract otherwise).
- Data source: `useConversationHistory()` hook from `src/hooks/useConversationHistory.ts` (new, see below).

### `<Topbar>` (new)

**File:** `src/components/shell/Topbar.tsx` (~70 lines)

- Matches `prototype/styles.css:586-686` with Tailwind utility classes.
- Accepts `breadcrumb`, `switcher` (for 3-tab chat/wiki/workshop), and `actions` slots.
- Built-in `<IconButton>` subcomponent for `topbar__actions`.

### `<Composer>` (new)

**File:** `src/components/shell/Composer.tsx` (~120 lines)

- `variant: "chat" | "wiki"` — switches Tailwind utility class sets only (solid paper vs backdrop-blur + SVG noise).
- Props: `modeHint: string`, `placeholder`, `value`, `onChange`, `onSubmit`, `onOpenSettings`, `footer?: ReactNode`.
- Renders the input, attach button (disabled no-op in v1), settings gear (opens `<SettingsPanel>`), and send button exactly as `prototype/styles.css:1186-1344`.
- The fixed positioning math and z-index are copied from the prototype using Tailwind arbitrary values; spacing tokens come from `globals.css`.

### `<SettingsPanel>` (new)

**File:** `src/components/shell/SettingsPanel.tsx` (~180 lines)

- Slide-in panel on the right edge, backed by a `<dialog>` or a portal.
- Sections:
  - Provider + model picker. Reads `GET /api/models/config`. Stores selection in `localStorage` under `opswiki.modelSelection`.
  - Authorization code. Reads `GET /api/auth/status` once on mount; if required, accepts input and posts `POST /api/auth/validate`. Stores the validated code under `opswiki.authCode`.
  - Repository token (plain text input).
  - Included/excluded dirs and files (two textareas, newline-separated per backend convention in `docs/api/frontend-backend-apis.md` §3 ChatCompletionRequest).
- No wiki-type toggle (ADR-001 / PLAN-002 D4).
- Exposes a context `SettingsContext` so pages can read the current model, token, and filters without prop-drilling.

**File:** `src/contexts/SettingsContext.tsx` (new, ~80 lines)

- `SettingsProvider` stores user-selectable values in `localStorage` (through a wrapper that is SSR-safe: `typeof window !== "undefined"` guard).
- `useSettings()` hook returns `{ provider, model, authCode, token, excludedDirs, excludedFiles, includedDirs, includedFiles, setters... }`.
- Mounted once in `src/app/layout.tsx` alongside `LanguageProvider`.

## Conversation-history hook

**File:** `src/hooks/useConversationHistory.ts` (new, ~60 lines)

- Shape: `{ repos: RepoEntry[], addConversation(repoKey, conv), removeConversation(convId), upsertRepo(entry), removeRepo(repoKey) }`.
- Persists to `localStorage` under `opswiki.conversationHistory`.
- `RepoEntry = { type, owner, repo, convs: ConvEntry[] }`.
- `ConvEntry = { id, title, lastMessageAt, messageCount }`.
- One-time migration on read: if an old `deepwiki.*` key exists in `localStorage`, delete it (do not attempt to import — the shape is different).

This hook powers the sidebar's project tree. PLAN-005 writes into it from the chat page.

## Files to copy from `src_old/`

These files are copied verbatim into `src/` as part of this plan. They are not modified. The `src_old/` originals remain untouched.

- `src_old/components/Markdown.tsx` → `src/components/Markdown.tsx`
- `src_old/components/Mermaid.tsx` → `src/components/Mermaid.tsx`
- `src_old/utils/websocketClient.ts` → `src/utils/websocketClient.ts`
- `src_old/utils/agentChatStream.ts` → `src/utils/agentChatStream.ts`
- `src_old/utils/getRepoUrl.tsx` → `src/utils/getRepoUrl.tsx`
- `src_old/utils/urlDecoder.tsx` → `src/utils/urlDecoder.tsx`
- `src_old/hooks/useProcessedProjects.ts` → `src/hooks/useProcessedProjects.ts`
- `src_old/contexts/LanguageContext.tsx` → `src/contexts/LanguageContext.tsx`
- `src_old/types/**` → `src/types/**`
- `src_old/app/api/chat/agent-stream/route.ts` → `src/app/api/chat/agent-stream/route.ts` if `src/app` is being exercised as a standalone app root during the refinement branch.

After copying, update all internal imports in the copied files to reference `src/` paths instead of `src_old/` paths.

## Small primitives

**File:** `src/components/shell/IconButton.tsx` (~15 lines) — renders a Tailwind-styled icon button wrapping an SVG child.

**File:** `src/components/shell/Switcher.tsx` (~40 lines) — 3-tab switcher for Chat / Wiki / Workshop used by PLAN-005/006. Active state derived from the current pathname.

**File:** `src/components/shell/Eyebrow.tsx` (~10 lines) — small uppercase-tracked label used on wiki article headers and loading phase indicator.

**File:** `src/components/shell/DayDivider.tsx` (~15 lines) — hairline-with-text divider used in chat streams.

## Critical files referenced or modified

- `src/app/globals.css` — new (Tailwind import, Paper and Ink tokens, base element defaults only)
- `src/app/layout.tsx` — new (no ThemeProvider, has SettingsProvider)
- `tailwind.config.js` — updated to include `src/**` content glob
- `src/components/shell/*` — new folder, Tailwind-first shell styling
- `src/hooks/useConversationHistory.ts` — new
- `src/contexts/SettingsContext.tsx` — new
- `src/components/Markdown.tsx`, `src/components/Mermaid.tsx` — copied from `src_old/`
- `src/utils/*`, `src/hooks/useProcessedProjects.ts`, `src/contexts/LanguageContext.tsx`, `src/types/**` — copied from `src_old/`
- `src/utils/agentChatStream.ts`, `src/types/agentChat.ts`, `src/app/api/chat/agent-stream/route.ts` — PLAN-007 agent-chat frontend connector surface to preserve
- `prototype/styles.css` — canonical source for token values
- `prototype/app-chat.html`, `prototype/app-wiki.html`, `prototype/app-workshop.html` — canonical DOM structure for shell components
- `handbooks/adr/ADR-001-remove-wiki-type-toggle.md` — authority for dropping the toggle
- `docs/api/frontend-backend-apis.md` §4.4, §4.2, §4.3 — contract for SettingsPanel calls

## Verification

In addition to the PLAN-002 shared harness:

1. Run `bun run build` — no TypeScript errors from `src/`. The `src_old/` build is unaffected.
2. Open any `src/` route and confirm it renders without a `ThemeProvider`. A visual difference from `src_old/` is expected and acceptable.
3. Open DevTools → Application → Local Storage, confirm keys are namespaced under `opswiki.*` and a stale `deepwiki.*` key (if created manually) is removed on next load.
4. In the Console, call `getComputedStyle(document.body).getPropertyValue("--paper-main")` — it must equal `#F5F1EA`.
5. Resize to 1100px — responsive behavior from `prototype/styles.css:1350-1358` is reproduced with Tailwind utilities in the wiki components once PLAN-006 lands.
6. `grep -r "from.*src_old/" src/` returns no hits — all imports within `src/` are self-contained.
7. `src/utils/websocketClient.ts` exports both `createChatWebSocket` and `createAgentChatWebSocket`, and `src/utils/agentChatStream.ts` parses `application/x-ndjson` agent events.
8. `rg -n "^\\.[A-Za-z0-9_-]+" src/app/globals.css` returns no component selector blocks.

## Risks

- **Tailwind v4 token registration pitfalls.** Tailwind v4 syntax for `@theme` differs from v3 utility extension. A short smoke test (`<div className="text-[var(--ink-primary)]">`) in a scratch page catches wiring mistakes early.
- **Markdown component color tokens.** `src/components/Markdown.tsx` owns paper-ink typography, tables, blockquotes, inline code, and syntax highlighting through Tailwind utility classes plus the local `paperSyntaxTheme` object.
- **Import paths in copied files.** After copying files from `src_old/` to `src/`, all relative imports inside those files must be updated to reflect the new directory. A missed import will produce a build error, which is detectable and fixable immediately.
- **Agent connector drift.** `src_old/utils/websocketClient.ts` now contains both legacy raw-text chat and structured agent chat helpers. Keep both exports when copying; otherwise PLAN-005 cannot render tool-call progress without reintroducing backend-specific code in the page.
