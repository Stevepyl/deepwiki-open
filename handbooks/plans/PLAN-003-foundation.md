---
number: PLAN-003
name: Foundation — Design Tokens and App Shell
description: Replaces the dark-mode token set with the prototype Paper and Ink tokens, rebuilds the shared app shell, and deletes obsolete configuration modals.
update_at: 2026-05-05
category: improvement-plan
language: en
status: proposed
---

# PLAN-003 Foundation — Design Tokens and App Shell

## Context

PLAN-002 (overview) defers every shared primitive to this file. Without it, the welcome, projects, chat, and wiki sub-plans have nothing to import. Foundation must land before any of PLAN-004 through PLAN-006 can merge.

Two kinds of work live here:
1. **Tokens and global CSS.** Replace `src/app/globals.css` with the prototype's `:root` block. Remove `data-theme` and `.dark:` selectors. Wire the font imports from `prototype/styles.css:5`.
2. **App shell components.** Build the small, reusable primitives that every app page renders inside: `<AppShell>`, `<Sidebar>`, `<Topbar>`, `<Composer>`, `<Wordmark>`, `<SettingsPanel>`, plus a handful of icon and chrome helpers. These live in a new folder `src/components/shell/`.

Scope is deliberately limited to shared infrastructure. Page-level routes and per-page components are owned by the later sub-plans.

## Token rewrite

**File:** `src/app/globals.css`

- Delete the entire existing file contents.
- Replace with a Tailwind v4 `@theme` block that registers the tokens from `prototype/styles.css:7-47`. Values — paper, ink, hairline, accent, radii, sidebar and composer dimensions — are copied verbatim.
- Keep the Google Fonts `@import` from `prototype/styles.css:5` for v1 parity. A `next/font` migration is a separate follow-up.
- Drop every `data-theme="dark"` selector.
- Register global element defaults (`html, body`, `button`, `input`, `a`, `::selection`) by copying `prototype/styles.css:53-95`.
- Copy the `.wordmark`, `.wordmark--hero`, `.wordmark--sidebar` rules from `prototype/styles.css:101-125` — they are stylistic primitives used by the Wordmark component.

**File:** `tailwind.config.js`
- Remove any color palette overrides. Tailwind v4 reads from `@theme` in `globals.css`.
- Ensure `content` globs still cover `src/**/*.{ts,tsx}`.

**File:** `src/app/layout.tsx`
- Drop `ThemeProvider`.
- Keep `LanguageProvider` (see PLAN-002 D9).
- Set `<html lang="en">` and drop any `data-theme` attribute.

## Wordmark

**File:** `src/components/shell/Wordmark.tsx` (new, ~20 lines)

Props: `size: "hero" | "sidebar"`. Renders `<h1>` or `<span>` depending on size, with content `<em>Ops</em>Wiki`. The two sizes map to `.wordmark--hero` and `.wordmark--sidebar` classes already present in globals.

Used by the welcome page (hero) and the sidebar (brand).

## App shell

The shell is split into small components so each stays under ~120 lines. They live together in `src/components/shell/`.

### `<AppShell>` (new)

**File:** `src/components/shell/AppShell.tsx` (~60 lines)

- Two-column layout: `<Sidebar>` on the left, `<main className="main">` on the right.
- Applies the fade mask in `prototype/styles.css:699-712` as a `::after` pseudo on `.main`.
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

- Matches `prototype/styles.css:586-686`.
- Accepts `breadcrumb`, `switcher` (for 3-tab chat/wiki/workshop), and `actions` slots.
- Built-in `<IconButton>` subcomponent for `topbar__actions`.

### `<Composer>` (new)

**File:** `src/components/shell/Composer.tsx` (~120 lines)

- `variant: "chat" | "wiki"` — changes class only (solid paper vs backdrop-blur + SVG noise).
- Props: `modeHint: string`, `placeholder`, `value`, `onChange`, `onSubmit`, `onOpenSettings`, `footer?: ReactNode`.
- Renders the input, attach button (disabled no-op in v1), settings gear (opens `<SettingsPanel>`), and send button exactly as `prototype/styles.css:1186-1344`.
- The fixed positioning math and z-index are copied from the prototype; spacing tokens come from `globals.css`.

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

## Deletions

These components go away in this plan. PLAN-004 through PLAN-006 do not reference them.

- `src/components/ConfigurationModal.tsx`
- `src/components/ModelSelectionModal.tsx`
- `src/components/WikiTypeSelector.tsx`
- `src/components/UserSelector.tsx`
- `src/components/TokenInput.tsx` (folded into `<SettingsPanel>`)
- `src/components/ProcessedProjects.tsx`
- `src/components/theme-toggle.tsx`

Only delete after all sub-plans that previously referenced them have been merged, or after importers in the routes they support have been rewritten in the same PR. Foundation's deletion step is the last commit in this plan.

## Small primitives

**File:** `src/components/shell/IconButton.tsx` (~15 lines) — renders a `.icon-btn` wrapping an SVG child.

**File:** `src/components/shell/Switcher.tsx` (~40 lines) — 3-tab switcher for Chat / Wiki / Workshop used by PLAN-005/006. Active state derived from the current pathname.

**File:** `src/components/shell/Eyebrow.tsx` (~10 lines) — small uppercase-tracked label used on wiki article headers and loading phase indicator.

**File:** `src/components/shell/DayDivider.tsx` (~15 lines) — hairline-with-text divider used in chat streams.

## Critical files referenced or modified

- `src/app/globals.css` — full rewrite
- `src/app/layout.tsx` — drop ThemeProvider, add SettingsProvider
- `tailwind.config.js` — remove palette overrides
- `src/components/shell/*` — new folder
- `src/hooks/useConversationHistory.ts` — new
- `src/contexts/SettingsContext.tsx` — new
- `prototype/styles.css` — canonical source for token values
- `prototype/app-chat.html`, `prototype/app-wiki.html`, `prototype/app-workshop.html` — canonical DOM structure for shell components
- `handbooks/adr/ADR-001-remove-wiki-type-toggle.md` — authority for dropping the toggle
- `docs/api/frontend-backend-apis.md` §4.4, §4.2, §4.3 — contract for SettingsPanel calls

## Verification

In addition to the PLAN-002 shared harness:

1. Run `bun run build` — no TypeScript errors from deleted components referenced elsewhere.
2. Open any route — even a page that is not yet rewritten in this plan — and confirm the unchanged pages still render without `ThemeProvider`. A graceful visual regression here is acceptable; a runtime error is not.
3. Open DevTools → Application → Local Storage, confirm keys are namespaced under `opswiki.*` and a stale `deepwiki.*` key (if created manually) is removed on next load.
4. In the Console, call `getComputedStyle(document.body).getPropertyValue("--paper-main")` — it must equal `#F5F1EA`.
5. Resize to 1100px — responsive guard at `prototype/styles.css:1350-1358` holds in the globals.

## Risks

- **Cascading import errors when deleting modals.** PLAN-004 and PLAN-005 must land before the import graph is clean. Sequence: PLAN-003 creates the new shell but leaves old components in place; PLAN-004/005/006 rewrite pages; a final commit in PLAN-003's branch deletes the orphans. If merging sub-plans separately, each sub-plan deletes only its own orphans.
- **Tailwind v4 token registration pitfalls.** Tailwind v4 syntax for `@theme` differs from v3 utility extension. A short smoke test (`<div className="text-[var(--ink-primary)]">`) in a scratch page catches wiring mistakes early.
- **Markdown component color tokens.** `src/components/Markdown.tsx` likely references dark-mode Tailwind classes for syntax highlighting. Audit it during this plan and rewrite to use the paper-ink palette — specifically the `.k`, `.s`, `.c`, `.n`, `.f` classes at `prototype/styles.css:836-839, 1144-1147`.
