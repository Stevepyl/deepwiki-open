---
number: PLAN-004
name: Welcome and Projects Directory
description: Rewrites the home route and the projects directory to match the OpsWiki welcome page and project library prototypes.
update_at: 2026-05-05
category: improvement-plan
language: en
status: proposed
---

# PLAN-004 Welcome and Projects Directory

## Context

Two entry points. `src/app/page.tsx` is 627 lines of demo-diagrams, processed-projects grid, and configuration-modal scaffolding. `src/app/wiki/projects/page.tsx` duplicates parts of that. The prototype replaces both with a much smaller surface:

- `prototype/index.html` — centered hero, paper-scratch search, 2×2 example repo grid, top-left breadcrumb to `/projects`.
- `prototype/app-projects.html` — alphabetical rail on the left, grouped card grid on the right, filter input, grid/list toggle.

This plan rewrites both, reusing the `useProcessedProjects` hook and the shell primitives from PLAN-003.

## Welcome page

### Target file

`src/app/page.tsx` — full rewrite, target under 120 lines.

### Structure

- Root: `<main className="welcome">`. Uses CSS already provided by the prototype tokens in `globals.css`; no Tailwind utility classes except for small layout helpers.
- Top-left strip:
  - "OpsWiki · v0 prototype" eyebrow (see `prototype/index.html:12`).
  - Right-side `All projects →` link to `/projects`.
- Hero:
  - `<Wordmark size="hero">` rendering `<em>Ops</em>Wiki`.
  - Serif italic subtitle: "Understand repos with AI".
- Paper-scratch search form:
  - Single text input, placeholder "Paste a GitHub URL, or describe what you want to understand…".
  - Submit handler: parse the input with `src/utils/urlDecoder.tsx`. If it is a repo URL, navigate to `/[owner]/[repo]?status=generating`. If not, navigate to `/?q=<query>` with a toast — do not block. For v1, accept only URLs; non-URL input just shows an inline hint.
- Example cards (2×2 grid of four hardcoded repos matching `prototype/index.html:39-55`). Click navigates to `/[owner]/[repo]?status=generating` for chat-preferred examples and `/[owner]/[repo]` for wiki-preferred examples, as the prototype links suggest.
- Footer: keyboard hint with `<kbd>↵</kbd>` and `<kbd>⌘K</kbd>`. `⌘K` focuses the page's search input. `↵` in the focused input submits.

### Data and side effects

- No backend call on mount. The welcome page is purely static apart from the form submit.
- `GET /api/auth/status` is NOT called here — it is called inside `<SettingsPanel>` when the user opens settings. Rationale: the welcome page does not yet know the user intends to generate anything.
- `useRouter().push(...)` for navigation; prefer `next/link` on the example cards for prefetch.

### Components to build

- `src/components/welcome/ScratchInput.tsx` (~50 lines) — the paper-scratch input from `prototype/styles.css:178-240`.
- `src/components/welcome/ExampleCard.tsx` (~25 lines) — `prototype/styles.css:275-305`.

### Components to delete after this plan

- `src/components/ConfigurationModal.tsx` (already noted in PLAN-003; delete in the final commit of this plan if not already deleted).
- `src/components/ProcessedProjects.tsx`.

## Projects directory

### Target file

`src/app/projects/page.tsx` — new route (replaces `src/app/wiki/projects/page.tsx`; delete the old file in the same commit).

Target length: under 180 lines.

### Structure

- Two-column flex layout:
  - Left: narrow alphabetical rail (~56px wide) mirroring `prototype/app-projects.html:13-33`. Each letter is a section anchor; a home icon at the top returns to `/`.
  - Right: the projects body (header, toolbar, content grid, footer).
- Header: serif `<h1>` "Projects", eyebrow with count, sort and `…` icon buttons.
- Toolbar: paper-scratch filter input, `Grid` / `List` pill switcher using `<Switcher>` from PLAN-003.
- Content: projects grouped by first letter of `owner/repo`. Each group has a big-serif letter heading and a 2-column grid (grid view) or single-column stack (list view) of cards.
- Card (`<DirCard>`): mono path, italic-serif description, sans metadata row (`N wiki pages · N convs · <relative time>`).
- Footer: `+ Analyze new repo` link to `/`, `⌘K` hint.

### Data and side effects

- `useProcessedProjects()` hook (existing, reused from `src/hooks/useProcessedProjects.ts`) returns the list from `GET /api/wiki/projects`.
- Project description comes from... the backend does not provide one. For v1, the card description is blank. A future enhancement can join wiki cache metadata. Plan placeholder: "Generated wiki for `<path>`." — kept in one helper, not inlined.
- Conversation count comes from `useConversationHistory()` (PLAN-003): count conversations whose `repoKey === ${type}:${owner}/${repo}`.
- Wiki-pages count: the backend `GET /api/wiki/projects` does not include a page count. For v1, omit that metric. Display only `N convs · <time>`.
- Delete flow (card hover action): calls `DELETE /api/wiki/projects` with `{ owner, repo, repo_type, language }` (language from settings or `"en"`). Documented caveat in `docs/api/frontend-backend-apis.md` §4.6: backend auth-mode is not forwarded; accept that limitation in v1 and surface a toast "Deletion failed — auth required" if the backend returns 401.

### Components to build

- `src/components/projects/AlphaRail.tsx` (~60 lines).
- `src/components/projects/DirCard.tsx` (~40 lines).
- `src/components/projects/FilterInput.tsx` (~25 lines) — reuses `<ScratchInput>` visual styling, smaller size.

### Routing

- Add `src/app/projects/page.tsx`.
- Delete `src/app/wiki/projects/page.tsx`.
- Delete the parent `src/app/wiki/` folder if it contains nothing else (grep to confirm).

## Critical files referenced or modified

- `prototype/index.html`, `prototype/app-projects.html` — DOM source of truth
- `prototype/styles.css` — visual source of truth
- `src/app/page.tsx` — rewritten
- `src/app/projects/page.tsx` — new
- `src/app/wiki/projects/page.tsx` — deleted
- `src/components/welcome/*` — new folder
- `src/components/projects/*` — new folder
- `src/hooks/useProcessedProjects.ts` — reused
- `src/hooks/useConversationHistory.ts` — consumed for the `N convs` count (introduced in PLAN-003)
- `src/utils/urlDecoder.tsx` — reused on welcome submit
- `docs/api/frontend-backend-apis.md` §4.5, §4.6 — backend contract for projects list/delete

## Verification

In addition to the PLAN-002 shared harness:

1. `/` renders the hero centered at 1440×900 with the subtitle, the scratch input, and the 2×2 example grid. Compare against `prototype/index.html` in a second tab.
2. Typing a valid GitHub URL into the scratch input and pressing `↵` navigates to `/[owner]/[repo]?status=generating`.
3. Clicking an example card navigates to the right target (chat vs wiki — matching the prototype links).
4. `/projects` lists the contents of `~/.adalflow/wikicache/`. Deleting one project in the UI removes the corresponding cache file on disk (verify with `ls ~/.adalflow/wikicache/` on the backend host).
5. Switching `Grid` / `List` in the projects toolbar re-renders the card grid; `localStorage.opswiki.projectsView` persists the choice across reloads.
6. The alpha rail anchors navigate to the correct letter groups. The active letter (as the user scrolls) is highlighted — implement with an `IntersectionObserver` on the group headings.
7. `grep -r "ProcessedProjects" src/` and `grep -r "ConfigurationModal" src/` return no hits after the plan.

## Risks

- **Missing description data.** Without a backend-supplied description, the card meta row is thin. Accept that in v1; the metric that matters is recency and conversation count.
- **Delete without auth code.** Per the documented caveat, delete can fail with 401. The error path is a toast, not a blocker.
- **`⌘K` collision.** `<Sidebar>` from PLAN-003 binds `⌘K` to sidebar search. The welcome page has no sidebar, so it binds `⌘K` to the scratch input. Confirm only one binding is active per route — the welcome page does not render `<AppShell>`.
