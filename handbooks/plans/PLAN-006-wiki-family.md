---
number: PLAN-006
name: Wiki Workshop Slides and Loading
description: Rewrites the wiki reading view, workshop view, slides presenter, and generation loading screen to match the OpsWiki prototypes while preserving the legacy wiki-generation stream contract.
update_at: 2026-05-06
category: improvement-plan
language: en
status: proposed
---

# PLAN-006 Wiki, Workshop, Slides, and Loading

## Context

The "wiki family" is the bulk of the app. Today:

- `src/app/[owner]/[repo]/page.tsx` (2,267 lines) hosts wiki generation, cache, sidebar tree, composer, model selection, and export.
- `src/app/[owner]/[repo]/workshop/page.tsx` (634 lines) renders a single long-form workshop.
- `src/app/[owner]/[repo]/slides/page.tsx` (1,300 lines) renders wiki content as slides.
- There is no dedicated loading screen; the wiki page shows inline spinners while streaming.

The prototype introduces four distinct surfaces for these: `app-wiki.html`, `app-workshop.html`, `app-slides.html`, `app-loading.html`. All share the same app shell except slides (chrome-minimal) and loading (full-screen without sidebar).

All new code is written to `src_v2/`. The existing `src/` files are left untouched. This plan groups the wiki family because they share data sources (`GET /api/wiki_cache`, `WebSocket /ws/chat`) and the cached `WikiStructure` from `src_v2/types/wiki/` (copied from `src/` in PLAN-003).

PLAN-007 added structured agent chat connectors for PLAN-005. This wiki-family plan does not switch wiki/workshop/slides generation to agent chat. Keep using the legacy raw-text `/ws/chat` generation path unless a later plan explicitly migrates these routes to `/ws/agent-wiki`.

## Shared behavior

- All three interactive routes (wiki, workshop, slides) first call `GET /api/wiki_cache`. If a cache exists, render immediately. If not, redirect to `/[owner]/[repo]?status=generating` — which mounts the loading screen and triggers generation.
- Generation itself uses `WebSocket /ws/chat` following the structure-then-pages pattern described in `docs/api/frontend-backend-apis.md` §6.2. On completion, `POST /api/wiki_cache` persists the result. Do not parse `/ws/chat` as `AgentChatEvent`; it streams raw text.
- All three routes read the cached `WikiStructure` shape (see `docs/api/frontend-backend-apis.md` §3). Types are defined in `src_v2/types/wiki/wikistructure.tsx` and `src_v2/types/wiki/wikipage.tsx` (copied from `src/` in PLAN-003).

## Sub-route 1 — Wiki reading view

### Target file

`src_v2/app/[owner]/[repo]/page.tsx` — new file, target under 250 lines. Implements the `app-wiki.html` layout.

### Structure

- `<AppShell>` with Topbar (switcher → Wiki active) and Composer (variant `"wiki"`).
- Content grid (`prototype/styles.css:896-903`):
  - Left: sticky `<WikiToc>` — hierarchical tree of sections and pages.
  - Right: `<WikiArticle>` rendering the currently-selected page.
- Composer submit: redirect to `/[owner]/[repo]/ask?q=<question>` (PLAN-002 D7).

### Components to build

- `src_v2/components/wiki/WikiToc.tsx` (~90 lines) — new, matching prototype. Consumes `WikiStructure.sections` and nested pages. Active state from route search param `?page=<id>`.
- `src_v2/components/wiki/WikiArticle.tsx` (~120 lines) — eyebrow, h1, italic lede (first sentence of page content), metadata row (Module / Entry / Updated / Confidence), then `<Markdown>` for the body. Mermaid blocks render via `src_v2/components/Mermaid.tsx` (copied in PLAN-003).

### Data

- `GET /api/wiki_cache?owner&repo&repo_type&language`. If null, redirect to loading.
- Currently-selected page resolves via `?page=<id>` query param; default to the first page of the first section.
- `POST /api/wiki_cache` is only called by the loading screen, not from this route.

### Export

- The topbar `...` menu exposes Markdown and JSON export via `POST /export/wiki` per `docs/api/frontend-backend-apis.md` §4.10. Single action, no modal.

## Sub-route 2 — Workshop view

### Target file

`src_v2/app/[owner]/[repo]/workshop/page.tsx` — new file, target under 200 lines. Implements `app-workshop.html`.

### Structure

- `<AppShell>` with Topbar (switcher → Workshop active) and Composer (variant `"wiki"`).
- Content grid (180px progress rail + 1fr article, see `prototype/app-workshop.html`):
  - Left: `<WorkshopRail>` — numbered step list with done / active / upcoming states.
  - Right: `<WorkshopArticle>` — step headers, prose, exercise blocks (`.workshop-exercise`).

### Components to build

- `src_v2/components/workshop/WorkshopRail.tsx` (~60 lines).
- `src_v2/components/workshop/WorkshopArticle.tsx` (~80 lines).
- `src_v2/components/workshop/ExerciseBlock.tsx` (~25 lines) — paper-panel with left rust border, "Try it" label, task body, and italic hint.

### Data

- Workshop content is stored alongside the wiki cache. Current backend writes it under the same `~/.adalflow/wikicache/` file? It doesn't — the workshop page today regenerates from `/ws/chat` on every load. Accept that for v1: if no local state is cached, regenerate and display.
- localStorage cache: write generated workshop HTML under `opswiki.workshop.${repoKey}.${language}` to avoid regeneration on every visit.

## Sub-route 3 — Slides presenter

### Target file

`src_v2/app/[owner]/[repo]/slides/page.tsx` — new file, target under 200 lines. Implements `app-slides.html`.

### Structure

- Full-viewport container — no `<AppShell>` sidebar. Linen background (`--paper-hover`).
- Minimal top chrome: back link to `/[owner]/[repo]`, slide counter.
- Main: slide stage with current slide centered and prev/next stubs receding.
- Bottom nav: prev/next buttons, fraction, dot progress, fullscreen button, keyboard hint.
- Keyboard bindings: `←` prev, `→` next, `f` fullscreen, `esc` exit fullscreen.

### Components to build

- `src_v2/components/slides/SlideStage.tsx` (~80 lines).
- `src_v2/components/slides/SlideCard.tsx` (~60 lines) — variant `"divider" | "content" | "diagram"`.
- `src_v2/components/slides/SlideNav.tsx` (~50 lines).

### Data

- Slides derive from the wiki cache: one slide per page, plus a `divider` card per section.
- Generation path matches existing behavior (`/ws/chat` streaming plan-then-slides). Cache HTML per slide in `localStorage.opswiki.slides.${repoKey}.${language}`.
- The rendered slide bodies use `<Markdown>` from `src_v2/components/Markdown.tsx` (copied in PLAN-003) — no custom code-highlight tokens.

## Sub-route 4 — Generation loading screen

### Target file

`src_v2/app/[owner]/[repo]/page.tsx` also handles `?status=generating`. When that query param is set, render `<GenerationLoader>` instead of the wiki content. Alternatively (simpler), introduce a dedicated `src_v2/components/generation/GenerationLoader.tsx` component and let the wiki page gate its render.

The dedicated-component approach is preferred: the wiki page stays small; the loader is self-contained.

### Component

`src_v2/components/generation/GenerationLoader.tsx` — new, target under 150 lines. Implements `app-loading.html`.

Renders:

- Breadcrumb back link + repo path.
- Phase group: eyebrow ("Now running"), serif `<h1>` with current phase, line of strikethrough-previous + bold-current + faint-upcoming phases (Clone → Chunk → Embed → Generate).
- Log panel: paper-panel with hairline border, fixed height, `overflow-y: auto`, auto-scroll to bottom. Each line is a `.log-line` (prototype class).
- Hairline progress bar with percentage label.
- Italic time-estimate aside.
- Cancel link back to `/`.

### Phase parsing (PLAN-002 D6)

- Hook `useGenerationPhases(stream: ReadableStream<string>)` returns `{ phase, lines, percent, done, error }`.
- Parser: regexes against incoming text:
  - `Cloning repository` → phase `clone`.
  - `Chunking (\d+) files` → phase `chunk`, extract file count.
  - `Embedding chunk (\d+)/(\d+)` → phase `embed`, compute `percent = current/total * 50%` (embedding is half the visual bar).
  - `Generating wiki structure` → phase `generate`.
  - `Streaming wiki page (\d+)/(\d+)` → phase `generate`, compute `percent = 50% + current/total * 50%`.
  - Anything else → append as a log line with faint styling.
- On stream close without error → transition to wiki view; `POST /api/wiki_cache`; navigate to `/[owner]/[repo]`.

### Components to build

- `src_v2/components/generation/GenerationLoader.tsx`.
- `src_v2/components/generation/PhasePipeline.tsx` (~40 lines).
- `src_v2/components/generation/LogPanel.tsx` (~60 lines).
- `src_v2/hooks/useGenerationPhases.ts` (~80 lines).

## Components to delete after this plan

None — `src/` is left untouched. `src/components/WikiTreeView.tsx` and any inline loading spinners remain in `src/` as-is.

## Critical files referenced or modified

- `prototype/app-wiki.html`, `prototype/app-workshop.html`, `prototype/app-slides.html`, `prototype/app-loading.html` — DOM sources
- `prototype/styles.css:892-1358` — layout and composer styles
- `src_v2/app/[owner]/[repo]/page.tsx` — new
- `src_v2/app/[owner]/[repo]/workshop/page.tsx` — new
- `src_v2/app/[owner]/[repo]/slides/page.tsx` — new
- `src_v2/components/wiki/*`, `src_v2/components/workshop/*`, `src_v2/components/slides/*`, `src_v2/components/generation/*` — new
- `src_v2/components/Markdown.tsx`, `src_v2/components/Mermaid.tsx` — copied from `src/` in PLAN-003
- `src_v2/utils/websocketClient.ts` — copied from `src/` in PLAN-003; use `createChatWebSocket` for wiki-family raw text generation, not `createAgentChatWebSocket`
- `src_v2/hooks/useGenerationPhases.ts` — new
- `docs/api/frontend-backend-apis.md` §4.7, §4.8, §4.9, §4.10, §5.1, §5.3, §6.2, §6.4 — contract boundaries; §5.3 is documented to avoid confusing agent chat with wiki generation

## Verification

In addition to the PLAN-002 shared harness:

1. `GET /[owner]/[repo]` on a cached repo renders the wiki in under 500ms (no extra WS call). Compare against `prototype/app-wiki.html`.
2. `GET /[owner]/[repo]` on an uncached repo renders `<GenerationLoader>` and shows the phase pipeline advancing as the WS streams. Compare against `prototype/app-loading.html`.
3. Generation completes → page automatically navigates to the cached wiki.
4. Cancel link on the loader closes the WebSocket and navigates to `/`.
5. `/[owner]/[repo]/workshop` renders the progress rail with the correct done/active/upcoming state.
6. `/[owner]/[repo]/slides` — arrow keys advance slides, `f` toggles fullscreen, `esc` exits. The slide counter updates.
7. The composer on wiki and workshop submits → navigates to `/[owner]/[repo]/ask?q=<query>` (redirect model per PLAN-002 D7).
8. The "Present as slides" icon in the topbar links to `/[owner]/[repo]/slides`.
9. Export via the topbar menu downloads a `.md` or `.json` file matching `docs/api/frontend-backend-apis.md` §4.10.
10. Resize to 1100px: wiki TOC collapses (responsive guard from `prototype/styles.css:1350-1358`).
11. `grep -r "from.*src/" src_v2/` returns no hits — no cross-directory imports.

## Risks

- **Phase regex brittleness.** The phase parser (PLAN-002 D6) depends on backend text formatting. If the backend log strings change, phases silently stop advancing. Mitigation: fall back to an indeterminate progress bar that animates regardless of parser state; do not block navigation on phase detection.
- **Agent-chat protocol mismatch.** PLAN-007's `/ws/agent-chat` emits structured tool events for Ask; this plan's loading/wiki/workshop/slides flows still consume raw text from `/ws/chat`. Accidentally switching the loader to `createAgentChatWebSocket` will break phase parsing and cache persistence.
- **Workshop and slides regenerate on every visit.** Current backend does not persist them. Local cache per `${repoKey}.${language}` hides this cost on revisits but first-time users still wait. Accept in v1; an agent-wiki extension can add structured persistence later.
- **Mermaid rendering on slides.** `<Mermaid>` sizes diagrams to container width; on a 16:9 slide, large diagrams overflow. Add a `maxHeight` prop to clip and provide a fullscreen zoom as today's `<Mermaid>` already does.
- **Cache-key collision.** Per RISK-003, `owner/repo` without host can collide across GitHub/GitLab/Bitbucket. This plan does not fix that — it inherits the risk. Surface the repo type in the topbar breadcrumb so users notice a mismatch.
- **Long wiki page trees.** The sticky TOC is bounded by `calc(100vh - var(--topbar-h) - 80px)` with `overflow-y: auto`. For deeply nested structures the tree can grow taller than viewport — confirm the inner scrollbar works.
