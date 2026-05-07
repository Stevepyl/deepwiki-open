"use client";

import Link from "next/link";
import { FiGrid, FiList, FiMoreHorizontal, FiPlus, FiSliders } from "react-icons/fi";
import { AlphaRail } from "../../components/projects/AlphaRail";
import { DirCard } from "../../components/projects/DirCard";
import { FilterInput } from "../../components/projects/FilterInput";
import { IconButton } from "../../components/shell/IconButton";
import { Switcher } from "../../components/shell/Switcher";
import {
  descriptionFor,
  pathOf,
  projectHref,
  relativeTime,
  type ViewMode,
  useProjectsDirectory,
} from "../../hooks/useProjectsDirectory";

export default function ProjectsPage() {
  const state = useProjectsDirectory();

  return (
    <div className="flex min-h-screen">
      <AlphaRail activeLetter={state.activeLetter} letters={state.groups.map((group) => group.letter)} />
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[var(--topbar-h)] shrink-0 items-center justify-between border-b border-[var(--hairline)] bg-[var(--paper-main)] px-10">
          <div className="flex items-baseline gap-3.5">
            <h1 className="font-serif text-[22px] font-semibold tracking-normal text-[var(--ink-primary)]">
              Projects
            </h1>
            <span className="font-sans text-[11px] font-medium uppercase tracking-[0.14em] text-[var(--ink-muted)]">
              {state.projects.length} repositories
            </span>
          </div>
          <div className="flex items-center gap-1">
            <IconButton
              aria-label={`Sort by ${state.sortMode === "name" ? "recent" : "name"}`}
              onClick={() => state.setSortMode(state.sortMode === "name" ? "recent" : "name")}
            >
              <FiSliders aria-hidden="true" />
            </IconButton>
            <IconButton aria-label="More project actions">
              <FiMoreHorizontal aria-hidden="true" />
            </IconButton>
          </div>
        </header>

        <div className="flex items-center gap-4 border-b border-[var(--hairline)] px-10 py-5">
          <FilterInput onChange={state.setQuery} ref={state.filterRef} value={state.query} />
          <Switcher
            ariaLabel="Projects view"
            onValueChange={(nextView) => state.setView(nextView as ViewMode)}
            options={[
              { value: "grid", label: "Grid", icon: FiGrid },
              { value: "list", label: "List", icon: FiList },
            ]}
            value={state.view}
          />
        </div>

        {state.notice ? <div className="border-b border-[var(--hairline)] px-10 py-2 text-xs text-[var(--accent)]">{state.notice}</div> : null}
        {state.error ? <div className="border-b border-[var(--hairline)] px-10 py-3 text-sm text-[var(--accent)]">{state.error}</div> : null}

        <div
          className="flex-1 overflow-y-auto px-10 pb-16 pt-8 [&::-webkit-scrollbar]:w-2.5 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb]:bg-[var(--hairline-strong)]"
          ref={state.contentRef}
        >
          {state.isLoading ? <div className="font-serif text-lg italic text-[var(--ink-muted)]">Loading projects...</div> : null}
          {!state.isLoading && state.groups.length === 0 ? (
            <div className="font-serif text-lg italic text-[var(--ink-muted)]">No projects match this filter.</div>
          ) : null}
          {state.groups.map((group) => (
            <section key={group.letter}>
              <div
                className="mb-3.5 flex scroll-mt-6 items-center gap-2.5 border-b border-[var(--hairline)] pb-3 pt-4 font-mono text-[11px] font-medium uppercase tracking-[0.16em] text-[var(--ink-muted)]"
                data-letter={group.letter}
                id={`group-${group.letter.toLowerCase()}`}
              >
                <span className="font-serif text-xl font-medium normal-case tracking-normal text-[var(--ink-secondary)]">
                  {group.letter}
                </span>
              </div>
              <div className={`${state.view === "grid" ? "grid grid-cols-1 gap-2.5 lg:grid-cols-2" : "flex flex-col gap-2.5"} mb-10`}>
                {group.items.map((project) => (
                  <DirCard
                    conversationCount={state.conversationCounts.get(`${project.repo_type}:${pathOf(project)}`) ?? 0}
                    description={descriptionFor(project)}
                    href={projectHref(project)}
                    isDeleting={state.deletingId === project.id}
                    key={project.id}
                    onDelete={() => state.deleteProject(project)}
                    path={pathOf(project)}
                    updatedAt={relativeTime(project.submittedAt)}
                    view={state.view}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>

        <footer className="flex items-center justify-between border-t border-[var(--hairline)] px-10 pb-7 pt-5">
          <Link
            className="inline-flex items-center gap-2 rounded-[var(--radius-sm)] border border-[var(--hairline)] px-3.5 py-2 font-sans text-[13px] font-medium text-[var(--ink-primary)] transition-colors duration-150 hover:border-[var(--hairline-strong)] hover:bg-[var(--paper-hover)] [&>svg]:h-[13px] [&>svg]:w-[13px] [&>svg]:text-[var(--accent)]"
            href="/"
          >
            <FiPlus aria-hidden="true" />
            Analyze new repo
          </Link>
          <span className="font-sans text-[10.5px] tracking-[0.04em] text-[var(--ink-faint)]">
            Press <kbd className="rounded-[3px] border border-[var(--hairline)] bg-[var(--paper-panel)] px-1.5 py-px font-mono text-[10px] text-[var(--ink-muted)]">⌘K</kbd> to search
          </span>
        </footer>
      </main>
    </div>
  );
}
