"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { FiPlus, FiSearch, FiTrash2 } from "react-icons/fi";
import { useConversationHistory, type RepoEntry } from "../../hooks/useConversationHistory";
import { shellMessages } from "../../messages/en";
import { Wordmark } from "./Wordmark";

function repoKey(entry: RepoEntry) {
  return `${entry.type}:${entry.owner}/${entry.repo}`;
}

function formatMeta(timestamp: number, messageCount: number) {
  const date = new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(timestamp);
  return `${date} · ${messageCount} msg`;
}

export function Sidebar() {
  const { repos, removeConversation } = useConversationHistory();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [query, setQuery] = useState("");
  const [openRepos, setOpenRepos] = useState<Record<string, boolean>>({});
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const newChatHref = useMemo(() => {
    const [owner, repo, section] = pathname.split("/").filter(Boolean);
    if (!owner || !repo || section !== "ask") return null;
    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.delete("convId");
    nextParams.delete("q");
    const queryString = nextParams.toString();
    return `/${owner}/${repo}/ask${queryString ? `?${queryString}` : ""}`;
  }, [pathname, searchParams]);

  const visibleRepos = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return repos;
    }
    return repos
      .map((repo) => ({
        ...repo,
        convs: repo.convs.filter((conv) => conv.title.toLowerCase().includes(normalizedQuery)),
      }))
      .filter((repo) => `${repo.owner}/${repo.repo}`.toLowerCase().includes(normalizedQuery) || repo.convs.length > 0);
  }, [query, repos]);

  return (
    <aside className="flex w-[var(--sidebar-w)] shrink-0 flex-col overflow-hidden border-r border-[var(--hairline)] bg-[var(--paper-panel)]">
      <div className="border-b border-[var(--hairline)] px-7 pb-5 pt-6">
        <Link href="/" aria-label={shellMessages.sidebar.homeLabel}>
          <Wordmark size="sidebar" />
        </Link>
      </div>

      <label className="mx-5 mb-2 mt-4 flex items-center gap-2.5 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-main)] px-3 py-2 transition-colors duration-150 focus-within:border-[var(--hairline-strong)] [&>svg]:h-3.5 [&>svg]:w-3.5 [&>svg]:text-[var(--ink-muted)]">
        <FiSearch aria-hidden="true" />
        <input
          className="min-w-0 flex-1 text-[13px] text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)]"
          ref={searchRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={shellMessages.sidebar.searchPlaceholder}
        />
        <kbd className="rounded-[3px] bg-[var(--paper-hover)] px-[5px] py-0.5 font-mono text-[10px] text-[var(--ink-muted)]">⌘K</kbd>
      </label>

      {newChatHref ? (
        <Link
          className="mx-5 mb-2 flex items-center gap-2.5 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-main)] px-3 py-2 font-sans text-[13px] font-medium text-[var(--ink-primary)] transition-colors duration-[120ms] hover:border-[var(--hairline-strong)] hover:bg-[var(--paper-hover)]"
          href={newChatHref}
        >
          <FiPlus className="h-3.5 w-3.5 text-[var(--accent)]" aria-hidden="true" />
          {shellMessages.sidebar.newChat}
        </Link>
      ) : null}

      <div className="flex-1 overflow-y-auto px-0 pb-5 pt-3 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb]:bg-[var(--hairline-strong)]">
        {visibleRepos.map((repo) => {
          const key = repoKey(repo);
          const isOpen = openRepos[key] ?? true;
          return (
            <section className="mb-3.5 px-3.5" key={key}>
              <button
                className="flex w-full cursor-pointer items-center gap-2 rounded-[var(--radius-sm)] px-2.5 py-2 text-[var(--ink-secondary)] transition-colors duration-[120ms] hover:bg-[var(--paper-hover)]"
                type="button"
                onClick={() => setOpenRepos((current) => ({ ...current, [key]: !isOpen }))}
              >
                <span className="w-2.5 shrink-0 font-mono text-[9px] text-[var(--ink-muted)]">{isOpen ? "▾" : "▸"}</span>
                <span className="min-w-0 flex-1 truncate text-left font-mono text-[12.5px] font-medium text-[var(--ink-primary)]">
                  {repo.owner}/{repo.repo}
                </span>
                <span className="rounded-[10px] bg-[var(--paper-hover)] px-1.5 py-0.5 font-sans text-[10px] leading-none text-[var(--ink-muted)]">
                  {repo.convs.length}
                </span>
              </button>
              {isOpen && (
                <div className="relative mt-1 pl-[18px] before:absolute before:bottom-1 before:left-2.5 before:top-1 before:w-px before:bg-[var(--hairline)] before:content-['']">
                  {repo.convs.map((conv) => (
                    <div
                      className="group relative block w-full cursor-pointer rounded-[var(--radius-sm)] px-2.5 py-2 text-left text-[var(--ink-secondary)] transition-colors duration-[120ms] hover:bg-[var(--paper-hover)]"
                      key={conv.id}
                    >
                      <Link
                        className="flex flex-col gap-0.5 pr-5"
                        href={`/${repo.owner}/${repo.repo}/ask?convId=${conv.id}`}
                      >
                        <span className="truncate font-serif text-[13.5px] font-normal leading-[1.35] text-[var(--ink-primary)]">
                          {conv.title}
                        </span>
                        <span className="font-sans text-[11px] text-[var(--ink-muted)]">
                          {formatMeta(conv.lastMessageAt, conv.messageCount)}
                        </span>
                      </Link>
                      <button
                        aria-label={shellMessages.sidebar.deleteConversation(conv.title)}
                        className="absolute right-1.5 top-[7px] flex h-5 w-5 items-center justify-center rounded text-[var(--ink-muted)] opacity-0 transition-opacity group-hover:opacity-100 hover:bg-[var(--paper-main)] hover:text-[var(--accent)]"
                        type="button"
                        onClick={() => removeConversation(conv.id)}
                      >
                        <FiTrash2 aria-hidden="true" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </div>

      <footer className="flex items-center justify-between border-t border-[var(--hairline)] px-5 pb-5 pt-3">
        <Link
          className="flex items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 font-sans text-[12.5px] font-medium text-[var(--ink-primary)] transition-colors duration-[120ms] hover:bg-[var(--paper-hover)]"
          href="/"
        >
          <FiPlus className="h-3.5 w-3.5 text-[var(--accent)]" aria-hidden="true" />
          {shellMessages.sidebar.newRepo}
        </Link>
        <div
          className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--ink-secondary)] font-serif text-[13px] font-medium text-[var(--paper-main)]"
          aria-label={shellMessages.sidebar.currentUser}
        >
          S
        </div>
      </footer>
    </aside>
  );
}
