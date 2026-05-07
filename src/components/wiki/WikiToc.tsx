"use client";

import Link from "next/link";
import type { WikiSectionView } from "../../utils/wiki";

interface WikiTocProps {
  activePageId: string;
  basePath: string;
  sections: WikiSectionView[];
}

export function WikiToc({ activePageId, basePath, sections }: WikiTocProps) {
  return (
    <nav className="wiki-toc sticky top-10 max-h-[calc(100vh-var(--topbar-h)-80px)] self-start overflow-y-auto border-r border-[var(--hairline)] pr-6">
      <div className="mb-4 flex items-center gap-2.5 font-sans text-[10.5px] font-medium uppercase tracking-normal text-[var(--ink-muted)] before:text-xs before:text-[var(--accent)] before:content-['◌']">
        Contents
      </div>
      <ul className="m-0 list-none p-0">
        {sections.map((section, index) => (
          <li
            className={`wiki-tree__section ${index === 0 ? "" : "mt-[18px] border-t border-[var(--hairline)] pt-3.5"}`}
            key={section.id}
          >
            <div className="mb-1.5 px-2.5 font-sans text-[10.5px] font-medium uppercase tracking-normal text-[var(--ink-muted)]">
              {section.title}
            </div>
            <ul className="m-0 list-none p-0">
              {section.pages.map((page) => {
                const active = page.id === activePageId;
                return (
                  <li className="mb-0.5" key={page.id}>
                    <Link
                      className={`block rounded-[var(--radius-sm)] px-2.5 py-1.5 font-serif text-[13.5px] leading-[1.4] transition-all duration-[120ms] ${
                        active
                          ? "bg-[var(--accent-soft)] text-[var(--accent)] italic"
                          : "text-[var(--ink-secondary)] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
                      }`}
                      href={`${basePath}?page=${encodeURIComponent(page.id)}`}
                    >
                      {page.title}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </li>
        ))}
      </ul>
    </nav>
  );
}
