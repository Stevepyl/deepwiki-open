"use client";

import Link from "next/link";
import { FiMonitor, FiMoreHorizontal, FiShare2 } from "react-icons/fi";
import { IconButton } from "../shell/IconButton";
import { Switcher } from "../shell/Switcher";
import { Topbar } from "../shell/Topbar";

interface ChatTopbarProps {
  owner: string;
  repo: string;
  title: string;
  deepResearch: boolean;
  streaming: boolean;
  onDeepResearchChange: (value: boolean) => void;
}

export function ChatTopbar({ owner, repo, title, deepResearch, streaming, onDeepResearchChange }: ChatTopbarProps) {
  return (
    <Topbar
      breadcrumb={
        <>
          <span className="font-mono text-[12.5px]">{owner}/{repo}</span>
          <span className="text-[var(--ink-faint)]">/</span>
          <span className="max-w-[280px] truncate font-serif italic">{title}</span>
        </>
      }
      switcher={<Switcher owner={owner} repo={repo} />}
      actions={
        <>
          <label className="mr-2 flex items-center gap-1.5 font-sans text-[11px] text-[var(--ink-muted)]">
            <input
              type="checkbox"
              checked={deepResearch}
              disabled={streaming}
              onChange={(event) => onDeepResearchChange(event.target.checked)}
            />
            Deep research
          </label>
          <Link
            className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] hover:bg-[var(--paper-panel)] hover:text-[var(--ink-primary)] [&>svg]:h-[15px] [&>svg]:w-[15px]"
            href={`/${owner}/${repo}/slides`}
            aria-label="Open slides"
          >
            <FiMonitor aria-hidden="true" />
          </Link>
          <IconButton aria-label="Share chat" onClick={() => navigator.clipboard.writeText(window.location.href)}>
            <FiShare2 aria-hidden="true" />
          </IconButton>
          <IconButton aria-label="More chat actions">
            <FiMoreHorizontal aria-hidden="true" />
          </IconButton>
        </>
      }
    />
  );
}
