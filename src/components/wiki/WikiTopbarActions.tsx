"use client";

import Link from "next/link";
import { useState } from "react";
import { FiDownload, FiMonitor, FiMoreHorizontal } from "react-icons/fi";
import { IconButton } from "../shell/IconButton";

interface WikiTopbarActionsProps {
  slidesHref: string;
  onExport: (format: "markdown" | "json") => void;
}

export function WikiTopbarActions({ slidesHref, onExport }: WikiTopbarActionsProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative flex items-center gap-1">
      <Link
        className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] hover:bg-[var(--paper-panel)] hover:text-[var(--ink-primary)]"
        href={slidesHref}
        aria-label="Present as slides"
        title="Present as slides"
      >
        <FiMonitor aria-hidden="true" className="h-[15px] w-[15px]" />
      </Link>
      <IconButton aria-label="Export wiki" title="Export wiki" onClick={() => setOpen((value) => !value)}>
        <FiMoreHorizontal aria-hidden="true" />
      </IconButton>
      {open && (
        <div className="absolute right-0 top-10 z-[70] min-w-40 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)] p-1 shadow-[0_10px_24px_rgba(26,24,21,0.08)]">
          {(["markdown", "json"] as const).map((format) => (
            <button
              className="flex w-full items-center gap-2 rounded-[5px] px-3 py-2 text-left font-[var(--font-sans)] text-xs text-[var(--ink-secondary)] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
              key={format}
              type="button"
              onClick={() => {
                setOpen(false);
                onExport(format);
              }}
            >
              <FiDownload aria-hidden="true" className="h-3.5 w-3.5 text-[var(--accent)]" />
              {format === "markdown" ? "Markdown" : "JSON"}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
