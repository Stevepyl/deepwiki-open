"use client";

import { useState } from "react";
import { FiAlertTriangle, FiCheckCircle, FiChevronRight, FiTerminal } from "react-icons/fi";
import type { ChatToolEvent } from "./types";

interface ToolEventProps {
  event: ChatToolEvent;
}

function formatDuration(durationMs?: number) {
  if (durationMs === undefined) {
    return "";
  }
  return durationMs >= 1000 ? `${(durationMs / 1000).toFixed(1)}s` : `${durationMs}ms`;
}

export function ToolEvent({ event }: ToolEventProps) {
  const [open, setOpen] = useState(false);
  const isError = event.status === "error";
  const Icon = event.status === "running" ? FiTerminal : isError ? FiAlertTriangle : FiCheckCircle;
  const summary = event.resultSummary || (event.args ? JSON.stringify(event.args) : "");

  return (
    <div className="rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)]">
      <button
        className="flex w-full items-center gap-2 px-3 py-2 text-left font-[var(--font-sans)] text-[12px] text-[var(--ink-secondary)]"
        type="button"
        onClick={() => setOpen((current) => !current)}
      >
        <FiChevronRight
          className={`h-3 w-3 text-[var(--ink-muted)] transition-transform duration-150 ${open ? "rotate-90" : ""}`}
          aria-hidden="true"
        />
        <Icon
          className={`h-3.5 w-3.5 ${event.status === "running" ? "text-[var(--accent)]" : isError ? "text-[#9F2F1F]" : "text-[#3F6F4A]"}`}
          aria-hidden="true"
        />
        <span className="font-[var(--font-mono)]">{event.toolName}</span>
        <span className="ml-auto text-[11px] text-[var(--ink-muted)]">
          {event.status === "running" ? "running" : isError ? "error" : "complete"} {formatDuration(event.durationMs)}
        </span>
      </button>
      {open && summary ? (
        <pre className="max-h-40 overflow-auto border-t border-[var(--hairline)] px-3 py-2 font-[var(--font-mono)] text-[11.5px] leading-relaxed text-[var(--ink-secondary)] whitespace-pre-wrap">
          {summary}
        </pre>
      ) : null}
    </div>
  );
}
