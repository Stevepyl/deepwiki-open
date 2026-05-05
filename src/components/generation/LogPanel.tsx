"use client";

import { useEffect, useRef } from "react";
import type { GenerationLogLine } from "../../hooks/useGenerationPhases";

interface LogPanelProps {
  lines: GenerationLogLine[];
}

export function LogPanel({ lines }: LogPanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const panel = panelRef.current;
    if (panel) {
      panel.scrollTop = panel.scrollHeight;
    }
  }, [lines]);

  return (
    <div
      className="relative h-64 w-[600px] max-w-[90vw] overflow-y-auto rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)] px-6 py-5 shadow-[inset_0_-56px_56px_rgba(251,248,242,0.76)] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb]:bg-[var(--hairline-strong)]"
      ref={panelRef}
      role="log"
      aria-live="polite"
      aria-label="Generation progress"
    >
      <div className="flex flex-col gap-[7px]">
        {lines.map((line) => (
          <div className={`log-line log-line--${line.tone}`} key={line.id}>
            <span className="shrink-0 text-[10px] text-[var(--accent)]">▸</span>
            <span>{line.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
