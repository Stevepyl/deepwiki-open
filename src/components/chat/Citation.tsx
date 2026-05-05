interface CitationProps {
  index: number;
  path: string;
}

export function Citation({ index, path }: CitationProps) {
  return (
    <a
      className="inline-flex max-w-full items-center gap-1.5 rounded border border-[var(--hairline)] bg-[var(--paper-panel)] px-2 py-1 font-[var(--font-mono)] text-[11.5px] text-[var(--ink-secondary)] transition-all duration-[120ms] hover:border-[var(--accent-line)] hover:text-[var(--accent)]"
      href={`#source-${index}`}
      title={path}
    >
      <span className="text-[var(--ink-muted)]">{index}</span>
      <span className="truncate">{path}</span>
    </a>
  );
}
