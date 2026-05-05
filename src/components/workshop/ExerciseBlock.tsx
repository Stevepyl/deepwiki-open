interface ExerciseBlockProps {
  task: string;
  hint: string;
}

export function ExerciseBlock({ task, hint }: ExerciseBlockProps) {
  return (
    <div className="workshop-exercise my-7 flex flex-col gap-2.5 rounded-r-[var(--radius-sm)] border border-l-2 border-[var(--hairline)] border-l-[var(--accent)] bg-[var(--paper-panel)] px-6 py-5">
      <div className="flex items-center gap-2 font-[var(--font-sans)] text-[10.5px] font-medium uppercase tracking-normal text-[var(--accent)] after:text-xs after:content-['→']">
        Try it
      </div>
      <div className="font-[var(--font-mono)] text-[13.5px] leading-[1.55] text-[var(--ink-primary)]">{task}</div>
      <div className="font-[var(--font-serif)] text-[13.5px] italic leading-[1.5] text-[var(--ink-muted)]">{hint}</div>
    </div>
  );
}
