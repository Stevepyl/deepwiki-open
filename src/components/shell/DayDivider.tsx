import type { ReactNode } from "react";

export function DayDivider({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-3 font-[var(--font-sans)] text-[11px] uppercase text-[var(--ink-muted)] before:h-px before:flex-1 before:bg-[var(--hairline)] before:content-[''] after:h-px after:flex-1 after:bg-[var(--hairline)] after:content-['']">
      {children}
    </div>
  );
}
