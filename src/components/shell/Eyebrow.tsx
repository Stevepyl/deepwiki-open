import type { ReactNode } from "react";

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <span className="font-sans text-[11px] font-medium uppercase tracking-normal text-[var(--ink-muted)]">
      {children}
    </span>
  );
}
