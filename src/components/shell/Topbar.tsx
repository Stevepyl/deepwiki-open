import type { ReactNode } from "react";

interface TopbarProps {
  breadcrumb: ReactNode;
  switcher?: ReactNode;
  actions?: ReactNode;
}

export function Topbar({ breadcrumb, switcher, actions }: TopbarProps) {
  return (
    <header className="flex h-[var(--topbar-h)] shrink-0 items-center justify-between border-b border-[var(--hairline)] bg-[var(--paper-main)] px-8">
      <div className="flex items-center gap-2.5 text-[var(--ink-secondary)]">{breadcrumb}</div>
      {switcher}
      <div className="flex items-center gap-1">{actions}</div>
    </header>
  );
}
