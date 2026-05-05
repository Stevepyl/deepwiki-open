import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

interface AppShellProps {
  topbar: ReactNode;
  children: ReactNode;
  composer?: ReactNode;
}

export function AppShell({ topbar, children, composer }: AppShellProps) {
  return (
    <div className="flex h-screen w-full">
      <Sidebar />
      <main className="relative flex min-w-0 flex-1 flex-col bg-[var(--paper-main)] after:pointer-events-none after:absolute after:inset-x-0 after:bottom-0 after:z-40 after:h-[180px] after:bg-[linear-gradient(to_bottom,rgba(245,241,234,0)_0%,rgba(245,241,234,0.6)_35%,rgba(245,241,234,1)_75%)] after:content-['']">
        {topbar}
        <div className="flex-1 overflow-y-auto pb-[calc(var(--composer-h)+var(--composer-bottom)+160px)] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb]:bg-[var(--hairline-strong)]">
          {children}
        </div>
        {composer}
      </main>
    </div>
  );
}
