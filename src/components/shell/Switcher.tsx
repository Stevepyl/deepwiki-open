"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FiBookOpen, FiEdit3, FiMessageSquare } from "react-icons/fi";

interface SwitcherProps {
  owner: string;
  repo: string;
}

export function Switcher({ owner, repo }: SwitcherProps) {
  const pathname = usePathname();
  const base = `/${owner}/${repo}`;
  const items = [
    { label: "Chat", href: `${base}/ask`, icon: FiMessageSquare },
    { label: "Wiki", href: base, icon: FiBookOpen },
    { label: "Workshop", href: `${base}/workshop`, icon: FiEdit3 },
  ];

  return (
    <nav
      className="flex gap-0.5 rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--paper-panel)] p-[3px]"
      aria-label="Repository workspace"
    >
      {items.map(({ label, href, icon: Icon }) => {
        const active = pathname === href;
        return (
          <Link
            className={`flex items-center gap-1.5 rounded-[5px] px-4 py-1.5 font-[var(--font-sans)] text-xs font-medium tracking-normal transition-all duration-150 [&>svg]:h-3 [&>svg]:w-3 ${
              active
                ? "bg-[var(--paper-main)] text-[var(--ink-primary)] shadow-[0_1px_2px_rgba(0,0,0,0.04)]"
                : "text-[var(--ink-muted)] hover:text-[var(--ink-primary)]"
            }`}
            href={href}
            key={href}
          >
            <Icon aria-hidden="true" />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
