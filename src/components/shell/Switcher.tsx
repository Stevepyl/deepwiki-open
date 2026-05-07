"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { IconType } from "react-icons";
import { FiBookOpen, FiEdit3, FiMessageSquare } from "react-icons/fi";

interface RepoSwitcherProps {
  owner: string;
  repo: string;
}

interface ChoiceSwitcherProps {
  ariaLabel: string;
  options: Array<{ value: string; label: string; icon: IconType }>;
  value: string;
  onValueChange: (value: string) => void;
}

type SwitcherProps = RepoSwitcherProps | ChoiceSwitcherProps;

export function Switcher(props: SwitcherProps) {
  const pathname = usePathname();

  if ("options" in props) {
    return (
      <div
        aria-label={props.ariaLabel}
        className="flex gap-0.5 rounded-[var(--radius-md)] border border-[var(--hairline)] bg-[var(--paper-panel)] p-[3px]"
        role="tablist"
      >
        {props.options.map(({ value, label, icon: Icon }) => {
          const active = props.value === value;
          return (
            <button
              aria-selected={active}
              className={`flex items-center gap-1.5 rounded-[5px] px-3 py-1.5 font-sans text-[11.5px] font-medium tracking-normal transition-all duration-150 [&>svg]:h-3 [&>svg]:w-3 ${
                active
                  ? "bg-[var(--paper-main)] text-[var(--ink-primary)] shadow-[0_1px_2px_rgba(0,0,0,0.04)]"
                  : "text-[var(--ink-muted)] hover:text-[var(--ink-primary)]"
              }`}
              key={value}
              onClick={() => props.onValueChange(value)}
              role="tab"
              type="button"
            >
              <Icon aria-hidden="true" />
              {label}
            </button>
          );
        })}
      </div>
    );
  }

  const { owner, repo } = props;
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
            className={`flex items-center gap-1.5 rounded-[5px] px-4 py-1.5 font-sans text-xs font-medium tracking-normal transition-all duration-150 [&>svg]:h-3 [&>svg]:w-3 ${
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
