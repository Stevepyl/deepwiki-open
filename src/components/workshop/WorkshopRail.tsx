"use client";

import Link from "next/link";
import type { WorkshopStep } from "./types";

interface WorkshopRailProps {
  activeStepId: string;
  basePath: string;
  steps: WorkshopStep[];
}

export function WorkshopRail({ activeStepId, basePath, steps }: WorkshopRailProps) {
  const activeIndex = Math.max(0, steps.findIndex((step) => step.id === activeStepId));
  return (
    <nav className="sticky top-10 max-h-[calc(100vh_-_var(--topbar-h)_-_80px)] self-start overflow-y-auto">
      <div className="mb-5 flex items-center gap-2.5 font-[var(--font-sans)] text-[10.5px] font-medium uppercase tracking-normal text-[var(--ink-muted)] before:text-xs before:text-[var(--accent)] before:content-['◌']">
        Progress
      </div>
      <ul className="relative m-0 flex list-none flex-col gap-0.5 pl-8 before:absolute before:bottom-3 before:left-3 before:top-3 before:w-px before:bg-[var(--hairline)] before:content-['']">
        {steps.map((step, index) => {
          const active = index === activeIndex;
          const done = index < activeIndex;
          return (
            <li className="relative" key={step.id}>
              <span
                className={`absolute -left-8 top-2 z-[1] flex h-[22px] w-[22px] items-center justify-center rounded-full border font-[var(--font-mono)] text-[10px] font-medium ${
                  done
                    ? "border-[var(--accent-line)] bg-[var(--accent-soft)] text-[var(--accent)]"
                    : active
                      ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--paper-main)]"
                      : "border-[var(--hairline)] bg-[var(--paper-panel)] text-[var(--ink-muted)]"
                }`}
              >
                {done ? "✓" : String(index + 1).padStart(2, "0")}
              </span>
              <Link
                className={`block rounded-[var(--radius-sm)] px-2.5 py-[7px] font-[var(--font-serif)] text-[13.5px] leading-[1.4] transition-all duration-[120ms] ${
                  active
                    ? "bg-[var(--accent-soft)] text-[var(--accent)] italic"
                    : "text-[var(--ink-secondary)] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
                }`}
                href={`${basePath}?step=${encodeURIComponent(step.id)}`}
              >
                {step.title}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
