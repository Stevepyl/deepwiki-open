"use client";

import type { FormEvent, ReactNode } from "react";
import { FiArrowUp, FiPaperclip, FiSettings } from "react-icons/fi";
import { shellMessages } from "../../messages/en";

interface ComposerProps {
  variant: "chat" | "wiki";
  modeHint: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onOpenSettings: () => void;
  footer?: ReactNode;
}

const noiseBackground =
  "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3CfeColorMatrix values='0 0 0 0 0.1 0 0 0 0 0.09 0 0 0 0 0.08 0 0 0 0.03 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")";

export function Composer({
  variant,
  modeHint,
  placeholder,
  value,
  onChange,
  onSubmit,
  onOpenSettings,
  footer,
}: ComposerProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (value.trim()) {
      onSubmit();
    }
  };

  return (
    <>
      <form
        className={`fixed bottom-[var(--composer-bottom)] left-[calc(var(--sidebar-w)_+_((100vw_-_var(--sidebar-w))_/_2))] z-50 flex min-h-[var(--composer-h)] w-[calc(100%_-_var(--sidebar-w)_-_80px)] max-w-[var(--composer-max-w)] -translate-x-1/2 items-center gap-3 rounded-[var(--radius-lg)] border border-[var(--hairline-strong)] px-2.5 py-0 pl-5 transition-[box-shadow,background,border-color] duration-[250ms] ${
          variant === "wiki"
            ? "bg-[rgba(251,248,242,0.88)] shadow-[0_-2px_4px_rgba(0,0,0,0.015),0_8px_24px_rgba(26,24,21,0.06)] backdrop-blur-[20px] backdrop-saturate-[120%]"
            : "bg-[var(--paper-main)] shadow-[0_2px_8px_rgba(0,0,0,0.03)]"
        }`}
        onSubmit={handleSubmit}
      >
        {variant === "wiki" && (
          <span
            className="pointer-events-none absolute inset-0 rounded-[var(--radius-lg)] opacity-40 mix-blend-multiply"
            style={{ backgroundImage: noiseBackground }}
          />
        )}
        <span
          className={`relative z-[1] shrink-0 rounded border border-[var(--hairline)] px-2.5 py-1 font-[var(--font-sans)] text-[10.5px] uppercase tracking-normal text-[var(--ink-muted)] ${
            variant === "wiki" ? "bg-[var(--paper-main)]" : "bg-[var(--paper-panel)]"
          }`}
        >
          {modeHint}
        </span>
        <input
          className="relative z-[1] min-w-0 flex-1 py-[18px] font-[var(--font-serif)] text-base text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)] placeholder:italic"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
        />
        <div className="relative z-[1] flex shrink-0 items-center gap-1">
          <button
            className="flex h-[34px] w-[34px] items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] enabled:hover:bg-[var(--paper-hover)] enabled:hover:text-[var(--ink-primary)] [&>svg]:h-[15px] [&>svg]:w-[15px]"
            disabled
            type="button"
            aria-label={shellMessages.composer.attach}
          >
            <FiPaperclip aria-hidden="true" />
          </button>
          <button
            className="flex h-[34px] w-[34px] items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] enabled:hover:bg-[var(--paper-hover)] enabled:hover:text-[var(--ink-primary)] [&>svg]:h-[15px] [&>svg]:w-[15px]"
            type="button"
            aria-label={shellMessages.composer.settings}
            onClick={onOpenSettings}
          >
            <FiSettings aria-hidden="true" />
          </button>
          <button
            className="ml-0.5 flex h-[38px] w-[38px] items-center justify-center rounded-full text-[var(--accent)] transition-all duration-150 enabled:hover:bg-[var(--accent-soft)] [&>svg]:h-4 [&>svg]:w-4"
            disabled={!value.trim()}
            type="submit"
            aria-label={shellMessages.composer.send}
          >
            <FiArrowUp aria-hidden="true" />
          </button>
        </div>
      </form>
      {footer && (
        <div className="pointer-events-none fixed bottom-2 left-[calc(var(--sidebar-w)_+_((100vw_-_var(--sidebar-w))_/_2))] z-[49] flex w-[calc(100%_-_var(--sidebar-w)_-_80px)] max-w-[var(--composer-max-w)] -translate-x-1/2 justify-between gap-5 px-2 font-[var(--font-sans)] text-[10.5px] text-[var(--ink-faint)] [&_kbd]:rounded-[3px] [&_kbd]:border [&_kbd]:border-[var(--hairline)] [&_kbd]:bg-[var(--paper-panel)] [&_kbd]:px-[5px] [&_kbd]:py-px [&_kbd]:font-[var(--font-mono)] [&_kbd]:text-[10px] [&_kbd]:text-[var(--ink-muted)]">
          {footer}
        </div>
      )}
    </>
  );
}
