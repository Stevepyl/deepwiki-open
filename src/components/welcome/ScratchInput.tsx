"use client";

import { forwardRef } from "react";
import { FiArrowRight, FiSearch } from "react-icons/fi";

interface ScratchInputProps {
  value: string;
  hint?: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export const ScratchInput = forwardRef<HTMLInputElement, ScratchInputProps>(function ScratchInput(
  { value, hint, onChange, onSubmit },
  ref,
) {
  return (
    <form
      className="relative flex w-full max-w-[640px] items-center border-b border-[var(--ink-secondary)] px-2 py-[18px] transition-colors duration-200 focus-within:border-[var(--accent)]"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <FiSearch aria-hidden="true" className="mr-3.5 h-[18px] w-[18px] shrink-0 text-[var(--ink-muted)]" />
      <input
        aria-label="Repository URL or question"
        className="min-w-0 flex-1 font-serif text-[20px] text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)] placeholder:italic"
        onChange={(event) => onChange(event.target.value)}
        placeholder="Paste a GitHub URL, or describe what you want to understand..."
        ref={ref}
        type="text"
        value={value}
      />
      <button
        aria-label="Submit repository"
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-[var(--ink-muted)] transition-all duration-150 hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] focus-visible:bg-[var(--accent-soft)] focus-visible:text-[var(--accent)]"
        type="submit"
      >
        <FiArrowRight aria-hidden="true" className="h-[18px] w-[18px]" />
      </button>
      {hint ? (
        <p className="absolute left-8 top-full mt-2 max-w-[560px] font-sans text-xs text-[var(--accent)]">
          {hint}
        </p>
      ) : null}
    </form>
  );
});
