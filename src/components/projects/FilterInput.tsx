"use client";

import { forwardRef } from "react";
import { FiSearch } from "react-icons/fi";

interface FilterInputProps {
  value: string;
  onChange: (value: string) => void;
}

export const FilterInput = forwardRef<HTMLInputElement, FilterInputProps>(function FilterInput(
  { value, onChange },
  ref,
) {
  return (
    <label className="flex w-full max-w-[400px] items-center gap-2.5 border-b border-[var(--ink-secondary)] px-3 py-2 transition-colors duration-150 focus-within:border-[var(--accent)]">
      <FiSearch aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-[var(--ink-muted)]" />
      <input
        aria-label="Filter repositories"
        className="min-w-0 flex-1 font-[var(--font-serif)] text-[15px] italic text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)]"
        onChange={(event) => onChange(event.target.value)}
        placeholder="Filter repositories..."
        ref={ref}
        type="text"
        value={value}
      />
    </label>
  );
});
