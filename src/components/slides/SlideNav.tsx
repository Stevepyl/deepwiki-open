"use client";

import { FiChevronLeft, FiChevronRight, FiMaximize2 } from "react-icons/fi";

interface SlideNavProps {
  currentIndex: number;
  total: number;
  onFullscreen: () => void;
  onGoTo: (index: number) => void;
  onNext: () => void;
  onPrevious: () => void;
}

export function SlideNav({ currentIndex, total, onFullscreen, onGoTo, onNext, onPrevious }: SlideNavProps) {
  return (
    <>
      <nav
        className="fixed inset-x-0 bottom-0 z-20 flex h-[72px] items-center justify-center gap-6 border-t border-[var(--hairline)] bg-[rgba(239,234,224,0.85)] backdrop-blur-[12px] backdrop-saturate-[110%]"
        aria-label="Slide navigation"
      >
        <button
          className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-main)] text-[var(--ink-secondary)] transition-all duration-[120ms] hover:border-[var(--hairline-strong)] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
          type="button"
          onClick={onPrevious}
          aria-label="Previous slide"
        >
          <FiChevronLeft aria-hidden="true" className="h-3.5 w-3.5" />
        </button>
        <span className="min-w-10 text-center font-[var(--font-sans)] text-[11px] font-medium tabular-nums tracking-normal text-[var(--ink-muted)]">
          {currentIndex + 1} / {total}
        </span>
        <div className="flex items-center gap-[5px]" role="tablist" aria-label="Slides">
          {Array.from({ length: total }, (_, index) => (
            <button
              className={`h-[5px] rounded-full transition-all duration-150 hover:scale-125 hover:bg-[var(--ink-muted)] ${
                index === currentIndex ? "w-3.5 bg-[var(--accent)]" : "w-[5px] bg-[var(--ink-faint)]"
              }`}
              key={index}
              role="tab"
              aria-selected={index === currentIndex}
              aria-label={`Slide ${index + 1}`}
              type="button"
              onClick={() => onGoTo(index)}
            />
          ))}
        </div>
        <button
          className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-main)] text-[var(--ink-secondary)] transition-all duration-[120ms] hover:border-[var(--hairline-strong)] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
          type="button"
          onClick={onNext}
          aria-label="Next slide"
        >
          <FiChevronRight aria-hidden="true" className="h-3.5 w-3.5" />
        </button>
      </nav>
      <button
        className="fixed bottom-[18px] right-6 z-[21] flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
        type="button"
        onClick={onFullscreen}
        aria-label="Fullscreen"
      >
        <FiMaximize2 aria-hidden="true" className="h-3.5 w-3.5" />
      </button>
      <div className="fixed bottom-[26px] right-16 z-[21] font-[var(--font-sans)] text-[10px] tracking-normal text-[var(--ink-faint)]">
        <kbd className="rounded border border-[var(--hairline)] bg-[var(--paper-panel)] px-[5px] py-px font-[var(--font-mono)] text-[9px] text-[var(--ink-muted)]">
          ←
        </kbd>{" "}
        <kbd className="rounded border border-[var(--hairline)] bg-[var(--paper-panel)] px-[5px] py-px font-[var(--font-mono)] text-[9px] text-[var(--ink-muted)]">
          →
        </kbd>{" "}
        to navigate ·{" "}
        <kbd className="rounded border border-[var(--hairline)] bg-[var(--paper-panel)] px-[5px] py-px font-[var(--font-mono)] text-[9px] text-[var(--ink-muted)]">
          F
        </kbd>{" "}
        fullscreen
      </div>
    </>
  );
}
