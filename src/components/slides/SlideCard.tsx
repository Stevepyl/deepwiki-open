import Markdown from "../Markdown";
import type { Slide } from "./types";

interface SlideCardProps {
  slide: Slide;
  stub?: boolean;
}

export function SlideCard({ slide, stub = false }: SlideCardProps) {
  const baseClass = `slide-card flex aspect-video w-[960px] max-w-[90vw] flex-col rounded-[var(--radius-md)] border border-[var(--hairline-strong)] bg-[var(--paper-main)] px-16 py-14 shadow-[0_4px_32px_rgba(26,24,21,0.07),0_1px_4px_rgba(26,24,21,0.04)] ${
    stub ? "pointer-events-none select-none" : ""
  }`;

  if (slide.variant === "divider") {
    return (
      <div className={`${baseClass} items-center justify-center gap-5 text-center`}>
        <div className="font-[var(--font-mono)] text-[11px] uppercase tracking-normal text-[var(--ink-muted)]">
          {slide.eyebrow}
        </div>
        <h1 className="m-0 text-balance font-[var(--font-serif)] text-[44px] font-semibold leading-[1.1] tracking-normal text-[var(--ink-primary)]">
          {slide.title}
        </h1>
        {slide.subtitle && (
          <p className="m-0 font-[var(--font-serif)] text-[19px] italic text-[var(--ink-secondary)]">{slide.subtitle}</p>
        )}
      </div>
    );
  }

  return (
    <div className={`${baseClass} gap-6 overflow-hidden`}>
      {slide.eyebrow && (
        <div className="flex items-center gap-2 font-[var(--font-sans)] text-[10.5px] font-medium uppercase tracking-normal text-[var(--accent)] before:h-px before:w-4 before:bg-[var(--accent-line)] before:content-['']">
          {slide.eyebrow}
        </div>
      )}
      <h2 className="m-0 font-[var(--font-serif)] text-[32px] font-semibold leading-[1.15] tracking-normal text-[var(--ink-primary)]">
        {slide.title}
      </h2>
      <div className="slide-body min-h-0 flex-1 overflow-hidden text-[var(--ink-primary)]">
        <Markdown content={slide.content ?? ""} />
      </div>
    </div>
  );
}
