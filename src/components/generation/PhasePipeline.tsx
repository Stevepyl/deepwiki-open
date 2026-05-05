import type { GenerationPhase } from "../../hooks/useGenerationPhases";

interface PhasePipelineProps {
  phase: GenerationPhase;
}

const steps: Array<{ id: GenerationPhase; label: string; title: string }> = [
  { id: "clone", label: "Clone", title: "Resolving repository..." },
  { id: "chunk", label: "Chunk", title: "Chunking source files..." },
  { id: "embed", label: "Embed", title: "Embedding source files..." },
  { id: "generate", label: "Generate", title: "Generating wiki pages..." },
];

export function phaseTitle(phase: GenerationPhase) {
  if (phase === "done") return "Wiki ready.";
  if (phase === "error") return "Generation stopped.";
  return steps.find((step) => step.id === phase)?.title ?? "Preparing wiki...";
}

export function PhasePipeline({ phase }: PhasePipelineProps) {
  const activeIndex = Math.max(0, steps.findIndex((step) => step.id === phase));
  return (
    <div className="flex flex-col items-center gap-2.5">
      <span className="font-[var(--font-sans)] text-[10.5px] font-medium uppercase tracking-normal text-[var(--ink-muted)]">
        Now running
      </span>
      <h1 className="m-0 text-center font-[var(--font-serif)] text-[32px] font-semibold italic tracking-normal text-[var(--ink-primary)]">
        {phaseTitle(phase)}
      </h1>
      <div className="mt-1 flex items-center gap-4">
        {steps.map((step, index) => {
          const complete = phase === "done" || index < activeIndex;
          const active = step.id === phase;
          return (
            <span
              className={`flex items-center gap-2 font-[var(--font-sans)] text-[11px] tracking-normal after:text-[var(--hairline-strong)] after:content-['→'] last:after:content-none ${
                complete
                  ? "text-[var(--ink-muted)] line-through decoration-[var(--hairline-strong)]"
                  : active
                    ? "font-medium text-[var(--ink-primary)]"
                    : "text-[var(--ink-faint)]"
              }`}
              key={step.id}
            >
              {step.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}
