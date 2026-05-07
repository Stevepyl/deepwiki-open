import Markdown from "../Markdown";
import { ExerciseBlock } from "./ExerciseBlock";
import type { WorkshopStep } from "./types";

interface WorkshopArticleProps {
  activeStepId: string;
  steps: WorkshopStep[];
}

export function WorkshopArticle({ activeStepId, steps }: WorkshopArticleProps) {
  const activeIndex = Math.max(0, steps.findIndex((step) => step.id === activeStepId));
  return (
    <article className="min-w-0 pb-10">
      {steps.map((step, index) => {
        const done = index < activeIndex;
        return (
          <section className={done ? "opacity-45" : ""} id={step.id} key={step.id}>
            <div className="mb-3 mt-12 flex items-baseline gap-3 first:mt-0">
              <span className="shrink-0 font-mono text-[10.5px] font-medium uppercase tracking-normal text-[var(--accent)]">
                {done ? `${String(index + 1).padStart(2, "0")} - done` : `Step ${String(index + 1).padStart(2, "0")}`}
              </span>
              <h2 className="m-0 font-serif text-2xl font-semibold leading-[1.2] tracking-normal text-[var(--ink-primary)]">
                {step.title}
              </h2>
            </div>
            <Markdown content={step.body} />
            <ExerciseBlock task={step.exercise} hint={step.hint} />
          </section>
        );
      })}
    </article>
  );
}
