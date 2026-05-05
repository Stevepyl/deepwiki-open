import { SlideCard } from "./SlideCard";
import type { Slide } from "./types";

interface SlideStageProps {
  currentIndex: number;
  slides: Slide[];
}

export function SlideStage({ currentIndex, slides }: SlideStageProps) {
  const previous = slides[currentIndex - 1];
  const current = slides[currentIndex];
  const next = slides[currentIndex + 1];

  return (
    <main className="flex min-h-screen flex-1 flex-col items-center justify-center gap-5 px-8 pb-[120px] pt-20">
      {previous && (
        <div className="w-[900px] max-w-[90vw] scale-[0.94] opacity-25 transition-all duration-200">
          <SlideCard slide={previous} stub />
        </div>
      )}
      {current && <SlideCard slide={current} />}
      {next && (
        <div className="w-[900px] max-w-[90vw] scale-[0.94] opacity-25 transition-all duration-200">
          <SlideCard slide={next} stub />
        </div>
      )}
    </main>
  );
}
