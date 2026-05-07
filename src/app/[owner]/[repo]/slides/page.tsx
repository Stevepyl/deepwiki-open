"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FiArrowLeft } from "react-icons/fi";
import { buildRepoInfo } from "../../../../components/chat/runtime";
import { SlideNav } from "../../../../components/slides/SlideNav";
import { SlideStage } from "../../../../components/slides/SlideStage";
import type { Slide } from "../../../../components/slides/types";
import { useLanguage } from "../../../../contexts/LanguageContext";
import { useSettings } from "../../../../contexts/SettingsContext";
import { buildSlides, slidesStorageKey } from "../../../../utils/slides";
import { fetchWikiCache, repoLabel } from "../../../../utils/wiki";

export default function SlidesPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const owner = decodeURIComponent(String(params.owner));
  const repo = decodeURIComponent(String(params.repo));
  const settings = useSettings();
  const { language } = useLanguage();
  const repoInfo = useMemo(() => buildRepoInfo(owner, repo, settings.token, searchParams), [owner, repo, searchParams, settings.token]);
  const [slides, setSlides] = useState<Slide[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const basePath = `/${owner}/${repo}`;

  useEffect(() => {
    let ignore = false;
    fetchWikiCache(owner, repo, repoInfo.type, language).then((cache) => {
      if (ignore) return;
      if (!cache) {
        router.replace(`${basePath}?status=generating`);
        return;
      }
      const key = slidesStorageKey(repoInfo, language);
      const cached = window.localStorage.getItem(key);
      const nextSlides = cached ? (JSON.parse(cached) as Slide[]) : buildSlides(cache);
      if (!cached) window.localStorage.setItem(key, JSON.stringify(nextSlides));
      setSlides(nextSlides);
    });
    return () => {
      ignore = true;
    };
  }, [basePath, language, owner, repo, repoInfo, router]);

  const previous = useCallback(() => setCurrentIndex((index) => Math.max(0, index - 1)), []);
  const next = useCallback(() => setCurrentIndex((index) => Math.min(slides.length - 1, index + 1)), [slides.length]);
  const fullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
      return;
    }
    void document.documentElement.requestFullscreen();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") previous();
      if (event.key === "ArrowRight") next();
      if (event.key.toLowerCase() === "f") fullscreen();
      if (event.key === "Escape" && document.fullscreenElement) void document.exitFullscreen();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen, next, previous]);

  const current = slides[currentIndex];

  return (
    <main className="flex min-h-screen flex-col bg-[var(--paper-hover)]">
      <header className="fixed inset-x-0 top-0 z-20 flex h-12 items-center justify-between border-b border-[var(--hairline)] bg-[rgba(239,234,224,0.85)] px-8 backdrop-blur-[12px] backdrop-saturate-[110%]">
        <Link className="flex items-center gap-2.5 text-[var(--ink-secondary)] hover:text-[var(--ink-primary)]" href={basePath}>
          <FiArrowLeft aria-hidden="true" className="h-[13px] w-[13px]" />
          <span className="font-mono text-xs text-[var(--ink-primary)]">{repoLabel(owner, repo, repoInfo.type)}</span>
          <span className="text-[var(--ink-faint)]">/</span>
          <span className="font-serif text-xs italic text-[var(--ink-secondary)]">{current?.title ?? "Slides"}</span>
        </Link>
        <div className="font-sans text-[11px] font-medium uppercase tracking-normal text-[var(--ink-muted)]">
          Slide {slides.length === 0 ? 0 : currentIndex + 1} of {slides.length}
        </div>
      </header>
      {slides.length === 0 ? (
        <div className="flex min-h-screen items-center justify-center font-serif text-[16px] italic text-[var(--ink-muted)]">
          Preparing slides...
        </div>
      ) : (
        <>
          <SlideStage currentIndex={currentIndex} slides={slides} />
          <SlideNav
            currentIndex={currentIndex}
            total={slides.length}
            onFullscreen={fullscreen}
            onGoTo={setCurrentIndex}
            onNext={next}
            onPrevious={previous}
          />
        </>
      )}
    </main>
  );
}
