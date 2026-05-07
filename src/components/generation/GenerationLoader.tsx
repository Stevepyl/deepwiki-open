"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { FiArrowLeft } from "react-icons/fi";
import type { SettingsContextValue } from "../../contexts/SettingsContext";
import { useGenerationPhases } from "../../hooks/useGenerationPhases";
import type RepoInfo from "../../types/repoinfo";
import { generateWikiCache } from "../../utils/wikiGeneration";
import getRepoUrl from "../../utils/getRepoUrl";
import { repoLabel, saveWikiCache } from "../../utils/wiki";
import { LogPanel } from "./LogPanel";
import { PhasePipeline } from "./PhasePipeline";

interface GenerationLoaderProps {
  owner: string;
  repo: string;
  repoInfo: RepoInfo;
  language: string;
  settings: SettingsContextValue;
}

export function GenerationLoader({ owner, repo, repoInfo, language, settings }: GenerationLoaderProps) {
  const router = useRouter();
  const phases = useGenerationPhases();
  const { complete, fail, ingest } = phases;
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!settings.hydrated) return;
    let cancelled = false;
    const startTimer = window.setTimeout(() => {
      void run();
    }, 0);

    async function run() {
      try {
        if (cancelled) return;
        ingest(`Cloning repository ${getRepoUrl(repoInfo)}`);
        ingest("Chunking repository files");
        ingest("Embedding chunk 1/2");
        ingest("Generating wiki structure");
        const cache = await generateWikiCache(repoInfo, language, settings, {
          onSocket: (socket) => {
            socketRef.current = socket;
          },
          onPageStart: (index, total, page) => {
            ingest(`Streaming wiki page ${index}/${total}: ${page.title}`);
          },
        });
        if (cancelled) return;
        await saveWikiCache(cache, repoInfo, language);
        complete();
        window.setTimeout(() => router.replace(`/${owner}/${repo}`), 650);
      } catch (error) {
        if (!cancelled) {
          fail(error instanceof Error ? error.message : "Wiki generation failed");
        }
      }
    }

    return () => {
      cancelled = true;
      window.clearTimeout(startTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [complete, fail, ingest, language, owner, repo, repoInfo, router, settings]);

  const cancel = () => {
    socketRef.current?.close();
    router.push("/");
  };

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center bg-[var(--paper-main)] px-8 py-16">
      <div className="absolute left-8 top-7 flex items-center gap-2.5 text-[var(--ink-muted)]">
        <Link
          className="flex items-center gap-1.5 font-sans text-[11px] text-[var(--ink-faint)] transition-colors duration-[120ms] hover:text-[var(--ink-secondary)]"
          href="/"
        >
          <FiArrowLeft aria-hidden="true" className="h-3 w-3" />
          Back
        </Link>
        <span className="text-[11px] text-[var(--ink-faint)]">/</span>
        <span className="font-mono text-xs tracking-normal text-[var(--ink-secondary)]">
          {repoLabel(owner, repo, repoInfo.type)}
        </span>
      </div>

      <div className="mb-10">
        <PhasePipeline phase={phases.phase} />
      </div>
      <LogPanel lines={phases.lines} />
      <div className="mt-5 flex w-[600px] max-w-[90vw] items-center gap-3">
        <div className="h-px flex-1 overflow-hidden rounded bg-[var(--hairline)]">
          <div className="hairline-progress h-full rounded bg-[var(--accent)]" style={{ width: `${phases.percent}%` }} />
        </div>
        <span className="w-10 text-right font-mono text-[10.5px] text-[var(--ink-muted)]">
          {phases.percent}%
        </span>
      </div>
      <p className="mt-7 max-w-[420px] text-center font-serif text-[14.5px] italic leading-[1.6] text-[var(--ink-muted)]">
        This usually takes 2-4 minutes for a repository of this size. You can leave this tab; the wiki will be ready when you return.
      </p>
      {phases.error && <p className="mt-3 max-w-[520px] text-center text-xs text-[var(--accent)]">{phases.error}</p>}
      <button
        className="mt-4 font-sans text-[11px] uppercase tracking-normal text-[var(--ink-faint)] transition-colors duration-[120ms] hover:text-[var(--ink-secondary)]"
        type="button"
        onClick={cancel}
      >
        Cancel
      </button>
    </main>
  );
}
