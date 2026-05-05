"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

export type GenerationPhase = "clone" | "chunk" | "embed" | "generate" | "done" | "error";

export interface GenerationLogLine {
  id: string;
  text: string;
  tone: "done" | "active" | "dim" | "error";
}

const phaseOrder: GenerationPhase[] = ["clone", "chunk", "embed", "generate"];

function lineId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function phaseFromText(text: string, currentPercent: number) {
  if (/Cloning repository/i.test(text)) {
    return { phase: "clone" as const, percent: Math.max(currentPercent, 6), tone: "done" as const };
  }
  if (/Chunking\s+(\d+)?\s*files?/i.test(text) || /Chunking repository files/i.test(text)) {
    return { phase: "chunk" as const, percent: Math.max(currentPercent, 22), tone: "done" as const };
  }
  const embed = text.match(/Embedding chunk\s+(\d+)\s*\/\s*(\d+)/i);
  if (embed) {
    const current = Number(embed[1]);
    const total = Math.max(Number(embed[2]), 1);
    return { phase: "embed" as const, percent: Math.max(currentPercent, Math.round((current / total) * 50)), tone: "active" as const };
  }
  if (/Generating wiki structure/i.test(text)) {
    return { phase: "generate" as const, percent: Math.max(currentPercent, 52), tone: "active" as const };
  }
  const page = text.match(/Streaming wiki page\s+(\d+)\s*\/\s*(\d+)/i);
  if (page) {
    const current = Number(page[1]);
    const total = Math.max(Number(page[2]), 1);
    return { phase: "generate" as const, percent: Math.min(99, Math.round(50 + (current / total) * 50)), tone: "active" as const };
  }
  return { phase: undefined, percent: currentPercent, tone: "dim" as const };
}

export function useGenerationPhases(stream?: ReadableStream<string>) {
  const [phase, setPhase] = useState<GenerationPhase>("clone");
  const [percent, setPercent] = useState(4);
  const [lines, setLines] = useState<GenerationLogLine[]>([]);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");

  const ingest = useCallback((text: string) => {
    const clean = text.trim();
    if (!clean) return;
    setPercent((currentPercent) => {
      const next = phaseFromText(clean, currentPercent);
      if (next.phase) setPhase(next.phase);
      setLines((current) => [...current.slice(-80), { id: lineId(), text: clean, tone: next.tone }]);
      return next.percent;
    });
  }, []);

  const complete = useCallback((text = "Wiki cache saved successfully") => {
    setPhase("done");
    setPercent(100);
    setDone(true);
    setLines((current) => [...current.slice(-80), { id: lineId(), text, tone: "done" }]);
  }, []);

  const fail = useCallback((message: string) => {
    setPhase("error");
    setError(message);
    setLines((current) => [...current.slice(-80), { id: lineId(), text: message, tone: "error" }]);
  }, []);

  useEffect(() => {
    if (!stream) return;
    let cancelled = false;
    const reader = stream.getReader();
    async function read() {
      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;
        ingest(value);
      }
      if (!cancelled) complete();
    }
    void read().catch((streamError: unknown) => {
      fail(streamError instanceof Error ? streamError.message : "Generation stream failed");
    });
    return () => {
      cancelled = true;
      void reader.cancel();
    };
  }, [complete, fail, ingest, stream]);

  const phaseIndex = useMemo(() => Math.max(0, phaseOrder.indexOf(phase)), [phase]);

  return { phase, phaseIndex, lines, percent, done, error, ingest, complete, fail };
}
