"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "opswiki.conversationHistory";
const HISTORY_EVENT = "opswiki:conversation-history";
const MAX_STORAGE_BYTES = 4 * 1024 * 1024;

export interface ConvMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  citations?: string[];
}

export interface ConvEntry {
  id: string;
  repoKey: string;
  title: string;
  messages: ConvMessage[];
  lastMessageAt: number;
  model?: string;
}

export interface ConvSummary extends ConvEntry {
  messageCount: number;
}

export interface RepoEntry {
  type: string;
  owner: string;
  repo: string;
  convs: ConvSummary[];
}

function repoKeyOf(entry: Pick<RepoEntry, "type" | "owner" | "repo">) {
  return `${entry.type}:${entry.owner}/${entry.repo}`;
}

function parseRepoKey(repoKey: string): RepoEntry {
  const [type = "github", path = ""] = repoKey.split(":");
  const [owner = "unknown", repo = "repository"] = path.split("/");
  return { type, owner, repo, convs: [] };
}

function legacyEntries(value: unknown): ConvEntry[] {
  if (!Array.isArray(value) || !value.some((item) => item && typeof item === "object" && "convs" in item)) {
    return [];
  }
  return value.flatMap((repo) => {
    const typedRepo = repo as RepoEntry;
    const repoKey = repoKeyOf(typedRepo);
    return (typedRepo.convs ?? []).map((conv) => ({
      id: conv.id,
      repoKey,
      title: conv.title,
      messages: conv.messages ?? [],
      lastMessageAt: conv.lastMessageAt,
      model: conv.model,
    }));
  });
}

function readHistory(): ConvEntry[] {
  if (typeof window === "undefined") {
    return [];
  }
  Object.keys(window.localStorage)
    .filter((key) => key.startsWith("deepwiki."))
    .forEach((key) => window.localStorage.removeItem(key));

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return [];
  }
  const parsed = JSON.parse(raw) as unknown;
  const migrated = legacyEntries(parsed);
  if (migrated.length > 0) {
    return migrated;
  }
  return Array.isArray(parsed) ? (parsed as ConvEntry[]) : [];
}

function trimForQuota(conversations: ConvEntry[]) {
  let next = [...conversations].sort((left, right) => right.lastMessageAt - left.lastMessageAt);
  while (JSON.stringify(next).length > MAX_STORAGE_BYTES && next.length > 1) {
    next = next.slice(0, -1);
  }
  return next;
}

function writeHistory(conversations: ConvEntry[]) {
  const next = trimForQuota(conversations);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  window.dispatchEvent(new Event(HISTORY_EVENT));
  return next;
}

function summarize(conversations: ConvEntry[]): RepoEntry[] {
  const repos = new Map<string, RepoEntry>();
  conversations.forEach((conv) => {
    const existing = repos.get(conv.repoKey) ?? parseRepoKey(conv.repoKey);
    repos.set(conv.repoKey, {
      ...existing,
      convs: [
        ...existing.convs,
        {
          ...conv,
          messageCount: conv.messages.length,
        },
      ].sort((left, right) => right.lastMessageAt - left.lastMessageAt),
    });
  });
  return Array.from(repos.values()).sort((left, right) => {
    const leftTime = left.convs[0]?.lastMessageAt ?? 0;
    const rightTime = right.convs[0]?.lastMessageAt ?? 0;
    return rightTime - leftTime;
  });
}

export function useConversationHistory() {
  const [conversations, setConversations] = useState<ConvEntry[]>([]);
  const [hydrated, setHydrated] = useState(false);

  const refresh = useCallback(() => {
    setConversations(readHistory());
    setHydrated(true);
  }, []);

  useEffect(() => {
    refresh();
    const onStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY) {
        refresh();
      }
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(HISTORY_EVENT, refresh);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(HISTORY_EVENT, refresh);
    };
  }, [refresh]);

  const persist = useCallback((updater: (current: ConvEntry[]) => ConvEntry[]) => {
    setConversations((current) => writeHistory(updater(current)));
  }, []);

  const addConversation = useCallback(
    (repoKey: string, conv: Omit<ConvEntry, "repoKey">) => {
      persist((current) => [{ ...conv, repoKey }, ...current.filter((item) => item.id !== conv.id)]);
    },
    [persist],
  );

  const removeConversation = useCallback(
    (convId: string) => {
      persist((current) => current.filter((conv) => conv.id !== convId));
    },
    [persist],
  );

  const removeRepo = useCallback(
    (repoKey: string) => {
      persist((current) => current.filter((conv) => conv.repoKey !== repoKey));
    },
    [persist],
  );

  const upsertRepo = useCallback(
    (entry: RepoEntry) => {
      const repoKey = repoKeyOf(entry);
      persist((current) => [
        ...entry.convs.map((conv) => ({
          id: conv.id,
          repoKey,
          title: conv.title,
          messages: conv.messages,
          lastMessageAt: conv.lastMessageAt,
          model: conv.model,
        })),
        ...current.filter((conv) => conv.repoKey !== repoKey),
      ]);
    },
    [persist],
  );

  const getConversation = useCallback(
    (convId: string) => conversations.find((conv) => conv.id === convId),
    [conversations],
  );

  const repos = useMemo(() => summarize(conversations), [conversations]);

  return { conversations, hydrated, repos, addConversation, getConversation, removeConversation, upsertRepo, removeRepo };
}
