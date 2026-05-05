"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "opswiki.conversationHistory";

export interface ConvEntry {
  id: string;
  title: string;
  lastMessageAt: number;
  messageCount: number;
}

export interface RepoEntry {
  type: string;
  owner: string;
  repo: string;
  convs: ConvEntry[];
}

function repoKeyOf(entry: Pick<RepoEntry, "type" | "owner" | "repo">) {
  return `${entry.type}:${entry.owner}/${entry.repo}`;
}

function parseRepoKey(repoKey: string): RepoEntry {
  const [type = "github", path = ""] = repoKey.split(":");
  const [owner = "unknown", repo = "repository"] = path.split("/");
  return { type, owner, repo, convs: [] };
}

function readHistory(): RepoEntry[] {
  if (typeof window === "undefined") {
    return [];
  }
  Object.keys(window.localStorage)
    .filter((key) => key.startsWith("deepwiki."))
    .forEach((key) => window.localStorage.removeItem(key));

  const raw = window.localStorage.getItem(STORAGE_KEY);
  return raw ? (JSON.parse(raw) as RepoEntry[]) : [];
}

export function useConversationHistory() {
  const [repos, setRepos] = useState<RepoEntry[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setRepos(readHistory());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (hydrated) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(repos));
    }
  }, [hydrated, repos]);

  const upsertRepo = useCallback((entry: RepoEntry) => {
    setRepos((current) => {
      const key = repoKeyOf(entry);
      const exists = current.some((repo) => repoKeyOf(repo) === key);
      return exists
        ? current.map((repo) => (repoKeyOf(repo) === key ? { ...repo, ...entry } : repo))
        : [...current, entry];
    });
  }, []);

  const removeRepo = useCallback((repoKey: string) => {
    setRepos((current) => current.filter((repo) => repoKeyOf(repo) !== repoKey));
  }, []);

  const addConversation = useCallback((repoKey: string, conv: ConvEntry) => {
    setRepos((current) => {
      const repoEntry = current.find((repo) => repoKeyOf(repo) === repoKey) ?? parseRepoKey(repoKey);
      const nextRepo = {
        ...repoEntry,
        convs: [conv, ...repoEntry.convs.filter((item) => item.id !== conv.id)],
      };
      return current.some((repo) => repoKeyOf(repo) === repoKey)
        ? current.map((repo) => (repoKeyOf(repo) === repoKey ? nextRepo : repo))
        : [nextRepo, ...current];
    });
  }, []);

  const removeConversation = useCallback((convId: string) => {
    setRepos((current) =>
      current.map((repo) => ({
        ...repo,
        convs: repo.convs.filter((conv) => conv.id !== convId),
      })),
    );
  }, []);

  return { repos, addConversation, removeConversation, upsertRepo, removeRepo };
}
