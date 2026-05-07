"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useLanguage } from "../contexts/LanguageContext";
import { useConversationHistory } from "./useConversationHistory";
import { type ProcessedProject, useProcessedProjects } from "./useProcessedProjects";

export type ViewMode = "grid" | "list";
export type SortMode = "name" | "recent";

const VIEW_KEY = "opswiki.projectsView";

export function pathOf(project: ProcessedProject) {
  return `${project.owner}/${project.repo}`;
}

export function projectHref(project: ProcessedProject) {
  const params = new URLSearchParams({ type: project.repo_type, language: project.language || "zh" });
  return `/${encodeURIComponent(project.owner)}/${encodeURIComponent(project.repo)}?${params}`;
}

export function descriptionFor(project: ProcessedProject) {
  return `Generated wiki for ${pathOf(project)}.`;
}

export function relativeTime(timestamp: number) {
  const diff = Date.now() - timestamp;
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < hour) return `${Math.max(1, Math.round(diff / minute))} min ago`;
  if (diff < day) return `${Math.round(diff / hour)} hours ago`;
  if (diff < 30 * day) return `${Math.round(diff / day)} days ago`;
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(new Date(timestamp));
}

export function useProjectsDirectory() {
  const { projects, isLoading, error } = useProcessedProjects();
  const { repos } = useConversationHistory();
  const { language } = useLanguage();
  const filterRef = useRef<HTMLInputElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [view, setView] = useState<ViewMode>("grid");
  const [sortMode, setSortMode] = useState<SortMode>("name");
  const [activeLetter, setActiveLetter] = useState("");
  const [removedIds, setRemovedIds] = useState<Set<string>>(new Set());
  const [deletingId, setDeletingId] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const stored = window.localStorage.getItem(VIEW_KEY);
    if (stored === "grid" || stored === "list") setView(stored);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(VIEW_KEY, view);
  }, [view]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        filterRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const groups = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const visible = projects
      .filter((project) => !removedIds.has(project.id))
      .filter((project) => pathOf(project).toLowerCase().includes(normalizedQuery))
      .sort((left, right) =>
        sortMode === "recent" ? right.submittedAt - left.submittedAt : pathOf(left).localeCompare(pathOf(right)),
      );

    return Object.entries(
      visible.reduce<Record<string, ProcessedProject[]>>((acc, project) => {
        const letter = pathOf(project).charAt(0).toUpperCase() || "#";
        return { ...acc, [letter]: [...(acc[letter] ?? []), project] };
      }, {}),
    )
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([letter, items]) => ({ letter, items }));
  }, [projects, query, removedIds, sortMode]);

  const conversationCounts = useMemo<Map<string, number>>(
    () => new Map(repos.map((repo) => [`${repo.type}:${repo.owner}/${repo.repo}`, repo.convs.length] as const)),
    [repos],
  );

  useEffect(() => {
    setActiveLetter(groups[0]?.letter ?? "");
  }, [groups]);

  useEffect(() => {
    const root = contentRef.current;
    if (!root) return;
    const headings = Array.from(root.querySelectorAll<HTMLElement>("[data-letter]"));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.find((entry) => entry.isIntersecting);
        const letter = visible?.target.getAttribute("data-letter");
        if (letter) setActiveLetter(letter);
      },
      { root, rootMargin: "0px 0px -70% 0px" },
    );
    headings.forEach((heading) => observer.observe(heading));
    return () => observer.disconnect();
  }, [groups]);

  const deleteProject = async (project: ProcessedProject) => {
    setDeletingId(project.id);
    setNotice("");
    try {
      const response = await fetch("/api/wiki/projects", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          owner: project.owner,
          repo: project.repo,
          repo_type: project.repo_type,
          language: project.language || language || "zh",
        }),
      });
      if (!response.ok) {
        setNotice(response.status === 401 ? "Deletion failed — auth required" : "Deletion failed.");
        return;
      }
      setRemovedIds((current) => new Set([...current, project.id]));
      setNotice(`Deleted ${pathOf(project)}.`);
    } catch {
      setNotice("Deletion failed.");
    } finally {
      setDeletingId("");
    }
  };

  return {
    activeLetter,
    contentRef,
    conversationCounts,
    deleteProject,
    deletingId,
    error,
    filterRef,
    groups,
    isLoading,
    notice,
    projects,
    query,
    setQuery,
    setSortMode,
    setView,
    sortMode,
    view,
  };
}
