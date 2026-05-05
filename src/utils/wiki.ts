import type RepoInfo from "../types/repoinfo";
import type { WikiPage } from "../types/wiki/wikipage";
import type { WikiSection, WikiStructure } from "../types/wiki/wikistructure";
import getRepoUrl from "./getRepoUrl";

export interface WikiCache {
  wiki_structure: WikiStructure;
  generated_pages: Record<string, string | WikiPage | { content?: string }>;
  repo?: RepoInfo;
  provider?: string;
  model?: string;
}

export interface WikiPageWithContent extends WikiPage {
  content: string;
}

export interface WikiSectionView {
  id: string;
  title: string;
  pages: WikiPageWithContent[];
}

const sectionFallback = "Contents";

export function buildCacheQuery(owner: string, repo: string, repoType: string, language: string) {
  const params = new URLSearchParams({
    owner,
    repo,
    repo_type: repoType,
    language,
  });
  return params.toString();
}

export async function fetchWikiCache(owner: string, repo: string, repoType: string, language: string) {
  const response = await fetch(`/api/wiki_cache?${buildCacheQuery(owner, repo, repoType, language)}`);
  if (!response.ok) {
    throw new Error(`Failed to load wiki cache (${response.status})`);
  }
  return (await response.json()) as WikiCache | null;
}

export async function saveWikiCache(cache: WikiCache, repoInfo: RepoInfo, language: string) {
  const response = await fetch("/api/wiki_cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo: repoInfo,
      language,
      wiki_structure: cache.wiki_structure,
      generated_pages: cache.generated_pages,
      provider: cache.provider,
      model: cache.model,
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save wiki cache (${response.status})`);
  }
}

export function wikiPageContent(page: WikiPage, generatedPages: WikiCache["generated_pages"]) {
  const generated = generatedPages[page.id];
  const content = typeof generated === "string" ? generated : generated?.content;
  return (content || page.content || "").trim();
}

export function hydrateWikiPages(cache: WikiCache): WikiPageWithContent[] {
  return cache.wiki_structure.pages.map((page) => ({
    ...page,
    content: wikiPageContent(page, cache.generated_pages),
  }));
}

export function defaultWikiPageId(cache: WikiCache) {
  const firstSectionPage = cache.wiki_structure.sections?.flatMap((section) => section.pages)[0];
  return firstSectionPage ?? cache.wiki_structure.pages[0]?.id ?? "";
}

export function findWikiPage(cache: WikiCache, pageId: string | null): WikiPageWithContent | null {
  const pages = hydrateWikiPages(cache);
  const id = pageId || defaultWikiPageId(cache);
  return pages.find((page) => page.id === id) ?? pages[0] ?? null;
}

export function wikiSectionViews(cache: WikiCache): WikiSectionView[] {
  const pages = hydrateWikiPages(cache);
  const pageById = new Map(pages.map((page) => [page.id, page]));
  const seen = new Set<string>();
  const sections = (cache.wiki_structure.sections ?? [])
    .map((section) => {
      const sectionPages = section.pages
        .map((id) => pageById.get(id))
        .filter((page): page is WikiPageWithContent => Boolean(page));
      sectionPages.forEach((page) => seen.add(page.id));
      return { id: section.id, title: section.title, pages: sectionPages };
    })
    .filter((section) => section.pages.length > 0);

  const unsectioned = pages.filter((page) => !seen.has(page.id));
  if (unsectioned.length > 0 || sections.length === 0) {
    sections.push({ id: "contents", title: sectionFallback, pages: unsectioned.length > 0 ? unsectioned : pages });
  }
  return sections;
}

export function pageSectionTitle(cache: WikiCache, pageId: string) {
  return wikiSectionViews(cache).find((section) => section.pages.some((page) => page.id === pageId))?.title ?? sectionFallback;
}

export function repoLabel(owner: string, repo: string, repoType: string) {
  return `${repoType}:${owner}/${repo}`;
}

export function localStorageRepoKey(repoInfo: RepoInfo, language: string) {
  return `${repoInfo.type}.${repoInfo.owner}.${repoInfo.repo}.${language}`;
}

export function stripLeadingTitle(content: string, title: string) {
  const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return content.replace(new RegExp(`^#\\s+${escaped}\\s*\\n+`, "i"), "").trim();
}

export async function exportWiki(cache: WikiCache, repoInfo: RepoInfo, format: "markdown" | "json") {
  const pages = cache.wiki_structure.pages.map((page) => ({
    ...page,
    content: wikiPageContent(page, cache.generated_pages),
  }));
  const response = await fetch("/export/wiki", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_url: getRepoUrl(repoInfo),
      pages,
      format,
    }),
  });
  if (!response.ok) {
    throw new Error(`Wiki export failed (${response.status})`);
  }
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="?([^";]+)"?/i);
  const filename = match?.[1] ?? `${repoInfo.repo}_wiki.${format === "markdown" ? "md" : "json"}`;
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 72) || "page";
}

export function normalizeGeneratedStructure(structure: WikiStructure): WikiStructure {
  const pages = structure.pages.map((page, index) => ({
    ...page,
    id: page.id || slugify(page.title || `page-${index + 1}`),
    title: page.title || `Page ${index + 1}`,
    content: page.content || "",
    filePaths: page.filePaths ?? [],
    importance: page.importance ?? "medium",
    relatedPages: page.relatedPages ?? [],
  }));
  const validIds = new Set(pages.map((page) => page.id));
  const sections: WikiSection[] =
    structure.sections
      ?.map((section, index) => ({
        id: section.id || slugify(section.title || `section-${index + 1}`),
        title: section.title || `Section ${index + 1}`,
        pages: section.pages.filter((id) => validIds.has(id)),
        subsections: section.subsections,
      }))
      .filter((section) => section.pages.length > 0) ?? [];

  return {
    id: structure.id || "wiki",
    title: structure.title || "Project Wiki",
    description: structure.description || "Generated repository wiki",
    pages,
    sections: sections.length > 0 ? sections : [{ id: "contents", title: sectionFallback, pages: pages.map((page) => page.id) }],
    rootSections: structure.rootSections,
  };
}
