import type RepoInfo from "../types/repoinfo";
import { localStorageRepoKey, stripLeadingTitle, wikiSectionViews, type WikiCache } from "./wiki";
import type { Slide } from "../components/slides/types";

export function slidesStorageKey(repoInfo: RepoInfo, language: string) {
  return `opswiki.slides.${localStorageRepoKey(repoInfo, language)}`;
}

function slideBody(content: string, title: string) {
  return stripLeadingTitle(content, title)
    .split(/\n{2,}/)
    .filter(Boolean)
    .slice(0, 4)
    .join("\n\n");
}

export function buildSlides(cache: WikiCache): Slide[] {
  return wikiSectionViews(cache).flatMap((section, sectionIndex) => [
    {
      id: `section-${section.id}`,
      variant: "divider" as const,
      eyebrow: `Section ${String(sectionIndex + 1).padStart(2, "0")}`,
      title: section.title,
      subtitle: `${section.pages.length} wiki pages`,
    },
    ...section.pages.map((page, pageIndex) => ({
      id: page.id,
      variant: page.content.includes("```mermaid") ? ("diagram" as const) : ("content" as const),
      eyebrow: `${section.title} · §${sectionIndex + 1}.${pageIndex + 1}`,
      title: page.title,
      content: slideBody(page.content, page.title),
    })),
  ]);
}
