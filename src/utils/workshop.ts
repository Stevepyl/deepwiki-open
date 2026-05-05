import type { SettingsContextValue } from "../contexts/SettingsContext";
import type RepoInfo from "../types/repoinfo";
import { generateWorkshopMarkdown } from "./wikiGeneration";
import { localStorageRepoKey, stripLeadingTitle, wikiSectionViews, type WikiCache } from "./wiki";
import type { WorkshopStep } from "../components/workshop/types";

type GenerationSettings = Pick<
  SettingsContextValue,
  "provider" | "model" | "token" | "excludedDirs" | "excludedFiles" | "includedDirs" | "includedFiles"
>;

export function workshopStorageKey(repoInfo: RepoInfo, language: string) {
  return `opswiki.workshop.${localStorageRepoKey(repoInfo, language)}`;
}

export function fallbackWorkshopSteps(cache: WikiCache): WorkshopStep[] {
  return wikiSectionViews(cache)
    .flatMap((section) => section.pages.map((page) => ({ page, section: section.title })))
    .slice(0, 8)
    .map(({ page, section }) => {
      const body = stripLeadingTitle(page.content, page.title).split("\n\n").slice(0, 3).join("\n\n");
      const source = page.filePaths[0] ?? "the repository files";
      return {
        id: page.id,
        title: page.title,
        body: body || `Review the ${section} material and connect it to the source files that implement this behavior.`,
        exercise: `Open ${source} and trace the implementation path described in this step.`,
        hint: "Start from the entry point named in the wiki metadata, then follow the imports or call sites outward.",
      };
    });
}

export function parseWorkshopMarkdown(markdown: string, fallback: WorkshopStep[]) {
  const matches = Array.from(markdown.matchAll(/^##\s+Step\s+(\d+)[:\-.]\s+(.+)$/gim));
  if (matches.length < 2) return fallback;
  return matches.map((match, index) => {
    const start = (match.index ?? 0) + match[0].length;
    const end = matches[index + 1]?.index ?? markdown.length;
    const content = markdown.slice(start, end).trim();
    const exercise = content.match(/(?:exercise|try it)[:\-\s]+(.+)/i)?.[1]?.trim();
    const hint = content.match(/hint[:\-\s]+(.+)/i)?.[1]?.trim();
    return {
      id: `step-${match[1].padStart(2, "0")}`,
      title: match[2].trim(),
      body: content.replace(/(?:exercise|try it|hint)[:\-\s]+.+/gi, "").trim(),
      exercise: exercise || fallback[index]?.exercise || "Apply this step to one concrete source file.",
      hint: hint || fallback[index]?.hint || "Use the wiki page sources as the shortest path into the code.",
    };
  });
}

export async function generateWorkshopSteps(repoInfo: RepoInfo, cache: WikiCache, language: string, settings: GenerationSettings) {
  const fallback = fallbackWorkshopSteps(cache);
  const markdown = await generateWorkshopMarkdown(repoInfo, cache, language, settings);
  return parseWorkshopMarkdown(markdown, fallback);
}
