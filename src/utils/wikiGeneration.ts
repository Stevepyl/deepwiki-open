import type { SettingsContextValue } from "../contexts/SettingsContext";
import type RepoInfo from "../types/repoinfo";
import type { WikiPage } from "../types/wiki/wikipage";
import type { WikiStructure } from "../types/wiki/wikistructure";
import { createChatWebSocket, type ChatCompletionRequest } from "./websocketClient";
import getRepoUrl from "./getRepoUrl";
import { normalizeGeneratedStructure, slugify, type WikiCache } from "./wiki";

type GenerationSettings = Pick<
  SettingsContextValue,
  "provider" | "model" | "token" | "excludedDirs" | "excludedFiles" | "includedDirs" | "includedFiles"
>;

interface StreamOptions {
  onText?: (text: string) => void;
  onSocket?: (socket: WebSocket) => void;
}

const WIKI_GENERATION_TIMEOUT_MS = 45 * 60 * 1000;

function requestFor(repoInfo: RepoInfo, language: string, settings: GenerationSettings, prompt: string): ChatCompletionRequest {
  return {
    repo_url: getRepoUrl(repoInfo),
    type: repoInfo.type,
    token: repoInfo.token || settings.token || undefined,
    provider: settings.provider || undefined,
    model: settings.model || undefined,
    language,
    excluded_dirs: settings.excludedDirs || undefined,
    excluded_files: settings.excludedFiles || undefined,
    included_dirs: settings.includedDirs || undefined,
    included_files: settings.includedFiles || undefined,
    messages: [{ role: "user", content: prompt }],
  };
}

function streamWebSocket(request: ChatCompletionRequest, options: StreamOptions) {
  return new Promise<string>((resolve, reject) => {
    let output = "";
    let settled = false;

    function settle(fn: () => void) {
      if (!settled) {
        settled = true;
        clearTimeout(timeout);
        fn();
      }
    }

    const socket = createChatWebSocket(
      request,
      (message) => {
        output += message;
        options.onText?.(message);
      },
      () => settle(() => reject(new Error("Wiki generation WebSocket failed"))),
      () => settle(() => resolve(output)),
    );
    const timeout = setTimeout(() => {
      socket.close();
      settle(() => reject(new Error("Wiki generation WebSocket timed out")));
    }, WIKI_GENERATION_TIMEOUT_MS);
    options.onSocket?.(socket);
  });
}

async function streamRawText(request: ChatCompletionRequest, options: StreamOptions = {}) {
  return streamWebSocket(request, options);
}

function extractJson(text: string): unknown {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
  const source = fenced ?? text;
  const start = source.indexOf("{");
  const end = source.lastIndexOf("}");
  if (start < 0 || end < start) {
    throw new Error("Wiki structure response did not contain JSON");
  }
  return JSON.parse(source.slice(start, end + 1));
}

function toStructure(value: unknown): WikiStructure {
  const data = value as { wiki_structure?: WikiStructure } & Partial<WikiStructure>;
  return normalizeGeneratedStructure(data.wiki_structure ?? (data as WikiStructure));
}

function buildStructurePrompt(repoInfo: RepoInfo, language: string) {
  return `Generate a comprehensive repository wiki structure for ${getRepoUrl(repoInfo)}.

Return JSON only. Do not wrap it in XML.

Schema:
{
  "id": "wiki",
  "title": "Project Wiki",
  "description": "One sentence summary",
  "sections": [{"id": "architecture", "title": "Architecture", "pages": ["overview"]}],
  "pages": [
    {
      "id": "overview",
      "title": "Architecture Overview",
      "content": "",
      "filePaths": ["README.md"],
      "importance": "high",
      "relatedPages": []
    }
  ]
}

Create 8-12 useful pages grouped into 3-5 sections. Use stable kebab-case ids. Prefer concrete source file paths when you know them. Generate all titles and descriptions in language code "${language}".`;
}

function buildPagePrompt(repoInfo: RepoInfo, structure: WikiStructure, page: WikiPage, language: string) {
  const siblingTitles = structure.pages.map((item) => item.title).join(", ");
  return `Write the wiki page "${page.title}" for ${getRepoUrl(repoInfo)}.

Known file paths for this page: ${page.filePaths.length > 0 ? page.filePaths.join(", ") : "infer from repository context"}.
Related wiki pages: ${siblingTitles}.

Return Markdown only. Start with a concise technical overview, then use H2/H3 sections, code references, and Mermaid diagrams when they clarify architecture. Do not include frontmatter. Write in language code "${language}".`;
}

export async function generateWikiCache(
  repoInfo: RepoInfo,
  language: string,
  settings: GenerationSettings,
  options: StreamOptions & { onPageStart?: (index: number, total: number, page: WikiPage) => void } = {},
): Promise<WikiCache> {
  const structureText = await streamRawText(requestFor(repoInfo, language, settings, buildStructurePrompt(repoInfo, language)), options);
  const wikiStructure = toStructure(extractJson(structureText));
  const generatedPages: Record<string, string> = {};
  for (const [index, page] of wikiStructure.pages.entries()) {
    options.onPageStart?.(index + 1, wikiStructure.pages.length, page);
    const content = await streamRawText(requestFor(repoInfo, language, settings, buildPagePrompt(repoInfo, wikiStructure, page, language)), {
      onSocket: options.onSocket,
    });
    generatedPages[page.id || slugify(page.title)] = content.trim();
  }
  return {
    wiki_structure: wikiStructure,
    generated_pages: generatedPages,
    repo: repoInfo,
    provider: settings.provider || undefined,
    model: settings.model || undefined,
  };
}

export async function generateWorkshopMarkdown(repoInfo: RepoInfo, cache: WikiCache, language: string, settings: GenerationSettings) {
  const pageList = cache.wiki_structure.pages.map((page, index) => `${index + 1}. ${page.title}`).join("\n");
  const prompt = `Create a hands-on workshop for ${getRepoUrl(repoInfo)} from this wiki outline:
${pageList}

Return Markdown with 5-8 sections. Each section must start with "## Step NN: Title". Include one practical exercise and one short hint per step. Write in language code "${language}".`;
  return streamRawText(requestFor(repoInfo, language, settings, prompt));
}
