import type { SettingsContextValue } from "../contexts/SettingsContext";
import type RepoInfo from "../types/repoinfo";
import type { WikiPage } from "../types/wiki/wikipage";
import {
  createAgentWikiWebSocket,
  createChatWebSocket,
  type AgentWikiRequest,
  type ChatCompletionRequest,
} from "./websocketClient";
import getRepoUrl from "./getRepoUrl";
import { normalizeGeneratedStructure, type WikiCache } from "./wiki";

type GenerationSettings = Pick<
  SettingsContextValue,
  "provider" | "model" | "token" | "excludedDirs" | "excludedFiles" | "includedDirs" | "includedFiles"
>;

interface StreamOptions {
  onText?: (text: string) => void;
  onSocket?: (socket: WebSocket) => void;
}

interface GenerateWikiOptions extends StreamOptions {
  onPageStart?: (index: number, total: number, page: WikiPage) => void;
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

function agentWikiRequestFor(repoInfo: RepoInfo, language: string, settings: GenerationSettings): AgentWikiRequest {
  return {
    repo_url: getRepoUrl(repoInfo),
    type: repoInfo.type,
    token: repoInfo.token || settings.token || undefined,
    provider: settings.provider || undefined,
    model: settings.model || undefined,
    language,
    comprehensive: true,
    excluded_dirs: settings.excludedDirs || undefined,
    excluded_files: settings.excludedFiles || undefined,
    included_dirs: settings.includedDirs || undefined,
    included_files: settings.includedFiles || undefined,
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

function streamAgentWikiCache(
  repoInfo: RepoInfo,
  language: string,
  settings: GenerationSettings,
  options: GenerateWikiOptions,
): Promise<WikiCache> {
  return new Promise<WikiCache>((resolve, reject) => {
    let wikiStructure: WikiCache["wiki_structure"] | null = null;
    const generatedPages: WikiCache["generated_pages"] = {};
    let settled = false;
    let socket: WebSocket | null = null;

    function settle(fn: () => void) {
      if (!settled) {
        settled = true;
        clearTimeout(timeout);
        fn();
      }
    }

    function fail(message: string) {
      socket?.close();
      settle(() => reject(new Error(message)));
    }

    const timeout = setTimeout(() => {
      socket?.close();
      settle(() => reject(new Error("Agent wiki generation WebSocket timed out")));
    }, WIKI_GENERATION_TIMEOUT_MS);

    socket = createAgentWikiWebSocket(
      agentWikiRequestFor(repoInfo, language, settings),
      (event) => {
        switch (event.type) {
          case "text_delta":
            options.onText?.(event.content);
            break;
          case "wiki_structure_ready":
            wikiStructure = normalizeGeneratedStructure(event.structure);
            break;
          case "wiki_page_done": {
            const page = wikiStructure?.pages.find((item) => item.id === event.page_id) ?? {
              id: event.page_id,
              title: event.page_title,
              content: "",
              filePaths: [],
              importance: "medium" as const,
              relatedPages: [],
            };
            options.onPageStart?.(event.page_index + 1, event.total_pages, page);
            generatedPages[event.page_id] = event.content.trim();
            break;
          }
          case "wiki_structure_error":
            fail(event.message);
            break;
          case "wiki_page_error":
            fail(`Wiki page generation failed for ${event.page_id}: ${event.message}`);
            break;
          case "error":
            fail(event.error);
            break;
          case "finish":
            settle(() => {
              if (!wikiStructure) {
                reject(new Error("Agent wiki generation finished without wiki structure"));
                return;
              }
              resolve({
                wiki_structure: wikiStructure,
                generated_pages: generatedPages,
                repo: repoInfo,
                provider: settings.provider || undefined,
                model: settings.model || undefined,
              });
            });
            break;
        }
      },
      () => fail("Agent wiki generation WebSocket failed"),
      () => settle(() => reject(new Error("Agent wiki generation WebSocket closed before completion"))),
    );
    options.onSocket?.(socket);
  });
}

export async function generateWikiCache(
  repoInfo: RepoInfo,
  language: string,
  settings: GenerationSettings,
  options: GenerateWikiOptions = {},
): Promise<WikiCache> {
  return streamAgentWikiCache(repoInfo, language, settings, options);
}

export async function generateWorkshopMarkdown(repoInfo: RepoInfo, cache: WikiCache, language: string, settings: GenerationSettings) {
  const pageList = cache.wiki_structure.pages.map((page, index) => `${index + 1}. ${page.title}`).join("\n");
  const prompt = `Create a hands-on workshop for ${getRepoUrl(repoInfo)} from this wiki outline:
${pageList}

Return Markdown with 5-8 sections. Each section must start with "## Step NN: Title". Include one practical exercise and one short hint per step. Write in language code "${language}".`;
  return streamRawText(requestFor(repoInfo, language, settings, prompt));
}
