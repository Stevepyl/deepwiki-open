import type { AgentChatEvent, AgentChatRequest } from "../../types/agentChat";
import type RepoInfo from "../../types/repoinfo";
import { createAgentChatWebSocket } from "../../utils/websocketClient";
import type { ChatMessage, ChatToolEvent } from "./types";

export type AgentName = NonNullable<AgentChatRequest["agent_name"]>;

export function newId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

export function inferRepoType(repoUrl: string | null, explicitType: string | null) {
  if (explicitType) return explicitType;
  try {
    const host = repoUrl ? new URL(repoUrl).hostname.toLowerCase() : "";
    if (host.includes("gitlab")) return "gitlab";
    if (host.includes("bitbucket")) return "bitbucket";
  } catch {
    return "github";
  }
  return "github";
}

export function buildRepoInfo(owner: string, repo: string, token: string, query: { get(name: string): string | null }): RepoInfo {
  const repoUrl = query.get("repo_url");
  return {
    owner,
    repo,
    type: inferRepoType(repoUrl, query.get("type")),
    token: token || query.get("token"),
    localPath: query.get("local_path"),
    repoUrl,
  };
}

export function extractCitations(content: string) {
  const match = content.match(/\n{2,}Sources:\s*\n((?:-\s+.+\n?)+)\s*$/i);
  if (!match) return { body: content, citations: [] as string[] };
  return {
    body: content.slice(0, match.index).trimEnd(),
    citations: match[1]
      .split("\n")
      .map((line) => line.replace(/^-\s+/, "").trim())
      .filter(Boolean),
  };
}

export function titleFrom(messages: ChatMessage[]) {
  const firstUser = messages.find((message) => message.role === "user")?.content.trim() ?? "New chat";
  return firstUser.length > 54 ? `${firstUser.slice(0, 51)}...` : firstUser;
}

export function toolUpdate(events: ChatToolEvent[] = [], event: AgentChatEvent): ChatToolEvent[] {
  if (event.type === "tool_call_start") {
    const next = {
      id: event.tool_call_id,
      toolName: event.tool_name,
      status: "running" as const,
      startedAt: Date.now(),
      args: event.tool_args,
    };
    return events.some((item) => item.id === next.id) ? events.map((item) => (item.id === next.id ? next : item)) : [...events, next];
  }
  if (event.type !== "tool_call_end") return events;
  return events.map((item) =>
    item.id === event.tool_call_id
      ? {
          ...item,
          status: event.is_error ? "error" : "complete",
          durationMs: event.duration_ms,
          resultSummary: event.result_summary,
        }
      : item,
  );
}

export function streamAgentChatWs(
  request: AgentChatRequest,
  onEvent: (event: AgentChatEvent) => void,
  onSocket: (ws: WebSocket) => void,
) {
  return new Promise<void>((resolve, reject) => {
    let finished = false;
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) {
        settled = true;
        clearTimeout(timeout);
        fn();
      }
    };
    const ws = createAgentChatWebSocket(
      request,
      (event) => {
        onEvent(event);
        if (event.type === "finish" || event.type === "error") {
          finished = true;
          settle(resolve);
          ws.close();
        }
      },
      () => settle(() => reject(new Error("Agent chat WebSocket failed"))),
      () => {
        if (!finished) settle(() => reject(new Error("Agent chat WebSocket closed before finish")));
      },
    );
    const timeout = setTimeout(() => {
      if (!finished) {
        ws.close();
        settle(() => reject(new Error("Agent chat WebSocket timed out")));
      }
    }, 90000);
    onSocket(ws);
  });
}
