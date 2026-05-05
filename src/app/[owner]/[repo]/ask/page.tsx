"use client";

import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatComposer } from "../../../../components/chat/ChatComposer";
import { ChatStream } from "../../../../components/chat/ChatStream";
import { ChatTopbar } from "../../../../components/chat/ChatTopbar";
import {
  buildRepoInfo,
  extractCitations,
  newId,
  streamAgentChatWs,
  titleFrom,
  toolUpdate,
  type AgentName,
} from "../../../../components/chat/runtime";
import type { ChatMessage } from "../../../../components/chat/types";
import { AppShell } from "../../../../components/shell/AppShell";
import { useLanguage } from "../../../../contexts/LanguageContext";
import { useSettings } from "../../../../contexts/SettingsContext";
import { useConversationHistory } from "../../../../hooks/useConversationHistory";
import type { AgentChatEvent, AgentChatRequest } from "../../../../types/agentChat";
import { streamAgentChatHttp } from "../../../../utils/agentChatStream";
import getRepoUrl from "../../../../utils/getRepoUrl";

export default function AskPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();
  const owner = decodeURIComponent(String(params.owner));
  const repo = decodeURIComponent(String(params.repo));
  const settings = useSettings();
  const { language } = useLanguage();
  const { addConversation, getConversation, hydrated } = useConversationHistory();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [agentName, setAgentName] = useState<AgentName>("explore");
  const [conversationId, setConversationId] = useState(searchParams.get("convId") ?? newId());
  const [loadedConvId, setLoadedConvId] = useState("");
  const activeWs = useRef<WebSocket | null>(null);
  const autoSubmitted = useRef(false);

  const repoInfo = useMemo(() => buildRepoInfo(owner, repo, settings.token, searchParams), [owner, repo, searchParams, settings.token]);
  const repoKey = `${repoInfo.type}:${owner}/${repo}`;
  const currentTitle = titleFrom(messages);

  useEffect(() => () => activeWs.current?.close(), []);

  useEffect(() => {
    const convId = searchParams.get("convId");
    if (!hydrated || !convId || loadedConvId === convId) return;
    const conv = getConversation(convId);
    if (conv) {
      setConversationId(conv.id);
      setMessages(conv.messages.map((message) => ({ ...message, id: newId() })));
    }
    setLoadedConvId(convId);
  }, [getConversation, hydrated, loadedConvId, searchParams]);

  const updateUrl = useCallback(() => {
    const next = new URLSearchParams(searchParams.toString());
    next.set("convId", conversationId);
    next.delete("q");
    router.replace(`${pathname}?${next.toString()}`, { scroll: false });
  }, [conversationId, pathname, router, searchParams]);

  const persistConversation = useCallback(
    (nextMessages: ChatMessage[]) => {
      addConversation(repoKey, {
        id: conversationId,
        title: titleFrom(nextMessages),
        messages: nextMessages.map(({ role, content, timestamp, citations }) => ({ role, content, timestamp, citations })),
        lastMessageAt: Date.now(),
        model: settings.model || undefined,
      });
    },
    [addConversation, conversationId, repoKey, settings.model],
  );

  const submit = useCallback(
    async (forcedQuestion?: string) => {
      const question = (forcedQuestion ?? draft).trim();
      if (!question || streaming) return;
      const userMessage: ChatMessage = { id: newId(), role: "user", content: question, timestamp: Date.now() };
      const assistantId = newId();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        model: settings.model || undefined,
        streaming: true,
      };
      const requestMessages = [...messages, userMessage].map(({ role, content }) => ({ role, content }));
      const request: AgentChatRequest = {
        repo_url: getRepoUrl(repoInfo),
        type: repoInfo.type,
        token: repoInfo.token || undefined,
        provider: settings.provider || undefined,
        model: settings.model || undefined,
        language,
        messages: requestMessages,
        agent_name: deepResearch ? "deep-research" : agentName,
        excluded_dirs: settings.excludedDirs || undefined,
        excluded_files: settings.excludedFiles || undefined,
        included_dirs: settings.includedDirs || undefined,
        included_files: settings.includedFiles || undefined,
      };
      let assistantText = "";
      let errorText = "";
      const applyEvent = (event: AgentChatEvent) => {
        if (event.type === "text_delta") assistantText += event.content;
        if (event.type === "error") errorText = event.error;
        setMessages((current) =>
          current.map((message) => {
            if (message.id !== assistantId) return message;
            if (event.type === "text_delta") return { ...message, content: message.content + event.content };
            if (event.type === "tool_call_start" || event.type === "tool_call_end") return { ...message, toolEvents: toolUpdate(message.toolEvents, event) };
            if (event.type === "error") return { ...message, error: event.error, streaming: false };
            if (event.type === "finish") return { ...message, streaming: false };
            return message;
          }),
        );
      };

      setDraft("");
      setStreaming(true);
      setMessages((current) => [...current, userMessage, assistantMessage]);
      updateUrl();
      try {
        try {
          await streamAgentChatWs(request, applyEvent, (ws) => {
            activeWs.current = ws;
          });
        } catch {
          assistantText = "";
          setMessages((current) => current.map((message) => (message.id === assistantId ? { ...message, content: "", toolEvents: [] } : message)));
          await streamAgentChatHttp(request, applyEvent);
        }
      } catch (error) {
        errorText = error instanceof Error ? error.message : "Agent chat failed";
      } finally {
        const { body, citations } = extractCitations(assistantText);
        setMessages((current) => {
          const finalized = current.map((message) =>
            message.id === assistantId ? { ...message, content: body, citations, error: errorText || message.error, streaming: false } : message,
          );
          persistConversation(finalized);
          return finalized;
        });
        setStreaming(false);
        activeWs.current = null;
      }
    },
    [agentName, deepResearch, draft, language, messages, persistConversation, repoInfo, settings, streaming, updateUrl],
  );

  useEffect(() => {
    const q = searchParams.get("q");
    const convId = searchParams.get("convId");
    if (!hydrated || !q || autoSubmitted.current || streaming) return;
    if (convId && loadedConvId !== convId) return;
    autoSubmitted.current = true;
    void submit(q);
  }, [hydrated, loadedConvId, searchParams, streaming, submit]);

  return (
    <AppShell
      topbar={
        <ChatTopbar
          deepResearch={deepResearch}
          owner={owner}
          repo={repo}
          streaming={streaming}
          title={currentTitle}
          onDeepResearchChange={setDeepResearch}
        />
      }
      composer={
        <ChatComposer
          agentName={agentName}
          deepResearch={deepResearch}
          streaming={streaming}
          value={draft}
          onAgentNameChange={setAgentName}
          onChange={setDraft}
          onSubmit={() => void submit()}
        />
      }
    >
      <ChatStream messages={messages} />
    </AppShell>
  );
}
