'use client';

import AskComposer, { AskSubmitOptions } from '@/components/AskComposer';
import AskResultView from '@/components/AskResultView';
import ThemeToggle from '@/components/theme-toggle';
import { useLanguage } from '@/contexts/LanguageContext';
import { AskHistoryItem } from '@/types/ask';
import { RepoInfo } from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import { createChatWebSocket, closeWebSocket, ChatCompletionRequest } from '@/utils/websocketClient';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FaArrowLeft } from 'react-icons/fa';

interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const checkIfResearchComplete = (content: string): boolean => {
  if (content.includes('## Final Conclusion')) {
    return true;
  }

  if ((content.includes('## Conclusion') || content.includes('## Summary')) &&
    !content.includes('I will now proceed to') &&
    !content.includes('Next Steps') &&
    !content.includes('next iteration')) {
    return true;
  }

  if (content.includes('This concludes our research') ||
    content.includes('This completes our investigation') ||
    content.includes('This concludes the deep research process') ||
    content.includes('Key Findings and Implementation Details') ||
    content.includes('In conclusion,') ||
    (content.includes('Final') && content.includes('Conclusion'))) {
    return true;
  }

  return false;
};

export default function AskPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const { messages } = useLanguage();

  const owner = params.owner as string;
  const repo = params.repo as string;

  const token = searchParams.get('token') || '';
  const localPath = searchParams.get('local_path') ? decodeURIComponent(searchParams.get('local_path') || '') : undefined;
  const repoUrl = searchParams.get('repo_url') ? decodeURIComponent(searchParams.get('repo_url') || '') : undefined;
  const providerParam = searchParams.get('provider') || '';
  const modelParam = searchParams.get('model') || '';
  const isCustomModelParam = searchParams.get('is_custom_model') === 'true';
  const customModelParam = searchParams.get('custom_model') || '';
  const language = searchParams.get('language') || 'en';
  const initialQuestion = searchParams.get('question') || '';
  const initialDeepResearch = searchParams.get('deep_research') === 'true';

  const repoHost = (() => {
    if (!repoUrl) return '';
    try {
      return new URL(repoUrl).hostname.toLowerCase();
    } catch {
      console.warn(`Invalid repoUrl provided: ${repoUrl}`);
      return '';
    }
  })();

  const repoType = repoHost?.includes('bitbucket')
    ? 'bitbucket'
    : repoHost?.includes('gitlab')
      ? 'gitlab'
      : repoHost?.includes('github')
        ? 'github'
        : searchParams.get('type') || 'github';

  const repoInfo = useMemo<RepoInfo>(() => ({
    owner,
    repo,
    type: repoType,
    token: token || null,
    localPath: localPath || null,
    repoUrl: repoUrl || null
  }), [owner, repo, repoType, token, localPath, repoUrl]);

  const [questionInput, setQuestionInput] = useState('');
  const [deepResearchInput, setDeepResearchInput] = useState(initialDeepResearch);
  const [history, setHistory] = useState<AskHistoryItem[]>([]);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(providerParam);
  const [selectedModel, setSelectedModel] = useState(modelParam);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModelParam);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModelParam);

  const activeRunsRef = useRef<Map<string, number>>(new Map());
  const webSocketsRef = useRef<Map<string, WebSocket | null>>(new Map());
  const hasSeededInitialQuestionRef = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const composerShellRef = useRef<HTMLDivElement>(null);
  const [composerHeight, setComposerHeight] = useState(120);

  const updateHistoryItem = (itemId: string, updater: (item: AskHistoryItem) => AskHistoryItem) => {
    setHistory((currentHistory) => currentHistory.map((item) => (
      item.id === itemId ? updater(item) : item
    )));
  };

  const closeItemWebSocket = (itemId: string) => {
    const existingWebSocket = webSocketsRef.current.get(itemId);
    closeWebSocket(existingWebSocket || null);
    webSocketsRef.current.delete(itemId);
  };

  const buildRequestBody = (item: AskHistoryItem, messageHistory: Message[]): ChatCompletionRequest => {
    const requestBody: ChatCompletionRequest = {
      repo_url: getRepoUrl(repoInfo),
      type: repoInfo.type,
      messages: messageHistory.map((message) => ({
        role: message.role as 'user' | 'assistant',
        content: message.content
      })),
      provider: item.provider,
      model: item.isCustomModel ? item.customModel : item.model,
      language: item.language
    };

    if (repoInfo.token) {
      requestBody.token = repoInfo.token;
    }

    return requestBody;
  };

  const streamViaHttp = async (
    itemId: string,
    requestBody: ChatCompletionRequest,
    runId: number
  ): Promise<string> => {
    const apiResponse = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!apiResponse.ok) {
      throw new Error(`API error: ${apiResponse.status}`);
    }

    const reader = apiResponse.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    const decoder = new TextDecoder();
    let fullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      if (activeRunsRef.current.get(itemId) !== runId) {
        return fullResponse;
      }

      fullResponse += decoder.decode(value, { stream: true });
      updateHistoryItem(itemId, (item) => ({
        ...item,
        response: fullResponse,
      }));
    }

    return fullResponse;
  };

  const streamResponse = async (
    itemId: string,
    requestBody: ChatCompletionRequest,
    runId: number
  ): Promise<string> => {
    closeItemWebSocket(itemId);

    return new Promise((resolve, reject) => {
      let fullResponse = '';
      let resolved = false;
      let fallbackInProgress = false;

      const resolveOnce = (value: string) => {
        if (!resolved) {
          resolved = true;
          resolve(value);
        }
      };

      const rejectOnce = (error: unknown) => {
        if (!resolved) {
          resolved = true;
          reject(error);
        }
      };

      const webSocket = createChatWebSocket(
        requestBody,
        (message: string) => {
          if (activeRunsRef.current.get(itemId) !== runId) {
            return;
          }

          fullResponse += message;
          updateHistoryItem(itemId, (item) => ({
            ...item,
            response: fullResponse,
          }));
        },
        () => {
          if (fallbackInProgress || resolved || activeRunsRef.current.get(itemId) !== runId) {
            return;
          }

          fallbackInProgress = true;
          streamViaHttp(itemId, requestBody, runId)
            .then((httpResponse) => resolveOnce(httpResponse))
            .catch((httpError) => rejectOnce(httpError));
        },
        () => {
          if (fallbackInProgress || resolved) {
            return;
          }

          resolveOnce(fullResponse);
        }
      );

      webSocketsRef.current.set(itemId, webSocket);
    });
  };

  const runAsk = async (item: AskHistoryItem) => {
    const runId = (activeRunsRef.current.get(item.id) || 0) + 1;
    activeRunsRef.current.set(item.id, runId);

    updateHistoryItem(item.id, (currentItem) => ({
      ...currentItem,
      status: 'loading',
      response: '',
      error: null,
      researchIteration: 0,
      researchComplete: false,
    }));

    try {
      let messageHistory: Message[] = [{
        role: 'user',
        content: item.deepResearch ? `[DEEP RESEARCH] ${item.question}` : item.question
      }];

      let fullResponse = await streamResponse(item.id, buildRequestBody(item, messageHistory), runId);
      if (activeRunsRef.current.get(item.id) !== runId) {
        return;
      }

      if (!item.deepResearch) {
        updateHistoryItem(item.id, (currentItem) => ({
          ...currentItem,
          response: fullResponse,
          status: 'done',
        }));
        return;
      }

      let isComplete = checkIfResearchComplete(fullResponse);
      let iteration = 0;

      while (!isComplete && iteration < 5) {
        iteration += 1;
        updateHistoryItem(item.id, (currentItem) => ({
          ...currentItem,
          response: '',
          researchIteration: iteration,
        }));

        messageHistory = [
          ...messageHistory,
          { role: 'assistant', content: fullResponse },
          { role: 'user', content: '[DEEP RESEARCH] Continue the research' }
        ];

        fullResponse = await streamResponse(item.id, buildRequestBody(item, messageHistory), runId);
        if (activeRunsRef.current.get(item.id) !== runId) {
          return;
        }

        isComplete = checkIfResearchComplete(fullResponse);
      }

      if (!isComplete) {
        fullResponse += '\n\n## Final Conclusion\nAfter multiple iterations of deep research, we have gathered significant insights. This concludes the investigation process after reaching the maximum number of research iterations.';
        isComplete = true;
      }

      updateHistoryItem(item.id, (currentItem) => ({
        ...currentItem,
        response: fullResponse,
        status: 'done',
        researchComplete: isComplete,
      }));
    } catch (error) {
      if (activeRunsRef.current.get(item.id) !== runId) {
        return;
      }

      console.error('Error during answer generation:', error);
      updateHistoryItem(item.id, (currentItem) => ({
        ...currentItem,
        status: 'error',
        error: 'Failed to get a response. Please try again.',
      }));
    } finally {
      if (activeRunsRef.current.get(item.id) === runId) {
        activeRunsRef.current.delete(item.id);
      }
      closeItemWebSocket(item.id);
    }
  };

  const handleAskSubmit = async (askedQuestion: string, options: AskSubmitOptions) => {
    const effectiveModel = isCustomSelectedModel ? customSelectedModel.trim() : selectedModel.trim();
    if (!selectedProvider.trim() || !effectiveModel || isModelLoading) {
      return;
    }

    const nextItem: AskHistoryItem = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      question: askedQuestion,
      deepResearch: options.deepResearch,
      status: 'loading',
      response: '',
      error: null,
      researchIteration: 0,
      researchComplete: false,
      provider: selectedProvider,
      model: selectedModel,
      isCustomModel: isCustomSelectedModel,
      customModel: customSelectedModel,
      language
    };

    setHistory((currentHistory) => [...currentHistory, nextItem]);
    setQuestionInput('');
    window.setTimeout(() => {
      void runAsk(nextItem);
    }, 0);
  };

  useEffect(() => {
    setSelectedProvider(providerParam);
    setSelectedModel(modelParam);
    setIsCustomSelectedModel(isCustomModelParam);
    setCustomSelectedModel(customModelParam);
  }, [providerParam, modelParam, isCustomModelParam, customModelParam]);

  useEffect(() => {
    const currentModel = isCustomSelectedModel ? customSelectedModel.trim() : selectedModel.trim();
    if (selectedProvider.trim() && currentModel) {
      return;
    }

    let cancelled = false;

    const fetchModel = async () => {
      try {
        setIsModelLoading(true);
        const modelResponse = await fetch('/api/models/config');
        if (!modelResponse.ok) {
          throw new Error(`Error fetching model configurations: ${modelResponse.status}`);
        }

        const data = await modelResponse.json();
        if (cancelled) {
          return;
        }

        const fallbackProvider = selectedProvider.trim() || data.defaultProvider;
        setSelectedProvider(fallbackProvider);

        if (!currentModel) {
          const providerConfig = data.providers.find((providerConfig: Provider) => providerConfig.id === fallbackProvider);
          if (providerConfig && providerConfig.models.length > 0) {
            setSelectedModel(providerConfig.models[0].id);
            setIsCustomSelectedModel(false);
            setCustomSelectedModel('');
          }
        }
      } catch (error) {
        console.error('Failed to fetch model configurations:', error);
      } finally {
        if (!cancelled) {
          setIsModelLoading(false);
        }
      }
    };

    void fetchModel();

    return () => {
      cancelled = true;
    };
  }, [selectedProvider, selectedModel, isCustomSelectedModel, customSelectedModel]);

  useEffect(() => {
    const trimmedQuestion = initialQuestion.trim();
    const effectiveModel = isCustomSelectedModel ? customSelectedModel.trim() : selectedModel.trim();

    if (hasSeededInitialQuestionRef.current) {
      return;
    }

    if (!trimmedQuestion) {
      hasSeededInitialQuestionRef.current = true;
      return;
    }

    if (!selectedProvider.trim() || !effectiveModel || isModelLoading) {
      return;
    }

    hasSeededInitialQuestionRef.current = true;
    void handleAskSubmit(trimmedQuestion, { deepResearch: initialDeepResearch });

    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.delete('question');
    nextParams.delete('deep_research');

    const nextQuery = nextParams.toString();
    const nextUrl = nextQuery
      ? `/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/ask?${nextQuery}`
      : `/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/ask`;

    window.history.replaceState({}, '', nextUrl);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    initialQuestion,
    initialDeepResearch,
    searchParams,
    owner,
    repo,
    selectedProvider,
    selectedModel,
    isCustomSelectedModel,
    customSelectedModel,
    isModelLoading
  ]);

  useEffect(() => {
    const activeRuns = activeRunsRef.current;
    const webSockets = webSocketsRef.current;

    return () => {
      activeRuns.clear();
      webSockets.forEach((webSocket) => closeWebSocket(webSocket || null));
      webSockets.clear();
    };
  }, []);

  useEffect(() => {
    const composerShell = composerShellRef.current;
    if (!composerShell) {
      return;
    }

    const updateComposerHeight = () => {
      setComposerHeight(composerShell.getBoundingClientRect().height);
    };

    updateComposerHeight();

    const observer = new ResizeObserver(() => {
      updateComposerHeight();
    });

    observer.observe(composerShell);

    return () => {
      observer.disconnect();
    };
  }, []);

  const lastHistoryItem = history[history.length - 1];

  useEffect(() => {
    if (!lastHistoryItem) {
      return;
    }

    bottomRef.current?.scrollIntoView({
      behavior: lastHistoryItem.status === 'loading' ? 'auto' : 'smooth',
      block: 'end'
    });
  }, [lastHistoryItem]);

  const wikiParams = useMemo(() => {
    const p = new URLSearchParams(searchParams.toString());
    p.delete('question');
    p.delete('deep_research');
    return p.toString();
  }, [searchParams]);

  const backHref = wikiParams
    ? `/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}?${wikiParams}`
    : `/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`;

  const effectiveModel = isCustomSelectedModel ? customSelectedModel.trim() : selectedModel.trim();
  const isGenerating = history.some((item) => item.status === 'loading');
  const isComposerDisabled = isGenerating || isModelLoading || !selectedProvider.trim() || !effectiveModel;

  return (
    <div className="min-h-screen bg-white p-4 md:p-8">
      <header className="max-w-[95%] xl:max-w-[1400px] mx-auto mb-5">
        <div className="flex items-center justify-between gap-3">
          <Link href={backHref} className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors">
            <FaArrowLeft /> {messages.repoPage?.home || 'Back to Wiki'}
          </Link>
          <ThemeToggle />
        </div>
      </header>

      <main
        className="max-w-[95%] xl:max-w-[1400px] mx-auto space-y-4"
        style={{ paddingBottom: `${composerHeight + 24}px` }}
      >
        {history.length === 0 ? (
          <div className="rounded-xl border border-dashed border-gray-300 bg-white p-10 text-center text-gray-400">
            Enter a question above to view the answer.
          </div>
        ) : (
          history.map((item) => (
            <AskResultView key={item.id} item={item} />
          ))
        )}

        <div ref={bottomRef} />
      </main>

      <div className="fixed inset-x-0 bottom-0 z-40 border-t border-gray-200 bg-white/95 backdrop-blur">
        <div
          ref={composerShellRef}
          className="max-w-[95%] xl:max-w-[1400px] mx-auto px-4 py-3 md:px-8"
        >
          <AskComposer
            value={questionInput}
            onValueChange={setQuestionInput}
            deepResearch={deepResearchInput}
            onDeepResearchChange={setDeepResearchInput}
            onSubmit={handleAskSubmit}
            isSubmitting={isGenerating}
            disabled={isComposerDisabled}
            autoFocus={!initialQuestion}
          />
        </div>
      </div>
    </div>
  );
}


