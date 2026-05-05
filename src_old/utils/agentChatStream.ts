import type { AgentChatEvent, AgentChatRequest } from '@/types/agentChat';

export async function streamAgentChatHttp(
  request: AgentChatRequest,
  onEvent: (event: AgentChatEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch('/api/chat/agent-stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/x-ndjson',
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Agent chat stream failed: ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error('Agent chat stream response body is empty');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.trim()) {
          onEvent(JSON.parse(line) as AgentChatEvent);
        }
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      onEvent(JSON.parse(buffer) as AgentChatEvent);
    }
  } finally {
    reader.releaseLock();
  }
}
