/**
 * WebSocket client for chat completions
 * This replaces the HTTP streaming endpoint with a WebSocket connection
 */

import type { AgentChatEvent, AgentChatRequest } from '../types/agentChat';
import type { WikiStructure } from '../types/wiki/wikistructure';

// Get the server base URL from environment or use default
const SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

// Convert HTTP URL to WebSocket URL
const getWebSocketUrl = () => {
  const baseUrl = SERVER_BASE_URL;
  // Replace http:// with ws:// or https:// with wss://
  const wsBaseUrl = baseUrl.replace(/^http/, 'ws');
  return `${wsBaseUrl}/ws/chat`;
};

const getAgentChatWebSocketUrl = () => {
  const wsBaseUrl = SERVER_BASE_URL.replace(/^http/, 'ws');
  return `${wsBaseUrl}/ws/agent-chat`;
};

const getAgentWikiWebSocketUrl = () => {
  const wsBaseUrl = SERVER_BASE_URL.replace(/^http/, 'ws');
  return `${wsBaseUrl}/ws/agent-wiki`;
};

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatCompletionRequest {
  repo_url: string;
  messages: ChatMessage[];
  filePath?: string;
  token?: string;
  type?: string;
  provider?: string;
  model?: string;
  language?: string;
  excluded_dirs?: string;
  excluded_files?: string;
  included_dirs?: string;
  included_files?: string;
}

export interface AgentWikiRequest {
  repo_url: string;
  type?: string;
  token?: string;
  provider?: string;
  model?: string;
  language?: string;
  comprehensive?: boolean;
  file_tree_hint?: string;
  readme_hint?: string;
  excluded_dirs?: string;
  excluded_files?: string;
  included_dirs?: string;
  included_files?: string;
}

export type AgentWikiEvent =
  | { type: 'text_delta'; content: string; phase?: 'planning' | 'writing'; page_index?: number; page_id?: string }
  | { type: 'tool_call_start'; tool_call_id: string; tool_name: string; tool_args?: Record<string, unknown>; phase?: 'planning' | 'writing'; page_index?: number; page_id?: string }
  | { type: 'tool_call_end'; tool_call_id: string; tool_name: string; result_summary: string; is_error?: boolean; duration_ms?: number; phase?: 'planning' | 'writing'; page_index?: number; page_id?: string }
  | { type: 'wiki_structure_ready'; structure: WikiStructure }
  | { type: 'wiki_structure_error'; code: string; message: string }
  | { type: 'wiki_page_done'; page_id: string; page_title: string; page_index: number; total_pages: number; content: string }
  | { type: 'wiki_page_error'; page_id: string; page_index: number; code: string; message: string }
  | { type: 'error'; error: string; code?: string }
  | { type: 'finish'; finish_reason: string; usage?: Record<string, number> };

/**
 * Creates a WebSocket connection for chat completions
 * @param request The chat completion request
 * @param onMessage Callback for received messages
 * @param onError Callback for errors
 * @param onClose Callback for when the connection closes
 * @returns The WebSocket connection
 */
export const createChatWebSocket = (
  request: ChatCompletionRequest,
  onMessage: (message: string) => void,
  onError: (error: Event) => void,
  onClose: () => void
): WebSocket => {
  // Create WebSocket connection
  const ws = new WebSocket(getWebSocketUrl());
  
  // Set up event handlers
  ws.onopen = () => {
    console.log('WebSocket connection established');
    // Send the request as JSON
    ws.send(JSON.stringify(request));
  };
  
  ws.onmessage = (event) => {
    // Call the message handler with the received text
    onMessage(event.data);
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onError(error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket connection closed');
    onClose();
  };
  
  return ws;
};

export const createAgentChatWebSocket = (
  request: AgentChatRequest,
  onEvent: (event: AgentChatEvent) => void,
  onError: (error: Event) => void,
  onClose: () => void
): WebSocket => {
  const ws = new WebSocket(getAgentChatWebSocketUrl());

  ws.onopen = () => {
    ws.send(JSON.stringify(request));
  };

  ws.onmessage = (event) => {
    onEvent(JSON.parse(event.data) as AgentChatEvent);
  };

  ws.onerror = (error) => {
    onError(error);
  };

  ws.onclose = () => {
    onClose();
  };

  return ws;
};

export const createAgentWikiWebSocket = (
  request: AgentWikiRequest,
  onEvent: (event: AgentWikiEvent) => void,
  onError: (error: Event) => void,
  onClose: () => void
): WebSocket => {
  const ws = new WebSocket(getAgentWikiWebSocketUrl());

  ws.onopen = () => {
    ws.send(JSON.stringify(request));
  };

  ws.onmessage = (event) => {
    onEvent(JSON.parse(event.data) as AgentWikiEvent);
  };

  ws.onerror = (error) => {
    onError(error);
  };

  ws.onclose = () => {
    onClose();
  };

  return ws;
};

/**
 * Closes a WebSocket connection
 * @param ws The WebSocket connection to close
 */
export const closeWebSocket = (ws: WebSocket | null): void => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
};
