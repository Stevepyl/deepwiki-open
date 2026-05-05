export interface AgentChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface AgentChatRequest {
  repo_url: string;
  type?: string;
  token?: string;
  provider?: string;
  model?: string;
  language?: string;
  messages: AgentChatMessage[];
  agent_name?: 'wiki' | 'explore' | 'deep-research';
  excluded_dirs?: string;
  excluded_files?: string;
  included_dirs?: string;
  included_files?: string;
}

export interface TextDeltaEvent {
  type: 'text_delta';
  content: string;
}

export interface ToolCallStartEvent {
  type: 'tool_call_start';
  tool_call_id: string;
  tool_name: string;
  tool_args: Record<string, unknown>;
}

export interface ToolCallEndEvent {
  type: 'tool_call_end';
  tool_call_id: string;
  tool_name: string;
  result_summary: string;
  is_error: boolean;
  duration_ms: number;
  metadata: Record<string, unknown>;
}

export interface FinishAgentEvent {
  type: 'finish';
  finish_reason: string;
  usage?: Record<string, number> | null;
}

export interface ErrorAgentEvent {
  type: 'error';
  error: string;
  code?: string | null;
}

export type AgentChatEvent =
  | TextDeltaEvent
  | ToolCallStartEvent
  | ToolCallEndEvent
  | FinishAgentEvent
  | ErrorAgentEvent;
