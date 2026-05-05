export interface ChatToolEvent {
  id: string;
  toolName: string;
  status: "running" | "complete" | "error";
  startedAt: number;
  args?: Record<string, unknown>;
  durationMs?: number;
  resultSummary?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  model?: string;
  citations?: string[];
  toolEvents?: ChatToolEvent[];
  streaming?: boolean;
  error?: string;
}
