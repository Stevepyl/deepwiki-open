export type AskHistoryStatus = 'loading' | 'done' | 'error';

export interface AskCitation {
  index: number;
  file_path: string;
  start_line: number;
  end_line: number;
  symbol: string;
  chunk_type: string;
  score: number;
  snippet: string;
}

export interface AskHistoryItem {
  id: string;
  question: string;
  deepResearch: boolean;
  status: AskHistoryStatus;
  response: string;
  citations: AskCitation[];
  error: string | null;
  researchIteration: number;
  researchComplete: boolean;
  provider: string;
  model: string;
  isCustomModel: boolean;
  customModel: string;
  language: string;
}
