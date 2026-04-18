export type AskHistoryStatus = 'loading' | 'done' | 'error';

export interface AskHistoryItem {
  id: string;
  question: string;
  deepResearch: boolean;
  status: AskHistoryStatus;
  response: string;
  error: string | null;
  researchIteration: number;
  researchComplete: boolean;
  provider: string;
  model: string;
  isCustomModel: boolean;
  customModel: string;
  language: string;
}
