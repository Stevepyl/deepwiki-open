'use client';

import React from 'react';
import Markdown from './Markdown';
import { AskHistoryItem } from '@/types/ask';

interface AskResultViewProps {
  item: AskHistoryItem;
}

const AskResultView: React.FC<AskResultViewProps> = ({ item }) => {
  const downloadResponse = () => {
    if (!item.response.trim()) {
      return;
    }

    const content = `## Question\n${item.question}\n\n## Answer\n${item.response}`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `answer-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      <div className="border-b border-gray-200 px-5 py-3 bg-gray-50">
        <div className="text-xs text-gray-500 mb-1">
          {item.deepResearch
            ? `Deep Research${item.researchIteration > 0 ? ` - Iteration ${item.researchIteration}` : ''}${item.researchComplete ? ' - Complete' : ''}`
            : 'Standard Answer'}
        </div>
        <h2 className="text-base font-semibold text-gray-800 break-words">{item.question}</h2>
      </div>

      <div className="px-5 py-5 min-h-[240px]">
        {item.error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 text-red-700 px-4 py-3 text-sm">
            {item.error}
          </div>
        ) : item.response ? (
          <div className="prose prose-sm md:prose-base lg:prose-lg max-w-none">
            <Markdown content={item.response} />
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span className="h-2.5 w-2.5 rounded-full bg-blue-500 animate-pulse"></span>
            <span>{item.status === 'loading' ? 'Generating answer...' : 'Waiting for response...'}</span>
          </div>
        )}
      </div>

      <div className="border-t border-gray-200 px-5 py-3 flex justify-end">
        <button
          type="button"
          onClick={downloadResponse}
          disabled={!item.response.trim()}
          className={`text-sm ${
            item.response.trim() ? 'text-gray-600 hover:text-green-600' : 'text-gray-300 cursor-not-allowed'
          } transition-colors`}
        >
          Download
        </button>
      </div>
    </div>
  );
};

export default AskResultView;
