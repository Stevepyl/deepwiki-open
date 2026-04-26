'use client';

import React from 'react';
import { AskCitation } from '@/types/ask';

interface CitationPanelProps {
  citations: AskCitation[];
  activeCitationIndex: number | null;
  title?: string;
  onCitationSelect?: (citationIndex: number) => void;
}

const formatLocation = (citation: AskCitation) => {
  const startLine = citation.start_line || 0;
  const endLine = citation.end_line || 0;
  if (startLine && endLine) {
    return `${citation.file_path}:${startLine}-${endLine}`;
  }
  if (startLine) {
    return `${citation.file_path}:${startLine}`;
  }
  return citation.file_path || 'unknown';
};

const CitationPanel: React.FC<CitationPanelProps> = ({
  citations,
  activeCitationIndex,
  title = '参考代码',
  onCitationSelect,
}) => {
  const activeCardRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    activeCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [activeCitationIndex]);

  return (
    <aside className="rounded-lg border border-gray-200 bg-white shadow-sm overflow-hidden xl:sticky xl:top-4 xl:self-start xl:max-h-[calc(100vh-2rem)]">
      <div className="border-b border-gray-200 bg-gray-50 px-4 py-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-gray-800">{title}</h2>
        {citations.length > 0 && (
          <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
            {citations.length}
          </span>
        )}
      </div>

      {citations.length === 0 ? (
        <div className="px-4 py-8 text-center text-sm text-gray-500">
          提问后，相关代码引用会显示在这里。
        </div>
      ) : (
        <div className="max-h-[520px] xl:max-h-[calc(100vh-7rem)] overflow-y-auto p-3 space-y-3">
          {citations.map((citation) => {
            const isActive = activeCitationIndex === citation.index;
            const location = formatLocation(citation);
            const meta = [citation.symbol, citation.chunk_type].filter(Boolean).join(' · ');

            return (
              <div
                key={`${citation.index}-${location}`}
                ref={isActive ? activeCardRef : null}
                role="button"
                tabIndex={0}
                onClick={() => onCitationSelect?.(citation.index)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onCitationSelect?.(citation.index);
                  }
                }}
                className={`w-full overflow-hidden rounded-lg border text-left transition-colors ${
                  isActive
                    ? 'border-blue-500 bg-blue-50 shadow-sm'
                    : 'border-gray-200 bg-white hover:border-blue-300'
                }`}
              >
                <div className="px-3 py-2 border-b border-gray-200 bg-gray-50">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="font-mono text-xs font-semibold text-blue-700 break-all">
                        [{citation.index}] {location}
                      </div>
                      {meta && (
                        <div className="mt-1 text-xs text-gray-500 break-words">
                          {meta}
                        </div>
                      )}
                    </div>
                    <span className="shrink-0 rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] text-gray-600">
                      {citation.score.toFixed(2)}
                    </span>
                  </div>
                </div>
                <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words bg-gray-950 px-3 py-3 font-mono text-xs leading-relaxed text-gray-100">
                  {citation.snippet || 'No snippet available.'}
                </pre>
              </div>
            );
          })}
        </div>
      )}
    </aside>
  );
};

export default CitationPanel;
