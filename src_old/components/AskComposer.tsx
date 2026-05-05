'use client';

import React, { useEffect, useRef } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

export interface AskSubmitOptions {
  deepResearch: boolean;
}

interface AskComposerProps {
  onSubmit: (question: string, options: AskSubmitOptions) => void | Promise<void>;
  value: string;
  onValueChange: (value: string) => void;
  deepResearch: boolean;
  onDeepResearchChange: (value: boolean) => void;
  isSubmitting?: boolean;
  compact?: boolean;
  disabled?: boolean;
  autoFocus?: boolean;
  className?: string;
}

const AskComposer: React.FC<AskComposerProps> = ({
  onSubmit,
  value,
  onValueChange,
  deepResearch,
  onDeepResearchChange,
  isSubmitting = false,
  compact = false,
  disabled = false,
  autoFocus = false,
  className = ''
}) => {
  const { messages } = useLanguage();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const maxHeight = compact ? 140 : 260;

  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [autoFocus]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = 'auto';
    const nextHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
  }, [value, compact, maxHeight]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (disabled || isSubmitting) {
      return;
    }

    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }

    await onSubmit(trimmed, { deepResearch });
  };

  const handleKeyDown = async (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== 'Enter') {
      return;
    }

    if (e.shiftKey) {
      return;
    }

    if (!e.ctrlKey && !e.metaKey) {
      return;
    }

    e.preventDefault();

    if (busy) {
      return;
    }

    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }

    await onSubmit(trimmed, { deepResearch });
  };

  const busy = disabled || isSubmitting;

  return (
    <form
      onSubmit={handleSubmit}
      className={`rounded-xl border border-gray-200 bg-white shadow-sm ${compact ? 'p-2.5' : 'p-3.5'} ${className}`}
    >
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={messages.ask?.placeholder || 'What would you like to know about this codebase?'}
          className={`w-full rounded-lg border border-gray-300 bg-white text-gray-900 px-3 py-2 resize-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 focus:outline-none ${
            compact ? 'text-sm leading-5' : 'text-base leading-6'
          }`}
          style={{
            minHeight: compact ? '40px' : '56px',
            maxHeight: `${maxHeight}px`
          }}
          rows={compact ? 1 : 2}
          disabled={busy}
        />
        <button
          type="submit"
          disabled={busy || !value.trim()}
          className={`h-10 rounded-lg px-4 text-sm font-medium transition-colors ${
            busy || !value.trim()
              ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {isSubmitting ? (messages.common?.loading || 'Loading...') : (messages.ask?.askButton || 'Ask')}
        </button>
      </div>

      <div className="mt-2 flex items-center">
        <label className="flex items-center cursor-pointer select-none">
          <input
            type="checkbox"
            checked={deepResearch}
            onChange={() => onDeepResearchChange(!deepResearch)}
            className="sr-only"
            disabled={busy}
          />
          <div className={`relative w-10 h-5 rounded-full transition-colors ${deepResearch ? 'bg-blue-600' : 'bg-gray-300'}`}>
            <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${deepResearch ? 'translate-x-5' : ''}`}></div>
          </div>
          <span className="ml-2 text-xs text-gray-600">
            {messages.ask?.deepResearch || 'Deep Research'}
          </span>
        </label>
      </div>
    </form>
  );
};

export default AskComposer;
