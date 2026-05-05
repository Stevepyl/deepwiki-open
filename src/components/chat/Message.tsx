import Markdown from "../Markdown";
import { Citation } from "./Citation";
import { ToolEvent } from "./ToolEvent";
import type { ChatMessage } from "./types";

interface MessageProps {
  message: ChatMessage;
}

function formatTime(timestamp: number) {
  return new Intl.DateTimeFormat("en", { hour: "numeric", minute: "2-digit" }).format(timestamp);
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === "user";
  const roleLabel = isUser ? "You" : message.model ? `AI · ${message.model}` : "AI";
  const visibleContent = message.content || (message.streaming ? "Thinking..." : "");

  return (
    <article className={`flex items-start gap-4 ${isUser ? "message--user" : "message--ai"}`}>
      <div
        className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full font-[var(--font-serif)] text-sm font-medium ${
          isUser
            ? "bg-[var(--ink-secondary)] text-[var(--paper-main)]"
            : "border border-[var(--hairline)] bg-[var(--paper-panel)] text-[var(--accent)] italic"
        }`}
        aria-hidden="true"
      >
        {isUser ? "S" : "A"}
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-2.5 flex items-center gap-2.5 font-[var(--font-sans)] text-[11px] font-medium uppercase text-[var(--ink-muted)]">
          <span>{roleLabel}</span>
          <span className="font-normal normal-case text-[var(--ink-faint)]">{formatTime(message.timestamp)}</span>
          {message.streaming ? <span className="text-[var(--accent)]">Streaming</span> : null}
        </div>
        <div className="font-[var(--font-serif)] text-base leading-[1.7] text-[var(--ink-primary)] [&_p:first-child]:mt-0 [&_p:last-child]:mb-0">
          <Markdown content={visibleContent} />
        </div>
        {message.error ? (
          <div className="mt-3 rounded-[var(--radius-sm)] border border-[var(--accent-line)] bg-[var(--accent-soft)] px-3 py-2 text-sm text-[var(--accent)]">
            {message.error}
          </div>
        ) : null}
        {message.toolEvents?.length ? (
          <div className="mt-4 flex flex-col gap-2">
            {message.toolEvents.map((event) => (
              <ToolEvent event={event} key={event.id} />
            ))}
          </div>
        ) : null}
        {message.citations?.length ? (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {message.citations.map((citation, index) => (
              <Citation index={index + 1} key={`${citation}-${index}`} path={citation} />
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}
