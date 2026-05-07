import { DayDivider } from "../shell/DayDivider";
import { Message } from "./Message";
import type { ChatMessage } from "./types";

interface ChatStreamProps {
  messages: ChatMessage[];
}

function dayKey(timestamp: number) {
  return new Date(timestamp).toDateString();
}

function dayLabel(timestamp: number) {
  const today = dayKey(Date.now());
  const day = dayKey(timestamp);
  if (day === today) {
    return "Today";
  }
  return new Intl.DateTimeFormat("en", { month: "long", day: "numeric", year: "numeric" }).format(timestamp);
}

export function ChatStream({ messages }: ChatStreamProps) {
  if (messages.length === 0) {
    return (
      <div className="mx-auto flex max-w-[560px] flex-col items-center justify-center py-28 text-center">
        <p className="font-serif text-[24px] leading-snug text-[var(--ink-secondary)]">
          Ask about the repository structure, decisions, or implementation details.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-[760px] flex-col gap-9 px-10 pb-8 pt-12">
      {messages.map((message, index) => {
        const previous = messages[index - 1];
        const showDivider = !previous || dayKey(previous.timestamp) !== dayKey(message.timestamp);
        return (
          <div className="flex flex-col gap-9" key={message.id}>
            {showDivider ? <DayDivider>{dayLabel(message.timestamp)}</DayDivider> : null}
            <Message message={message} />
          </div>
        );
      })}
    </div>
  );
}
