"use client";

import { useState } from "react";
import { Composer } from "../shell/Composer";
import { SettingsPanel } from "../shell/SettingsPanel";
import type { AgentName } from "./runtime";

interface ChatComposerProps {
  value: string;
  streaming: boolean;
  deepResearch: boolean;
  agentName: AgentName;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onAgentNameChange: (value: AgentName) => void;
}

export function ChatComposer({
  value,
  streaming,
  deepResearch,
  agentName,
  onChange,
  onSubmit,
  onAgentNameChange,
}: ChatComposerProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <>
      <Composer
        disabled={streaming}
        modeHint={streaming ? "Streaming" : deepResearch ? "Deep" : agentName}
        placeholder="Ask about this repository..."
        value={value}
        onChange={onChange}
        onSubmit={onSubmit}
        onOpenSettings={() => setSettingsOpen(true)}
        variant="chat"
        footer={
          <>
            <label className="flex items-center gap-2">
              <span>Agent</span>
              <select
                className="rounded border border-[var(--hairline)] bg-[var(--paper-panel)] px-1.5 py-0.5 text-[var(--ink-muted)]"
                disabled={deepResearch || streaming}
                value={agentName}
                onChange={(event) => onAgentNameChange(event.target.value as AgentName)}
              >
                <option value="explore">Explore</option>
                <option value="wiki">Wiki</option>
              </select>
            </label>
            <span>Press <kbd>↵</kbd> to send</span>
          </>
        }
      />
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
