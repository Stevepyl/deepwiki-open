"use client";

import { useEffect, useMemo, useState } from "react";
import { FiX } from "react-icons/fi";
import { useLanguage } from "../../contexts/LanguageContext";
import { useSettings } from "../../contexts/SettingsContext";
import { shellMessages } from "../../messages/en";
import { IconButton } from "./IconButton";

interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  supportsCustomModel?: boolean;
  models: Model[];
}

interface ModelConfig {
  providers: Provider[];
  defaultProvider: string;
}

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

const sectionTitleClass = "font-sans text-[11px] font-medium uppercase tracking-normal text-[var(--ink-muted)]";
const fieldClass = "flex flex-col gap-[7px] text-xs text-[var(--ink-secondary)]";
const controlClass =
  "w-full rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-main)] px-2.5 py-[9px] text-[var(--ink-primary)]";
const textareaClass = `${controlClass} min-h-[82px] resize-y font-mono text-xs`;
const sectionClass = "flex flex-col gap-3";
const actionButtonClass =
  "flex w-fit items-center gap-2 rounded-[var(--radius-sm)] px-3 py-2 font-sans text-[12.5px] font-medium text-[var(--ink-primary)] transition-colors duration-[120ms] hover:bg-[var(--paper-hover)]";

export function SettingsPanel({ open, onClose }: SettingsPanelProps) {
  const {
    provider,
    model,
    authCode,
    token,
    excludedDirs,
    excludedFiles,
    includedDirs,
    includedFiles,
    setProvider,
    setModel,
    setAuthCode,
    setToken,
    setExcludedDirs,
    setExcludedFiles,
    setIncludedDirs,
    setIncludedFiles,
  } = useSettings();
  const { language, setLanguage, supportedLanguages } = useLanguage();
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const [authStatus, setAuthStatus] = useState("");

  useEffect(() => {
    if (!open) {
      return;
    }
    fetch("/api/models/config")
      .then((response) => response.json())
      .then((data: ModelConfig) => {
        setModelConfig(data);
        if (!provider) {
          setProvider(data.defaultProvider);
          setModel(data.providers.find((item) => item.id === data.defaultProvider)?.models[0]?.id ?? "");
        }
      });

    fetch("/api/auth/status")
      .then((response) => response.json())
      .then((data: { auth_required: boolean }) => setAuthRequired(data.auth_required));
  }, [open, provider, setModel, setProvider]);

  const activeProvider = useMemo(
    () => modelConfig?.providers.find((item) => item.id === provider),
    [modelConfig, provider],
  );

  const validateAuth = async () => {
    const response = await fetch("/api/auth/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: authCode }),
    });
    const data = (await response.json()) as { success: boolean };
    setAuthStatus(data.success ? shellMessages.settings.authSaved : shellMessages.settings.authFailed);
  };

  if (!open) {
    return null;
  }

  return (
    <dialog
      className="fixed inset-0 z-[80] m-0 flex h-auto max-h-none w-screen max-w-none justify-end border-0 bg-[rgba(26,24,21,0.12)] p-0 text-[var(--ink-primary)]"
      aria-label={shellMessages.settings.title}
      open
    >
      <div className="flex h-full w-[min(420px,100vw)] flex-col border-l border-[var(--hairline-strong)] bg-[var(--paper-panel)] shadow-[-12px_0_36px_rgba(26,24,21,0.08)]">
        <header className="flex h-[var(--topbar-h)] items-center justify-between border-b border-[var(--hairline)] px-5">
          <div className="font-serif text-[17px] font-medium">{shellMessages.settings.title}</div>
          <IconButton aria-label={shellMessages.settings.close} onClick={onClose}>
            <FiX aria-hidden="true" />
          </IconButton>
        </header>

        <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-5 pb-8 pt-[22px] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb]:bg-[var(--hairline-strong)]">
          <section className={sectionClass}>
            <div className={sectionTitleClass}>{shellMessages.settings.model}</div>
            <div className="grid grid-cols-2 gap-2.5">
              <label className={fieldClass}>
                <span>{shellMessages.settings.provider}</span>
                <select
                  className={controlClass}
                  value={provider}
                  onChange={(event) => {
                    const providerId = event.target.value;
                    const nextProvider = modelConfig?.providers.find((item) => item.id === providerId);
                    setProvider(providerId);
                    setModel(nextProvider?.models[0]?.id ?? "");
                  }}
                >
                  {modelConfig?.providers.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className={fieldClass}>
                <span>{shellMessages.settings.model}</span>
                <select
                  className={controlClass}
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                >
                  {activeProvider?.models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </section>

          <section className={sectionClass}>
            <div className={sectionTitleClass}>{shellMessages.settings.authorization}</div>
            <label className={fieldClass}>
              <span>{authRequired ? shellMessages.settings.authorizationCode : shellMessages.settings.authorizationCodeOptional}</span>
              <input
                className={controlClass}
                value={authCode}
                onChange={(event) => setAuthCode(event.target.value)}
                type="password"
              />
            </label>
            <button className={actionButtonClass} type="button" onClick={validateAuth}>
              {shellMessages.settings.validateCode}
            </button>
            {authStatus && <div className="text-xs text-[var(--ink-muted)]">{authStatus}</div>}
          </section>

          <section className={sectionClass}>
            <div className={sectionTitleClass}>{shellMessages.settings.repository}</div>
            <label className={fieldClass}>
              <span>{shellMessages.settings.repositoryToken}</span>
              <input
                className={controlClass}
                value={token}
                onChange={(event) => setToken(event.target.value)}
                type="password"
              />
            </label>
          </section>

          <section className={sectionClass}>
            <div className={sectionTitleClass}>{shellMessages.settings.filters}</div>
            <label className={fieldClass}>
              <span>{shellMessages.settings.includedDirs}</span>
              <textarea className={textareaClass} value={includedDirs} onChange={(event) => setIncludedDirs(event.target.value)} />
            </label>
            <label className={fieldClass}>
              <span>{shellMessages.settings.includedFiles}</span>
              <textarea className={textareaClass} value={includedFiles} onChange={(event) => setIncludedFiles(event.target.value)} />
            </label>
            <label className={fieldClass}>
              <span>{shellMessages.settings.excludedDirs}</span>
              <textarea className={textareaClass} value={excludedDirs} onChange={(event) => setExcludedDirs(event.target.value)} />
            </label>
            <label className={fieldClass}>
              <span>{shellMessages.settings.excludedFiles}</span>
              <textarea className={textareaClass} value={excludedFiles} onChange={(event) => setExcludedFiles(event.target.value)} />
            </label>
          </section>

          <section className={sectionClass}>
            <div className={sectionTitleClass}>{shellMessages.settings.language}</div>
            <select className={controlClass} value={language} onChange={(event) => setLanguage(event.target.value)}>
              {Object.entries(supportedLanguages).map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          </section>
        </div>
      </div>
    </dialog>
  );
}
