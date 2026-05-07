"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

const MODEL_KEY = "opswiki.modelSelection";
const AUTH_KEY = "opswiki.authCode";
const SETTINGS_KEY = "opswiki.settings";

export interface ModelConfigModel {
  id: string;
  name: string;
}

export interface ModelConfigProvider {
  id: string;
  name: string;
  supportsCustomModel?: boolean;
  defaultModel?: string;
  models: ModelConfigModel[];
}

export interface ModelConfig {
  providers: ModelConfigProvider[];
  defaultProvider: string;
}

export interface SettingsContextValue {
  hydrated: boolean;
  provider: string;
  model: string;
  authCode: string;
  token: string;
  excludedDirs: string;
  excludedFiles: string;
  includedDirs: string;
  includedFiles: string;
  setProvider: (value: string) => void;
  setModel: (value: string) => void;
  setAuthCode: (value: string) => void;
  setToken: (value: string) => void;
  setExcludedDirs: (value: string) => void;
  setExcludedFiles: (value: string) => void;
  setIncludedDirs: (value: string) => void;
  setIncludedFiles: (value: string) => void;
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined);

function readJson<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") {
    return fallback;
  }
  const raw = window.localStorage.getItem(key);
  return raw ? (JSON.parse(raw) as T) : fallback;
}

export function normalizeModelSelection(
  config: ModelConfig,
  selection: { provider: string; model: string },
) {
  const providerConfig =
    config.providers.find((item) => item.id === selection.provider) ??
    config.providers.find((item) => item.id === config.defaultProvider) ??
    config.providers[0];

  if (!providerConfig) {
    return selection;
  }

  const modelExists = providerConfig.models.some((item) => item.id === selection.model);
  const model = modelExists ? selection.model : providerConfig.defaultModel ?? providerConfig.models[0]?.id ?? "";

  return {
    provider: providerConfig.id,
    model,
  };
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [hydrated, setHydrated] = useState(false);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [authCode, setAuthCode] = useState("");
  const [token, setToken] = useState("");
  const [excludedDirs, setExcludedDirs] = useState("");
  const [excludedFiles, setExcludedFiles] = useState("");
  const [includedDirs, setIncludedDirs] = useState("");
  const [includedFiles, setIncludedFiles] = useState("");

  useEffect(() => {
    let cancelled = false;
    const modelSelection = readJson(MODEL_KEY, { provider: "", model: "" });
    const settings = readJson(SETTINGS_KEY, {
      token: "",
      excludedDirs: "",
      excludedFiles: "",
      includedDirs: "",
      includedFiles: "",
    });

    const applySettings = (nextSelection: { provider: string; model: string }) => {
      if (cancelled) {
        return;
      }
      setProvider(nextSelection.provider);
      setModel(nextSelection.model);
      setAuthCode(window.localStorage.getItem(AUTH_KEY) ?? "");
      setToken(settings.token);
      setExcludedDirs(settings.excludedDirs);
      setExcludedFiles(settings.excludedFiles);
      setIncludedDirs(settings.includedDirs);
      setIncludedFiles(settings.includedFiles);
      setHydrated(true);
    };

    fetch("/api/models/config")
      .then((response) => (response.ok ? response.json() : null))
      .then((config: ModelConfig | null) => {
        applySettings(config ? normalizeModelSelection(config, modelSelection) : modelSelection);
      })
      .catch(() => applySettings(modelSelection));

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    window.localStorage.setItem(MODEL_KEY, JSON.stringify({ provider, model }));
    window.localStorage.setItem(AUTH_KEY, authCode);
    window.localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ token, excludedDirs, excludedFiles, includedDirs, includedFiles }),
    );
  }, [authCode, excludedDirs, excludedFiles, hydrated, includedDirs, includedFiles, model, provider, token]);

  const value = useMemo<SettingsContextValue>(
    () => ({
      hydrated,
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
    }),
    [authCode, excludedDirs, excludedFiles, hydrated, includedDirs, includedFiles, model, provider, token],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error("useSettings must be used within SettingsProvider");
  }
  return context;
}
