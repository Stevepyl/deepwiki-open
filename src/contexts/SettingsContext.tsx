"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

const MODEL_KEY = "opswiki.modelSelection";
const AUTH_KEY = "opswiki.authCode";
const SETTINGS_KEY = "opswiki.settings";

export interface SettingsContextValue {
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
    const modelSelection = readJson(MODEL_KEY, { provider: "", model: "" });
    const settings = readJson(SETTINGS_KEY, {
      token: "",
      excludedDirs: "",
      excludedFiles: "",
      includedDirs: "",
      includedFiles: "",
    });

    setProvider(modelSelection.provider);
    setModel(modelSelection.model);
    setAuthCode(window.localStorage.getItem(AUTH_KEY) ?? "");
    setToken(settings.token);
    setExcludedDirs(settings.excludedDirs);
    setExcludedFiles(settings.excludedFiles);
    setIncludedDirs(settings.includedDirs);
    setIncludedFiles(settings.includedFiles);
    setHydrated(true);
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
    [authCode, excludedDirs, excludedFiles, includedDirs, includedFiles, model, provider, token],
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
