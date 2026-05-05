"use client";

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { buildRepoInfo } from "../../../../components/chat/runtime";
import { Composer } from "../../../../components/shell/Composer";
import { AppShell } from "../../../../components/shell/AppShell";
import { SettingsPanel } from "../../../../components/shell/SettingsPanel";
import { Switcher } from "../../../../components/shell/Switcher";
import { Topbar } from "../../../../components/shell/Topbar";
import { WikiTopbarActions } from "../../../../components/wiki/WikiTopbarActions";
import { WorkshopArticle } from "../../../../components/workshop/WorkshopArticle";
import { WorkshopRail } from "../../../../components/workshop/WorkshopRail";
import type { WorkshopStep } from "../../../../components/workshop/types";
import { useLanguage } from "../../../../contexts/LanguageContext";
import { useSettings } from "../../../../contexts/SettingsContext";
import { exportWiki, fetchWikiCache, repoLabel, type WikiCache } from "../../../../utils/wiki";
import { fallbackWorkshopSteps, generateWorkshopSteps, workshopStorageKey } from "../../../../utils/workshop";

export default function WorkshopPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const owner = decodeURIComponent(String(params.owner));
  const repo = decodeURIComponent(String(params.repo));
  const settings = useSettings();
  const { language } = useLanguage();
  const repoInfo = useMemo(() => buildRepoInfo(owner, repo, settings.token, searchParams), [owner, repo, searchParams, settings.token]);
  const [cache, setCache] = useState<WikiCache | null>(null);
  const [steps, setSteps] = useState<WorkshopStep[]>([]);
  const [draft, setDraft] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const basePath = `/${owner}/${repo}`;

  useEffect(() => {
    let ignore = false;
    fetchWikiCache(owner, repo, repoInfo.type, language).then((data) => {
      if (ignore) return;
      if (!data) {
        router.replace(`${basePath}?status=generating`);
        return;
      }
      setCache(data);
      const fallback = fallbackWorkshopSteps(data);
      const key = workshopStorageKey(repoInfo, language);
      const cached = window.localStorage.getItem(key);
      if (cached) {
        setSteps(JSON.parse(cached) as WorkshopStep[]);
        return;
      }
      setSteps(fallback);
      generateWorkshopSteps(repoInfo, data, language, settings)
        .then((generated) => {
          window.localStorage.setItem(key, JSON.stringify(generated));
          setSteps(generated);
        })
        .catch(() => window.localStorage.setItem(key, JSON.stringify(fallback)));
    });
    return () => {
      ignore = true;
    };
  }, [basePath, language, owner, repo, repoInfo, router, settings]);

  const submit = () => {
    const question = draft.trim();
    if (!question) return;
    const next = new URLSearchParams(searchParams.toString());
    next.delete("step");
    next.set("q", question);
    router.push(`${basePath}/ask?${next.toString()}`);
  };

  const activeStepId = searchParams.get("step") ?? steps[0]?.id ?? "";

  return (
    <AppShell
      topbar={
        <Topbar
          breadcrumb={
            <>
              <span className="font-[var(--font-mono)] text-xs text-[var(--ink-primary)]">{repoLabel(owner, repo, repoInfo.type)}</span>
              <span className="text-[var(--ink-faint)]">/</span>
              <span className="font-[var(--font-serif)] text-xs italic text-[var(--ink-secondary)]">Hands-on workshop</span>
            </>
          }
          switcher={<Switcher owner={owner} repo={repo} />}
          actions={
            cache ? (
              <WikiTopbarActions slidesHref={`${basePath}/slides`} onExport={(format) => void exportWiki(cache, repoInfo, format)} />
            ) : null
          }
        />
      }
      composer={
        <Composer
          variant="wiki"
          modeHint="Ask wiki"
          placeholder="Ask a workshop follow-up..."
          value={draft}
          onChange={setDraft}
          onSubmit={submit}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      }
    >
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <section className="grid max-w-[1100px] grid-cols-[220px_1fr] gap-12 px-10 pb-6 pt-10 max-[1100px]:grid-cols-1">
        {steps.length === 0 ? (
          <div className="font-[var(--font-serif)] text-[16px] italic text-[var(--ink-muted)]">Preparing workshop...</div>
        ) : (
          <>
            <div className="max-[1100px]:hidden">
              <WorkshopRail activeStepId={activeStepId} basePath={`${basePath}/workshop`} steps={steps} />
            </div>
            <WorkshopArticle activeStepId={activeStepId} steps={steps} />
          </>
        )}
      </section>
    </AppShell>
  );
}
