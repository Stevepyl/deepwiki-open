"use client";

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { GenerationLoader } from "../../../components/generation/GenerationLoader";
import { Composer } from "../../../components/shell/Composer";
import { AppShell } from "../../../components/shell/AppShell";
import { SettingsPanel } from "../../../components/shell/SettingsPanel";
import { Switcher } from "../../../components/shell/Switcher";
import { Topbar } from "../../../components/shell/Topbar";
import { WikiArticle } from "../../../components/wiki/WikiArticle";
import { WikiToc } from "../../../components/wiki/WikiToc";
import { WikiTopbarActions } from "../../../components/wiki/WikiTopbarActions";
import { useLanguage } from "../../../contexts/LanguageContext";
import { useSettings } from "../../../contexts/SettingsContext";
import { buildRepoInfo } from "../../../components/chat/runtime";
import {
  exportWiki,
  fetchWikiCache,
  findWikiPage,
  pageSectionTitle,
  repoLabel,
  type WikiCache,
  wikiSectionViews,
} from "../../../utils/wiki";

export default function WikiPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const owner = decodeURIComponent(String(params.owner));
  const repo = decodeURIComponent(String(params.repo));
  const settings = useSettings();
  const { language } = useLanguage();
  const [cache, setCache] = useState<WikiCache | null>(null);
  const [loading, setLoading] = useState(true);
  const [draft, setDraft] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const basePath = `/${owner}/${repo}`;
  const generating = searchParams.get("status") === "generating";
  const repoInfo = useMemo(() => buildRepoInfo(owner, repo, settings.token, searchParams), [owner, repo, searchParams, settings.token]);

  useEffect(() => {
    if (generating) return;
    let ignore = false;
    setLoading(true);
    fetchWikiCache(owner, repo, repoInfo.type, language)
      .then((data) => {
        if (ignore) return;
        if (!data) {
          router.replace(`${basePath}?status=generating`);
          return;
        }
        setCache(data);
      })
      .finally(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, [basePath, generating, language, owner, repo, repoInfo.type, router]);

  const submit = () => {
    const question = draft.trim();
    if (!question) return;
    const next = new URLSearchParams(searchParams.toString());
    next.delete("page");
    next.delete("status");
    next.set("q", question);
    router.push(`${basePath}/ask?${next.toString()}`);
  };

  if (generating) {
    return <GenerationLoader owner={owner} repo={repo} repoInfo={repoInfo} language={language} settings={settings} />;
  }

  const activePage = cache ? findWikiPage(cache, searchParams.get("page")) : null;
  const sections = cache ? wikiSectionViews(cache) : [];
  const sectionTitle = cache && activePage ? pageSectionTitle(cache, activePage.id) : "Wiki";

  return (
    <AppShell
      className="font-medium [&_*]:font-medium"
      topbar={
        <Topbar
          breadcrumb={
            <>
              <span className="font-mono text-xs text-[var(--ink-primary)]">{repoLabel(owner, repo, repoInfo.type)}</span>
              <span className="text-[var(--ink-faint)]">/</span>
              <span className="font-serif text-xs italic text-[var(--ink-secondary)]">
                {activePage?.title ?? "Wiki"}
              </span>
            </>
          }
          switcher={<Switcher owner={owner} repo={repo} />}
          actions={
            cache ? (
              <WikiTopbarActions
                slidesHref={`${basePath}/slides`}
                onExport={(format) => {
                  void exportWiki(cache, repoInfo, format);
                }}
              />
            ) : null
          }
        />
      }
      composer={
        <Composer
          variant="wiki"
          modeHint="Ask wiki"
          placeholder="Ask a follow-up about this repository..."
          value={draft}
          onChange={setDraft}
          onSubmit={submit}
          onOpenSettings={() => setSettingsOpen(true)}
          footer={
            <>
              <span>
                <kbd>Enter</kbd> opens chat
              </span>
              <span>Reading cached wiki</span>
            </>
          }
        />
      }
    >
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <section className="grid max-w-[1100px] grid-cols-[240px_1fr] gap-12 px-10 pb-6 pt-10 max-[1100px]:grid-cols-1">
        {loading && <div className="font-serif text-[16px] italic text-[var(--ink-muted)]">Loading wiki...</div>}
        {activePage && (
          <>
            <div className="max-[1100px]:hidden">
              <WikiToc activePageId={activePage.id} basePath={basePath} sections={sections} />
            </div>
            <WikiArticle page={activePage} sectionTitle={sectionTitle} />
          </>
        )}
      </section>
    </AppShell>
  );
}
