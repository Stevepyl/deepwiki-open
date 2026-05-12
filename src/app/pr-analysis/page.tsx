'use client';

import Markdown from '@/components/Markdown';
import ThemeToggle from '@/components/theme-toggle';
import Link from 'next/link';
import React, { useMemo, useState } from 'react';
import { FaArrowLeft, FaGithub } from 'react-icons/fa';

type Mode = 'local' | 'github';

interface ImpactNode {
  id: string;
  role: string;
  relation?: string;
  target?: string;
  file?: string;
  line?: number;
  snippet?: string;
}

interface ImpactPath {
  path_id: string;
  changed_symbol: string;
  score: number;
  nodes: ImpactNode[];
}

interface Risk {
  rule: string;
  level: string;
  message: string;
  evidence_refs: string[];
}

interface AnalysisResult {
  session_id: string;
  report: string;
  risks: Risk[];
  impact_paths: ImpactPath[];
  changed_symbols: Array<Record<string, unknown>>;
  diff_summary: {
    file_count: number;
    python_file_count: number;
    changed_files: Array<{ path: string; change_type: string }>;
  };
}

export default function PrAnalysisPage() {
  const [mode, setMode] = useState<Mode>('local');
  const [repoPath, setRepoPath] = useState('D:\\my_lab\\deepwiki-open-z\\airflow-pr65718');
  const [base, setBase] = useState('base');
  const [head, setHead] = useState('head');
  const [prUrl, setPrUrl] = useState('');
  const [token, setToken] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [followupQuestion, setFollowupQuestion] = useState('');
  const [followupAnswer, setFollowupAnswer] = useState('');
  const [isAsking, setIsAsking] = useState(false);

  const canSubmit = useMemo(() => {
    if (isLoading) return false;
    if (mode === 'github') return prUrl.trim().length > 0;
    return repoPath.trim().length > 0 && base.trim().length > 0 && head.trim().length > 0;
  }, [base, head, isLoading, mode, prUrl, repoPath]);

  const runAnalysis = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);
    setFollowupAnswer('');

    try {
      const body = mode === 'github'
        ? { pr_url: prUrl.trim(), token: token.trim() || undefined }
        : { repo_path: repoPath.trim(), base: base.trim(), head: head.trim() };

      const response = await fetch('/api/pr-analysis/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || `Analysis failed with HTTP ${response.status}`);
      }

      setResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsLoading(false);
    }
  };

  const askFollowup = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!result || !followupQuestion.trim()) {
      return;
    }

    setIsAsking(true);
    setError('');
    try {
      const response = await fetch(`/api/pr-analysis/${result.session_id}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: followupQuestion.trim() }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || `Follow-up failed with HTTP ${response.status}`);
      }
      const payload = await response.json();
      setFollowupAnswer(payload.answer || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Follow-up failed');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-white p-4 md:p-8">
      <header className="max-w-6xl mx-auto mb-6 flex items-center justify-between">
        <Link href="/" className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700">
          <FaArrowLeft /> Back Home
        </Link>
        <ThemeToggle />
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)] gap-5">
        <section className="rounded-lg border border-gray-200 bg-white shadow-sm p-5 h-fit">
          <div className="mb-5">
            <h1 className="text-xl font-semibold text-gray-900">PR Impact Analysis</h1>
            <p className="text-sm text-gray-500 mt-1">Analyze Python symbol impacts, reverse references, and v1 risk rules.</p>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-5">
            <button
              type="button"
              onClick={() => setMode('local')}
              className={`rounded-md border px-3 py-2 text-sm ${mode === 'local' ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-gray-200 text-gray-600'}`}
            >
              Local Diff
            </button>
            <button
              type="button"
              onClick={() => setMode('github')}
              className={`inline-flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm ${mode === 'github' ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-gray-200 text-gray-600'}`}
            >
              <FaGithub /> GitHub PR
            </button>
          </div>

          <form onSubmit={runAnalysis} className="space-y-4">
            {mode === 'local' ? (
              <>
                <label className="block">
                  <span className="text-xs font-medium text-gray-600">Repository Path</span>
                  <input
                    value={repoPath}
                    onChange={(event) => setRepoPath(event.target.value)}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
                    placeholder="D:\my_lab\repo"
                  />
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <label className="block">
                    <span className="text-xs font-medium text-gray-600">Base Ref</span>
                    <input
                      value={base}
                      onChange={(event) => setBase(event.target.value)}
                      className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
                      placeholder="main"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs font-medium text-gray-600">Head Ref</span>
                    <input
                      value={head}
                      onChange={(event) => setHead(event.target.value)}
                      className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
                      placeholder="feature"
                    />
                  </label>
                </div>
              </>
            ) : (
              <>
                <label className="block">
                  <span className="text-xs font-medium text-gray-600">GitHub PR URL</span>
                  <input
                    value={prUrl}
                    onChange={(event) => setPrUrl(event.target.value)}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
                    placeholder="https://github.com/owner/repo/pull/123"
                  />
                </label>
                <label className="block">
                  <span className="text-xs font-medium text-gray-600">Token</span>
                  <input
                    value={token}
                    onChange={(event) => setToken(event.target.value)}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
                    placeholder="Optional for private repos"
                    type="password"
                  />
                </label>
              </>
            )}

            <button
              type="submit"
              disabled={!canSubmit}
              className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              {isLoading ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </form>

          {error && (
            <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
        </section>

        <section className="space-y-4 min-w-0">
          {!result ? (
            <div className="rounded-lg border border-dashed border-gray-300 p-10 text-center text-gray-400">
              Run an analysis to view impact paths and risks.
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Metric label="Files" value={result.diff_summary.file_count} />
                <Metric label="Python Files" value={result.diff_summary.python_file_count} />
                <Metric label="Changed Symbols" value={result.changed_symbols.length} />
                <Metric label="Risks" value={result.risks.length} />
              </div>

              <div className="rounded-lg border border-gray-200 bg-white shadow-sm p-5">
                <Markdown content={result.report} />
              </div>

              <div className="rounded-lg border border-gray-200 bg-white shadow-sm p-5">
                <h2 className="text-base font-semibold text-gray-900 mb-3">Risks</h2>
                {result.risks.length ? (
                  <div className="space-y-2">
                    {result.risks.map((risk) => (
                      <div key={risk.rule} className="rounded-md border border-red-100 bg-red-50 p-3">
                        <div className="text-sm font-semibold text-red-700">[{risk.level}] {risk.rule}</div>
                        <div className="text-sm text-red-800 mt-1">{risk.message}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">No v1 risk rule matched.</div>
                )}
              </div>

              <div className="rounded-lg border border-gray-200 bg-white shadow-sm p-5">
                <h2 className="text-base font-semibold text-gray-900 mb-3">Impact Paths</h2>
                {result.impact_paths.length ? (
                  <div className="space-y-3">
                    {result.impact_paths.map((path) => (
                      <div key={path.path_id} className="rounded-md border border-gray-200 p-3">
                        <div className="text-xs font-semibold text-gray-500">{path.path_id} score={path.score}</div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {path.nodes.map((node, index) => (
                            <span key={`${path.path_id}-${node.id}-${index}`} className="rounded bg-gray-100 px-2 py-1 font-mono text-xs text-gray-700">
                              {node.id}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">No reverse-reference impact path found.</div>
                )}
              </div>

              <form onSubmit={askFollowup} className="rounded-lg border border-gray-200 bg-white shadow-sm p-5">
                <h2 className="text-base font-semibold text-gray-900 mb-3">Ask About This Analysis</h2>
                <div className="flex flex-col sm:flex-row gap-2">
                  <input
                    value={followupQuestion}
                    onChange={(event) => setFollowupQuestion(event.target.value)}
                    className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm"
                    placeholder="Why is this risky? What should I test?"
                  />
                  <button
                    type="submit"
                    disabled={isAsking || !followupQuestion.trim()}
                    className="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white disabled:bg-gray-300"
                  >
                    {isAsking ? 'Asking...' : 'Ask'}
                  </button>
                </div>
                {followupAnswer && (
                  <div className="mt-4 border-t border-gray-100 pt-3">
                    <Markdown content={followupAnswer} />
                  </div>
                )}
              </form>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-gray-900">{value}</div>
    </div>
  );
}
