"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ExampleCard } from "../components/welcome/ExampleCard";
import { ScratchInput } from "../components/welcome/ScratchInput";
import { Wordmark } from "../components/shell/Wordmark";
import { extractUrlDomain, extractUrlPath } from "../utils/urlDecoder";

const examples = [
  {
    path: "anthropics/claude-code",
    description: "An agentic CLI that lives in your terminal - explore its hooks and MCP system.",
    href: "/anthropics/claude-code?status=generating",
  },
  {
    path: "vercel/next.js",
    description: "The React framework - understand the App Router, RSC, and build pipeline.",
    href: "/vercel/next.js?status=generating",
  },
  {
    path: "facebook/react",
    description: "Read the reconciler, hooks, and concurrent rendering as a guided wiki.",
    href: "/facebook/react",
  },
  {
    path: "openai/whisper",
    description: "Trace the audio encoder, decoder, and language detection end to end.",
    href: "/openai/whisper",
  },
];

function parseGithubRepo(input: string) {
  const domain = extractUrlDomain(input.trim());
  const path = extractUrlPath(input.trim());
  if (!domain || !path) {
    return null;
  }
  const hostname = new URL(domain).hostname.toLowerCase();
  const [owner, rawRepo] = path.split("/");
  const repo = rawRepo?.replace(/\.git$/, "");
  return hostname.includes("github.com") && owner && repo ? { owner, repo } : null;
}

export default function HomePage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [value, setValue] = useState("");
  const [hint, setHint] = useState("");

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const submit = () => {
    const repo = parseGithubRepo(value);
    if (!repo) {
      setHint("Paste a GitHub repository URL to begin generation.");
      return;
    }
    setHint("");
    router.push(`/${repo.owner}/${repo.repo}?status=generating`);
  };

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center px-8 py-16">
      <div className="absolute left-8 top-7 flex items-center gap-2.5 font-sans text-[11px] font-medium uppercase tracking-[0.18em] text-[var(--ink-muted)] before:text-[13px] before:text-[var(--accent)] before:content-['◌']">
        <span>OpsWiki · v0 prototype</span>
        <Link
          className="tracking-[0.14em] transition-colors duration-150 hover:text-[var(--ink-primary)]"
          href="/projects"
        >
          All projects →
        </Link>
      </div>

      <header className="mb-16 flex flex-col items-center gap-[18px]">
        <Wordmark size="hero" />
        <p className="font-serif text-xl tracking-normal text-[var(--ink-secondary)]">
          Understand repos with AI
        </p>
      </header>

      <ScratchInput hint={hint} onChange={setValue} onSubmit={submit} ref={inputRef} value={value} />

      <section className="mt-12 w-full max-w-[640px]" aria-label="Example repositories">
        <div className="mb-[18px] flex items-center gap-2.5 font-sans text-[11px] font-medium uppercase tracking-[0.14em] text-[var(--ink-muted)] before:h-px before:w-4 before:bg-[var(--hairline-strong)] before:content-['']">
          <span>Or start from</span>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {examples.map((example) => (
            <ExampleCard description={example.description} href={example.href} key={example.path} path={example.path} />
          ))}
        </div>
      </section>

      <footer className="absolute bottom-7 left-1/2 -translate-x-1/2 font-sans text-[11px] tracking-[0.04em] text-[var(--ink-faint)]">
        Press <kbd className="rounded-[3px] border border-[var(--hairline)] px-1.5 py-px font-mono text-[10px]">↵</kbd> to begin · or{" "}
        <kbd className="rounded-[3px] border border-[var(--hairline)] px-1.5 py-px font-mono text-[10px]">⌘K</kbd> to focus search
      </footer>
    </main>
  );
}
