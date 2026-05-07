import Markdown from "../Markdown";
import type { WikiPageWithContent } from "../../utils/wiki";
import { stripLeadingTitle } from "../../utils/wiki";

interface WikiArticleProps {
  page: WikiPageWithContent;
  sectionTitle: string;
}

function firstSentence(content: string) {
  const plain = content
    .replace(/^#+\s+/gm, "")
    .replace(/```[\s\S]*?```/g, "")
    .trim();
  return plain.match(/[^.!?\n]+[.!?]/)?.[0]?.trim() ?? plain.split("\n")[0] ?? "";
}

function pathParts(filePath: string | undefined) {
  if (!filePath) return { module: "Repository", entry: "Source map" };
  const parts = filePath.split("/");
  return {
    module: parts.length > 1 ? parts.slice(0, -1).join("/") : "Repository",
    entry: parts.at(-1) ?? filePath,
  };
}

export function WikiArticle({ page, sectionTitle }: WikiArticleProps) {
  const body = stripLeadingTitle(page.content, page.title);
  const { module, entry } = pathParts(page.filePaths[0]);
  const lede = firstSentence(body);

  return (
    <article className="min-w-0 pb-6 text-[var(--ink-primary)]">
      <div className="mb-3 flex items-center gap-2.5 font-sans text-[11px] font-medium uppercase tracking-normal text-[var(--accent)] before:h-px before:w-5 before:bg-[var(--accent-line)] before:content-['']">
        <span>{sectionTitle}</span>
      </div>
      <h1 className="m-0 mb-5 font-serif text-[40px] font-semibold leading-[1.15] tracking-normal text-[var(--ink-primary)]">
        {page.title}
      </h1>
      {lede && (
        <p className="mb-10 max-w-[640px] font-serif text-[19px] italic leading-[1.55] text-[var(--ink-secondary)]">
          {lede}
        </p>
      )}
      <div className="mb-10 flex flex-wrap gap-x-5 gap-y-2 border-y border-[var(--hairline)] py-3.5 font-sans text-[11.5px] tracking-normal text-[var(--ink-muted)]">
        <span>
          <strong className="mr-1.5 font-medium text-[var(--ink-secondary)]">Module</strong>
          {module}
        </span>
        <span>
          <strong className="mr-1.5 font-medium text-[var(--ink-secondary)]">Entry</strong>
          {entry}
        </span>
        <span>
          <strong className="mr-1.5 font-medium text-[var(--ink-secondary)]">Updated</strong>
          Cached wiki
        </span>
        <span>
          <strong className="mr-1.5 font-medium text-[var(--ink-secondary)]">Confidence</strong>
          {page.importance} · {page.filePaths.length} sources
        </span>
      </div>
      <Markdown content={body} />
    </article>
  );
}
