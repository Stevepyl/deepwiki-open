import Link from "next/link";
import { FiTrash2 } from "react-icons/fi";
import { IconButton } from "../shell/IconButton";

interface DirCardProps {
  href: string;
  path: string;
  description: string;
  conversationCount: number;
  updatedAt: string;
  view: "grid" | "list";
  isDeleting?: boolean;
  onDelete: () => void;
}

export function DirCard({
  href,
  path,
  description,
  conversationCount,
  updatedAt,
  view,
  isDeleting = false,
  onDelete,
}: DirCardProps) {
  return (
    <article
      className={`group relative rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)] transition-colors duration-150 hover:border-[var(--hairline-strong)] hover:bg-[var(--paper-hover)] ${
        view === "list" ? "max-w-[760px]" : ""
      }`}
    >
      <Link className="flex flex-col gap-[7px] px-5 py-[18px] pr-14" href={href}>
        <span className="font-[var(--font-mono)] text-[13px] font-medium tracking-normal text-[var(--ink-primary)]">
          {path}
        </span>
        <span className="font-[var(--font-serif)] text-[13.5px] italic leading-[1.45] text-[var(--ink-secondary)]">
          {description}
        </span>
        <span className="flex items-center gap-2.5 font-[var(--font-sans)] text-[10.5px] tracking-normal text-[var(--ink-muted)]">
          <span>{conversationCount} convs</span>
          <span className="h-0.5 w-0.5 rounded-full bg-[var(--ink-faint)]" />
          <span>{updatedAt}</span>
        </span>
      </Link>
      <IconButton
        aria-label={`Delete ${path}`}
        className="absolute right-3 top-3 opacity-0 hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] group-hover:opacity-100 focus-visible:opacity-100"
        disabled={isDeleting}
        onClick={onDelete}
      >
        <FiTrash2 aria-hidden="true" />
      </IconButton>
    </article>
  );
}
