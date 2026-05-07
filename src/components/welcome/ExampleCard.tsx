import Link from "next/link";

interface ExampleCardProps {
  path: string;
  description: string;
  href: string;
}

export function ExampleCard({ path, description, href }: ExampleCardProps) {
  return (
    <Link
      className="flex flex-col gap-1.5 rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)] px-[18px] py-4 font-sans transition-colors duration-150 hover:border-[var(--hairline-strong)] hover:bg-[var(--paper-hover)]"
      href={href}
      prefetch
    >
      <span className="text-xs tracking-normal text-[var(--ink-secondary)]">{path}</span>
      <span className="text-sm italic leading-[1.4] text-[var(--ink-muted)]">
        {description}
      </span>
    </Link>
  );
}
