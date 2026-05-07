import Link from "next/link";
import { FiHome } from "react-icons/fi";

interface AlphaRailProps {
  letters: string[];
  activeLetter: string;
}

export function AlphaRail({ letters, activeLetter }: AlphaRailProps) {
  return (
    <nav
      aria-label="Alphabetical index"
      className="sticky top-0 flex h-screen w-14 shrink-0 flex-col items-center gap-0.5 overflow-y-auto border-r border-[var(--hairline)] bg-[var(--paper-panel)] px-0 py-6"
    >
      <Link
        aria-label="OpsWiki home"
        className="mb-3 flex h-6 w-8 items-center justify-center rounded text-[var(--ink-faint)] transition-all duration-150 hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
        href="/"
        title="Home"
      >
        <FiHome aria-hidden="true" className="h-3.5 w-3.5" />
      </Link>
      {letters.map((letter) => (
        <div className="flex flex-col items-center gap-0.5" key={letter}>
          <span className="my-1 h-[3px] w-[3px] rounded-full bg-[var(--hairline-strong)]" />
          <a
            className={`flex h-6 w-8 items-center justify-center rounded font-mono text-[11px] font-medium tracking-[0.06em] transition-all duration-150 ${
              activeLetter === letter
                ? "bg-[var(--accent-soft)] text-[var(--accent)]"
                : "text-[var(--ink-faint)] hover:bg-[var(--paper-hover)] hover:text-[var(--ink-primary)]"
            }`}
            href={`#group-${letter.toLowerCase()}`}
          >
            {letter}
          </a>
        </div>
      ))}
    </nav>
  );
}
