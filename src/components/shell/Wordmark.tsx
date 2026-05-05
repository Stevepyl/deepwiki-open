interface WordmarkProps {
  size: "hero" | "sidebar";
}

export function Wordmark({ size }: WordmarkProps) {
  const className = `select-none font-[var(--font-serif)] font-medium leading-none tracking-normal text-[var(--ink-primary)] ${
    size === "hero" ? "text-[104px]" : "text-[22px]"
  }`;
  const content = (
    <>
      <em className="font-medium italic">Ops</em>Wiki
    </>
  );

  if (size === "hero") {
    return <h1 className={className}>{content}</h1>;
  }
  return <span className={className}>{content}</span>;
}
