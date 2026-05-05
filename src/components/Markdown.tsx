import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import Mermaid from "./Mermaid";

interface MarkdownProps {
  content: string;
}

const paperSyntaxTheme = {
  'code[class*="language-"]': {
    background: "var(--paper-panel)",
    color: "var(--ink-primary)",
    fontFamily: "var(--font-mono)",
    fontSize: "13px",
    lineHeight: 1.6,
    textShadow: "none",
  },
  'pre[class*="language-"]': {
    background: "var(--paper-panel)",
    color: "var(--ink-primary)",
    fontFamily: "var(--font-mono)",
    fontSize: "13px",
    lineHeight: 1.6,
    textShadow: "none",
  },
  keyword: { color: "#8F3E7C" },
  string: { color: "var(--accent)" },
  comment: { color: "var(--ink-muted)", fontStyle: "italic" },
  function: { color: "#2E5C8A" },
  operator: { color: "var(--ink-secondary)" },
  punctuation: { color: "var(--ink-secondary)" },
};

const Markdown: React.FC<MarkdownProps> = ({ content }) => {
  const components: React.ComponentProps<typeof ReactMarkdown>["components"] = {
    p({ children, ...props }) {
      return <p {...props} className="my-4 font-[var(--font-serif)] text-base leading-[1.75]">{children}</p>;
    },
    h1({ children, ...props }) {
      return <h1 {...props} className="mb-4 mt-8 font-[var(--font-serif)] text-[28px] leading-tight">{children}</h1>;
    },
    h2({ children, ...props }) {
      return <h2 {...props} className="mb-3 mt-7 font-[var(--font-serif)] text-[22px] leading-tight">{children}</h2>;
    },
    h3({ children, ...props }) {
      return <h3 {...props} className="mb-2 mt-6 font-[var(--font-serif)] text-lg leading-tight">{children}</h3>;
    },
    h4({ children, ...props }) {
      return <h4 {...props} className="mb-2 mt-5 font-[var(--font-sans)] text-[13px] font-semibold">{children}</h4>;
    },
    a({ children, href, ...props }) {
      return (
        <a
          {...props}
          className="border-b border-[var(--accent-line)] text-[var(--accent)]"
          href={href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {children}
        </a>
      );
    },
    ul({ children, ...props }) {
      return <ul {...props} className="my-4 list-disc pl-6">{children}</ul>;
    },
    ol({ children, ...props }) {
      return <ol {...props} className="my-4 list-decimal pl-6">{children}</ol>;
    },
    li({ children, ...props }) {
      return <li {...props} className="font-[var(--font-serif)] text-base leading-[1.75]">{children}</li>;
    },
    blockquote({ children, ...props }) {
      return (
        <blockquote
          {...props}
          className="my-6 border-l-2 border-[var(--accent)] px-5 py-1 font-[var(--font-serif)] text-[17px] italic leading-[1.6] text-[var(--ink-secondary)]"
        >
          {children}
        </blockquote>
      );
    },
    table({ children, ...props }) {
      return (
        <div className="my-6 overflow-x-auto">
          <table
            {...props}
            className="w-full border-collapse border border-[var(--hairline)] bg-[var(--paper-panel)] text-[13px]"
          >
            {children}
          </table>
        </div>
      );
    },
    th({ children, ...props }) {
      return (
        <th {...props} className="border-t border-[var(--hairline)] px-3 py-2.5 text-left font-semibold text-[var(--ink-secondary)]">
          {children}
        </th>
      );
    },
    td({ children, ...props }) {
      return (
        <td {...props} className="border-t border-[var(--hairline)] px-3 py-2.5 text-left">
          {children}
        </td>
      );
    },
    pre({ children }) {
      return <>{children}</>;
    },
    code(props) {
      const { className, children } = props;
      const match = /language-(\w+)/.exec(className ?? "");
      const codeContent = children ? String(children).replace(/\n$/, "") : "";

      if (match?.[1] === "mermaid") {
        return (
          <div className="my-5 overflow-hidden rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)]">
            <Mermaid chart={codeContent} className="w-full max-w-full" zoomingEnabled />
          </div>
        );
      }

      if (match) {
        return (
          <div className="my-5 overflow-hidden rounded-[var(--radius-sm)] border border-[var(--hairline)] bg-[var(--paper-panel)]">
            <div className="flex items-center justify-between border-b border-[var(--hairline)] px-4 py-2 font-[var(--font-mono)] text-[11px] text-[var(--ink-muted)]">
              <span>{match[1]}</span>
              <button
                type="button"
                onClick={() => navigator.clipboard.writeText(codeContent)}
                className="text-[var(--ink-secondary)] hover:text-[var(--accent)]"
              >
                Copy
              </button>
            </div>
            <SyntaxHighlighter
              language={match[1]}
              style={paperSyntaxTheme}
              customStyle={{ margin: 0, borderRadius: 0, padding: "18px 20px" }}
              showLineNumbers
              wrapLines
              wrapLongLines
            >
              {codeContent}
            </SyntaxHighlighter>
          </div>
        );
      }

      return (
        <code
          className={`rounded border border-[var(--hairline)] bg-[var(--paper-panel)] px-1.5 py-px font-[var(--font-mono)] text-[13.5px] text-[var(--accent)] ${className ?? ""}`}
        >
          {children}
        </code>
      );
    },
  };

  return (
    <div className="text-[var(--ink-primary)]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default Markdown;
