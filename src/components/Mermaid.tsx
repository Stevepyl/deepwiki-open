"use client";

import React, { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: true,
  theme: "base",
  securityLevel: "loose",
  suppressErrorRendering: true,
  logLevel: "error",
  maxTextSize: 100000,
  htmlLabels: true,
  flowchart: {
    htmlLabels: true,
    curve: "basis",
    nodeSpacing: 60,
    rankSpacing: 60,
    padding: 20,
  },
  themeVariables: {
    background: "#FBF8F2",
    primaryColor: "#FBF8F2",
    primaryBorderColor: "rgba(26, 24, 21, 0.14)",
    primaryTextColor: "#1A1815",
    lineColor: "#C04A1A",
    secondaryColor: "#EFEAE0",
    tertiaryColor: "#F5F1EA",
    fontFamily: "var(--font-sans)",
  },
  themeCSS: `
    .node rect, .node circle, .node ellipse, .node polygon, .node path {
      fill: #FBF8F2;
      stroke: rgba(26, 24, 21, 0.14);
      stroke-width: 1px;
    }
    .edgePath .path, .flowchart-link, .messageLine0, .messageLine1 {
      stroke: #C04A1A;
      stroke-width: 1.5px;
    }
    .edgeLabel, .label, .nodeLabel, text {
      color: #1A1815;
      fill: #1A1815 !important;
    }
    .cluster rect, .actor {
      fill: #F5F1EA;
      stroke: rgba(26, 24, 21, 0.14);
    }
    .clickable {
      transition: all 0.2s ease;
    }
    .clickable:hover {
      cursor: pointer;
      filter: brightness(0.98);
    }
  `,
  fontFamily: "var(--font-sans)",
  fontSize: 12,
});

interface MermaidProps {
  chart: string;
  className?: string;
  zoomingEnabled?: boolean;
}

function FullScreenModal({
  isOpen,
  onClose,
  children,
}: {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center bg-[rgba(26,24,21,0.28)] p-6">
      <div
        ref={modalRef}
        className="max-h-[90vh] w-full max-w-5xl overflow-auto rounded-[var(--radius-md)] border border-[var(--hairline-strong)] bg-[var(--paper-panel)] p-6 shadow-2xl"
      >
        <div className="mb-4 flex justify-end">
          <button
            className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--ink-muted)] transition-all duration-[120ms] hover:bg-[var(--paper-panel)] hover:text-[var(--ink-primary)]"
            type="button"
            onClick={onClose}
            aria-label="Close diagram"
          >
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

const Mermaid: React.FC<MermaidProps> = ({ chart, className = "", zoomingEnabled = false }) => {
  const [svg, setSvg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(`mermaid-${Math.random().toString(36).slice(2, 9)}`);

  useEffect(() => {
    if (!svg || !zoomingEnabled || !containerRef.current) {
      return;
    }

    const initializePanZoom = async () => {
      const svgElement = containerRef.current?.querySelector("svg");
      if (!svgElement) {
        return;
      }
      svgElement.style.maxWidth = "none";
      svgElement.style.width = "100%";
      svgElement.style.height = "100%";

      const svgPanZoom = (await import("svg-pan-zoom")).default;
      svgPanZoom(svgElement, {
        zoomEnabled: true,
        controlIconsEnabled: true,
        fit: true,
        center: true,
        minZoom: 0.1,
        maxZoom: 10,
        zoomScaleSensitivity: 0.3,
      });
    };

    void initializePanZoom();
  }, [svg, zoomingEnabled]);

  useEffect(() => {
    if (!chart) {
      return;
    }
    let mounted = true;

    const renderChart = async () => {
      try {
        setError(null);
        setSvg("");
        const { svg: renderedSvg } = await mermaid.render(idRef.current, chart);
        if (mounted) {
          setSvg(renderedSvg);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (mounted) {
          setError(message);
        }
      }
    };

    void renderChart();
    return () => {
      mounted = false;
    };
  }, [chart]);

  if (error) {
    return (
      <div className={`rounded-[var(--radius-sm)] border border-[var(--accent-line)] bg-[var(--accent-soft)] p-4 ${className}`}>
        <div className="mb-2 text-xs font-medium text-[var(--accent)]">Syntax error in diagram</div>
        <pre className="overflow-auto rounded-[var(--radius-sm)] bg-[var(--paper-panel)] p-3 text-xs text-[var(--ink-secondary)]">
          {chart}
        </pre>
      </div>
    );
  }

  if (!svg) {
    return (
      <div className={`flex items-center justify-center p-4 text-xs text-[var(--ink-muted)] ${className}`}>
        Rendering diagram...
      </div>
    );
  }

  return (
    <>
      <div ref={containerRef} className={`w-full max-w-full ${zoomingEnabled ? "h-[600px] p-4" : ""}`}>
        <div
          className={`flex justify-center overflow-auto rounded-[var(--radius-sm)] text-center ${className} ${
            zoomingEnabled ? "h-full border border-[var(--hairline)] bg-[var(--paper-panel)]" : "cursor-zoom-in"
          }`}
          dangerouslySetInnerHTML={{ __html: svg }}
          onClick={zoomingEnabled ? undefined : () => setIsFullscreen(true)}
          title={zoomingEnabled ? undefined : "Click to view fullscreen"}
        />
      </div>

      {!zoomingEnabled && (
        <FullScreenModal isOpen={isFullscreen} onClose={() => setIsFullscreen(false)}>
          <div dangerouslySetInnerHTML={{ __html: svg }} />
        </FullScreenModal>
      )}
    </>
  );
};

export default Mermaid;
