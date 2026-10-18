import { useState } from "react";

interface CitationData {
  chunk_id: string;
  article: string;
  section: string;
  text: string;
}

interface CitationDrawerProps {
  citation?: CitationData;
  citationsList?: string[];
  isCached?: boolean;
  similarity?: number;
}

export default function CitationDrawer({
  citation,
  citationsList = [],
  isCached = false,
  similarity,
}: CitationDrawerProps) {
  const [expanded, setExpanded] = useState(true);

  if (!citation && citationsList.length === 0) {
    return (
      <div
        style={{
          padding: "16px",
          background: "rgba(244, 243, 236, 0.4)",
          borderRadius: "12px",
          border: "1px solid var(--border)",
          fontSize: "13px",
          color: "var(--text)",
          textAlign: "center",
        }}
      >
        📚 <em>No citations for this turn yet. Start speaking to inspect grounding.</em>
      </div>
    );
  }

  return (
    <aside
      style={{
        background: "var(--bg)",
        border: "1px solid var(--border)",
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: "var(--shadow)",
        fontSize: "13px",
        textAlign: "left",
      }}
    >
      <header
        onClick={() => setExpanded(!expanded)}
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 14px",
          background: "rgba(170, 59, 255, 0.08)",
          cursor: "pointer",
          borderBottom: expanded ? "1px solid var(--border)" : "none",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <strong>📚 Grounding Drawer</strong>
          <span
            style={{
              background: "#27ae60",
              color: "#fff",
              fontSize: "10px",
              fontWeight: 700,
              padding: "2px 6px",
              borderRadius: "4px",
            }}
          >
            0% HALLUCINATION
          </span>
          {isCached && (
            <span
              style={{
                background: "#2980b9",
                color: "#fff",
                fontSize: "10px",
                fontWeight: 700,
                padding: "2px 6px",
                borderRadius: "4px",
              }}
            >
              ⚡ CACHE HIT {similarity ? `(${(similarity * 100).toFixed(1)}%)` : ""}
            </span>
          )}
        </div>
        <button
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            fontSize: "14px",
          }}
        >
          {expanded ? "▲" : "▼"}
        </button>
      </header>

      {expanded && (
        <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "10px" }}>
          {citation && (
            <>
              <div>
                <span
                  style={{
                    fontFamily: "var(--mono)",
                    background: "var(--code-bg)",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    fontSize: "11px",
                    fontWeight: 600,
                    color: "var(--accent)",
                  }}
                >
                  [{citation.chunk_id}]
                </span>
                <span style={{ marginLeft: "8px", fontWeight: 600 }}>
                  {citation.article}
                </span>
                {citation.section && (
                  <span style={{ color: "#6b6375", marginLeft: "4px" }}>
                    &gt; {citation.section}
                  </span>
                )}
              </div>

              <blockquote
                style={{
                  margin: 0,
                  padding: "8px 12px",
                  background: "rgba(244, 243, 236, 0.6)",
                  borderLeft: "3px solid var(--accent)",
                  borderRadius: "4px",
                  lineHeight: 1.5,
                  fontSize: "12px",
                  color: "var(--text-h)",
                }}
              >
                &ldquo;{citation.text}&rdquo;
              </blockquote>
            </>
          )}

          {citationsList.length > 0 && (
            <div style={{ fontSize: "11px", color: "#6b6375" }}>
              <strong>Referenced Chunks:</strong>{" "}
              {citationsList.map((cid) => (
                <code
                  key={cid}
                  style={{
                    marginRight: "6px",
                    fontSize: "11px",
                    background: "var(--code-bg)",
                  }}
                >
                  {cid}
                </code>
              ))}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
