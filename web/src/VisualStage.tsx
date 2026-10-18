import HistoricalMap from "./HistoricalMap";
import SubgraphGraph from "./SubgraphGraph";
import CitationDrawer from "./CitationDrawer";

export interface VisualStageData {
  event?: string;
  entity: string;
  dynasty: string;
  era_years: [number, number];
  subgraph: {
    nodes: Array<{ id: string; name: string; group?: number }>;
    links: Array<{ source: string; target: string; relation: string }>;
  };
  geo_coordinates: [number, number];
  citation: {
    chunk_id: string;
    article: string;
    section: string;
    text: string;
  };
  cached?: boolean;
  similarity?: number;
}

interface VisualStageProps {
  data?: VisualStageData | null;
  citationsList?: string[];
  isSpeaking: boolean;
}

export default function VisualStage({
  data,
  citationsList = [],
  isSpeaking,
}: VisualStageProps) {
  if (!data) {
    return (
      <section
        style={{
          marginTop: "0.5rem",
          padding: "24px",
          background: "rgba(244, 243, 236, 0.3)",
          borderRadius: "12px",
          border: "1px dashed var(--border)",
          textAlign: "center",
          color: "var(--text)",
          fontSize: "14px",
        }}
      >
        <p><strong>🎭 Multimodal Visual Stage Ready</strong></p>
        <p style={{ marginTop: "4px", fontSize: "12px" }}>
          Ask a question (e.g. <em>&ldquo;Ashoka ne Kalinga ke baad kya kiya?&rdquo;</em>) to activate dynamic Neo4j subgraph pulses and historical territory maps.
        </p>
      </section>
    );
  }

  return (
    <section
      style={{
        marginTop: "0.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "14px",
        textAlign: "left",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid var(--border)",
          paddingBottom: "8px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "18px" }}>👁️</span>
          <h2 style={{ fontSize: "15px", margin: 0, fontWeight: 700 }}>
            Visual Telemetry
          </h2>
          <span
            style={{
              fontSize: "11px",
              padding: "2px 8px",
              background: "rgba(170, 59, 255, 0.15)",
              color: "var(--accent)",
              borderRadius: "12px",
              fontWeight: 600,
            }}
          >
            Live Multimodal Sync
          </span>
        </div>

        {data.cached && (
          <span
            style={{
              fontSize: "11px",
              padding: "2px 8px",
              background: "#2980b9",
              color: "#ffffff",
              borderRadius: "12px",
              fontWeight: 600,
            }}
          >
            ⚡ Sub-120ms Cache Hit
          </span>
        )}
      </div>

      {/* Dual Panel: Map (Left) + Neo4j Subgraph (Right) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "14px",
          minHeight: "280px",
        }}
      >
        <div style={{ height: "280px" }}>
          <HistoricalMap
            coordinates={data.geo_coordinates}
            entityName={data.entity}
            dynastyName={data.dynasty}
            eraYears={data.era_years}
          />
        </div>

        <div style={{ height: "280px" }}>
          <SubgraphGraph
            subgraph={data.subgraph}
            activeEntity={data.entity}
            isSpeaking={isSpeaking}
          />
        </div>
      </div>

      {/* Grounding Drawer (Bottom) */}
      <CitationDrawer
        citation={data.citation}
        citationsList={citationsList}
        isCached={data.cached}
        similarity={data.similarity}
      />
    </section>
  );
}
