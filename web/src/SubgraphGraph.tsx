import { useMemo } from "react";

interface GraphNode {
  id: string;
  name: string;
  group?: number;
}

interface GraphLink {
  source: string;
  target: string;
  relation: string;
}

interface SubgraphGraphProps {
  subgraph: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  activeEntity: string;
  isSpeaking: boolean;
}

export default function SubgraphGraph({
  subgraph,
  activeEntity,
  isSpeaking,
}: SubgraphGraphProps) {
  const { nodes, links } = subgraph;

  // Compute circular layout coordinates for nodes
  const layout = useMemo(() => {
    const width = 360;
    const height = 260;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.35;

    const coords: Record<string, { x: number; y: number }> = {};
    const n = nodes.length;

    if (n === 1) {
      coords[nodes[0].id] = { x: cx, y: cy };
    } else {
      nodes.forEach((node, idx) => {
        // Place active entity at center if multiple nodes, or evenly on a ring
        if (
          node.name.toLowerCase() === activeEntity.toLowerCase() &&
          n > 2
        ) {
          coords[node.id] = { x: cx, y: cy };
        } else {
          const angle = (2 * Math.PI * idx) / n - Math.PI / 2;
          coords[node.id] = {
            x: cx + radius * Math.cos(angle),
            y: cy + radius * Math.sin(angle),
          };
        }
      });
    }

    return { width, height, coords };
  }, [nodes, activeEntity]);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        minHeight: "260px",
        background: "rgba(244, 243, 236, 0.4)",
        borderRadius: "12px",
        border: "1px solid var(--border)",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          zIndex: 10,
          background: "rgba(255, 255, 255, 0.92)",
          padding: "4px 10px",
          borderRadius: "6px",
          fontSize: "12px",
          fontWeight: 600,
          color: "#16171d",
          boxShadow: "0 1px 4px rgba(0,0,0,0.1)",
        }}
      >
        🕸️ Neo4j Knowledge Subgraph
      </div>

      <svg
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        style={{ width: "100%", height: "100%", flex: 1 }}
      >
        <defs>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="22"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#aa3bff" opacity={0.7} />
          </marker>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Links */}
        {links.map((link, i) => {
          const s = layout.coords[link.source];
          const t = layout.coords[link.target];
          if (!s || !t) return null;

          const midX = (s.x + t.x) / 2;
          const midY = (s.y + t.y) / 2;

          return (
            <g key={`link-${i}`}>
              <line
                x1={s.x}
                y1={s.y}
                x2={t.x}
                y2={t.y}
                stroke="#aa3bff"
                strokeWidth={2}
                strokeDasharray="4,4"
                opacity={0.65}
                markerEnd="url(#arrow)"
              />
              <rect
                x={midX - 28}
                y={midY - 9}
                width={56}
                height={16}
                rx={4}
                fill="#ffffff"
                stroke="#e5e4e7"
                strokeWidth={1}
                opacity={0.95}
              />
              <text
                x={midX}
                y={midY + 3}
                textAnchor="middle"
                fontSize="9"
                fontWeight="600"
                fill="#6b6375"
              >
                {link.relation}
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          const pt = layout.coords[node.id];
          if (!pt) return null;

          const isActive =
            node.name.toLowerCase().includes(activeEntity.toLowerCase()) ||
            activeEntity.toLowerCase().includes(node.name.toLowerCase());

          return (
            <g key={node.id} transform={`translate(${pt.x}, ${pt.y})`}>
              {/* Pulsing ring when bot is speaking */}
              {isActive && isSpeaking && (
                <circle
                  r={22}
                  fill="none"
                  stroke="#aa3bff"
                  strokeWidth={2}
                  opacity={0.7}
                  style={{
                    animation: "pulseRing 1.5s ease-out infinite",
                  }}
                />
              )}

              <circle
                r={16}
                fill={isActive ? "#aa3bff" : "#4169E1"}
                stroke="#ffffff"
                strokeWidth={3}
                filter={isActive ? "url(#glow)" : undefined}
                style={{ cursor: "pointer", transition: "all 0.3s ease" }}
              />

              <text
                y={28}
                textAnchor="middle"
                fontSize="11"
                fontWeight={isActive ? "700" : "500"}
                fill="var(--text-h)"
                style={{ textShadow: "0 1px 3px rgba(255,255,255,0.9)" }}
              >
                {node.name}
              </text>
            </g>
          );
        })}
      </svg>

      <style>{`
        @keyframes pulseRing {
          0% { r: 16px; opacity: 0.9; }
          100% { r: 32px; opacity: 0; }
        }
      `}</style>
    </div>
  );
}
