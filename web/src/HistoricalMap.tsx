import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

interface HistoricalMapProps {
  coordinates: [number, number];
  entityName: string;
  dynastyName: string;
  eraYears: [number, number];
}

// Fallback parchment-pattern tile for air-gapped / offline deployments
const OFFLINE_TILE_SVG =
  "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'><rect width='256' height='256' fill='%23f5f2eb'/><path d='M0 0h256v256H0z' fill='none' stroke='%23e6decb' stroke-width='1'/></svg>";

// Approximate Indian subcontinent geographic landmass for offline vector rendering
const SUBCONTINENT_OUTLINE: [number, number][] = [
  [37.0, 74.0], [35.5, 77.0], [32.0, 79.0], [30.0, 81.0], [28.0, 89.0],
  [27.5, 97.0], [24.0, 94.0], [22.0, 92.0], [21.5, 87.0], [16.0, 82.0],
  [13.0, 80.3], [10.0, 79.8], [8.0, 77.5], [10.0, 76.0], [13.0, 74.5],
  [16.0, 73.3], [19.0, 72.8], [21.0, 72.5], [23.0, 68.5], [25.0, 67.5],
  [28.0, 66.5], [31.5, 70.0], [35.0, 71.5], [37.0, 74.0],
];

// Approximate historical empire boundaries (GeoJSON polygons)
const EMPIRE_BOUNDARIES: Record<string, [number, number][]> = {
  "Maurya Empire": [
    [34.0, 71.0], [32.0, 76.0], [28.0, 84.0], [26.0, 90.0],
    [22.0, 89.0], [15.0, 76.0], [13.0, 78.0], [18.0, 73.0],
    [24.0, 68.0], [30.0, 67.0], [34.0, 71.0],
  ],
  "Mughal Empire": [
    [34.5, 69.0], [32.0, 77.0], [27.0, 85.0], [24.0, 91.0],
    [22.0, 88.0], [20.0, 82.0], [18.0, 74.0], [24.0, 69.0],
    [30.0, 66.0], [34.5, 69.0],
  ],
  "Chola Dynasty": [
    [14.0, 80.0], [13.0, 80.5], [10.0, 79.8], [8.0, 77.5],
    [10.0, 76.5], [12.0, 77.5], [14.0, 80.0],
  ],
  "Indus Valley Civilisation": [
    [32.0, 71.0], [30.0, 74.0], [27.0, 71.0], [23.0, 69.0],
    [24.0, 66.5], [28.0, 66.0], [32.0, 71.0],
  ],
  "Maratha Empire": [
    [24.0, 77.0], [22.0, 83.0], [18.0, 81.0], [15.0, 74.0],
    [17.0, 73.0], [20.0, 73.0], [24.0, 77.0],
  ],
};

function formatEra(years: [number, number]): string {
  const [start, end] = years;
  const startStr = start < 0 ? `${Math.abs(start)} BCE` : `${start} CE`;
  const endStr = end < 0 ? `${Math.abs(end)} BCE` : `${end} CE`;
  return start === end ? startStr : `${startStr} – ${endStr}`;
}

export default function HistoricalMap({
  coordinates,
  entityName,
  dynastyName,
  eraYears,
}: HistoricalMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);
  const polygonRef = useRef<L.Polygon | null>(null);
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: coordinates,
        zoom: 5,
        zoomControl: false,
      });

      // Air-gap friendly tile layer with local SVG fallback
      const tiles = L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 18,
        errorTileUrl: OFFLINE_TILE_SVG,
      });

      tiles.on("tileerror", () => {
        setIsOffline(true);
      });

      tiles.addTo(map);

      // Base subcontinent landmass vector for air-gapped / offline rendering
      L.polygon(SUBCONTINENT_OUTLINE, {
        color: "#c8beaf",
        weight: 1.5,
        fillColor: "#e6decb",
        fillOpacity: 0.25,
        interactive: false,
      }).addTo(map);

      mapRef.current = map;
    }

    const map = mapRef.current;
    map.flyTo(coordinates, 5, { duration: 1.2 });

    // Update Pulse Marker
    if (markerRef.current) {
      markerRef.current.remove();
    }
    const marker = L.circleMarker(coordinates, {
      radius: 12,
      fillColor: "#aa3bff",
      color: "#ffffff",
      weight: 3,
      opacity: 1,
      fillOpacity: 0.85,
    }).addTo(map);
    marker.bindPopup(`<b>${entityName}</b><br/>${dynastyName} (${formatEra(eraYears)})`).openPopup();
    markerRef.current = marker;

    // Update Territory Boundary
    if (polygonRef.current) {
      polygonRef.current.remove();
    }
    const boundary = EMPIRE_BOUNDARIES[dynastyName];
    if (boundary) {
      const polygon = L.polygon(boundary, {
        color: "#aa3bff",
        weight: 2,
        fillColor: "#c084fc",
        fillOpacity: 0.18,
        dashArray: "6, 6",
      }).addTo(map);
      polygonRef.current = polygon;
    }
  }, [coordinates, entityName, dynastyName, eraYears]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", borderRadius: "12px", overflow: "hidden" }}>
      <div
        ref={mapContainerRef}
        style={{ width: "100%", height: "100%", minHeight: "260px" }}
      />
      <div
        style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          zIndex: 1000,
          background: "rgba(255, 255, 255, 0.92)",
          padding: "6px 12px",
          borderRadius: "8px",
          fontSize: "12px",
          boxShadow: "0 2px 6px rgba(0,0,0,0.15)",
          color: "#16171d",
          backdropFilter: "blur(4px)",
        }}
      >
        <strong>🗺️ {dynastyName}</strong>
        {isOffline && (
          <span
            style={{
              marginLeft: "6px",
              padding: "2px 6px",
              background: "#e2e8f0",
              color: "#475569",
              borderRadius: "4px",
              fontSize: "10px",
              fontWeight: 600,
            }}
          >
            Air-Gapped / Offline
          </span>
        )}
        <div style={{ fontSize: "11px", color: "#6b6375" }}>
          {formatEra(eraYears)} • [{coordinates[0].toFixed(2)}°, {coordinates[1].toFixed(2)}°]
        </div>
      </div>
    </div>
  );
}
