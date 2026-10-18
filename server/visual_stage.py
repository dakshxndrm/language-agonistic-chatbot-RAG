"""Visual Stage WebRTC Data Channel telemetry generator (Phase 5).

Emits structured JSON events over LiveKit Data Channels synchronously with
audio playback so the React frontend can:
1. Animate active Neo4j subgraph nodes with react-force-graph-2d.
2. Center and polygon-shade historical dynasty maps with Leaflet.js.
3. Expose verbatim Wikipedia citation drawer items in real time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from server.providers.base import GraphHit
from server.rag.retrieve import Hit

logger = logging.getLogger(__name__)

# Pre-indexed historical coordinates [latitude, longitude]
HISTORICAL_GEO: dict[str, list[float]] = {
    "ashoka": [25.61, 85.14],                  # Pataliputra (Patna)
    "maurya": [25.61, 85.14],
    "kalinga": [20.19, 85.84],                 # Dhauli / Kalinga
    "chandragupta": [25.61, 85.14],
    "akbar": [27.18, 78.01],                   # Agra
    "mughal": [28.61, 77.21],                  # Delhi
    "aurangzeb": [27.18, 78.01],
    "shah jahan": [27.18, 78.01],
    "plassey": [23.80, 88.25],                 # Palashi, Bengal
    "panipat": [29.39, 76.97],                 # Panipat, Haryana
    "shivaji": [18.23, 73.44],                 # Raigad Fort
    "maratha": [18.52, 73.85],                 # Pune
    "chola": [10.78, 79.13],                   # Thanjavur
    "rajendra chola": [11.20, 79.45],          # Gangaikonda Cholapuram
    "mohenjo-daro": [27.32, 68.13],            # Sindh
    "indus valley": [30.63, 72.86],            # Harappa
    "jallianwala bagh": [31.63, 74.87],        # Amritsar
    "amritsar": [31.63, 74.87],
    "non-cooperation": [26.65, 83.78],         # Chauri Chaura
    "gandhi": [23.02, 72.57],                  # Ahmedabad / Sabarmati
    "constitution": [28.61, 77.21],            # New Delhi
    "ambedkar": [28.61, 77.21],
}

# Epoch time boundaries [start_year, end_year] (BCE as negative)
HISTORICAL_ERAS: dict[str, tuple[str, list[int]]] = {
    "maurya": ("Maurya Empire", [-322, -185]),
    "ashoka": ("Maurya Empire", [-268, -232]),
    "chandragupta": ("Maurya Empire", [-322, -298]),
    "kalinga": ("Kalinga War", [-261, -261]),
    "indus valley": ("Indus Valley Civilisation", [-2500, -1900]),
    "mohenjo-daro": ("Indus Valley Civilisation", [-2500, -1900]),
    "chola": ("Chola Dynasty", [848, 1279]),
    "rajendra chola": ("Chola Dynasty", [1014, 1044]),
    "mughal": ("Mughal Empire", [1526, 1857]),
    "akbar": ("Mughal Empire", [1556, 1605]),
    "shah jahan": ("Mughal Empire", [1628, 1658]),
    "aurangzeb": ("Mughal Empire", [1658, 1707]),
    "maratha": ("Maratha Empire", [1674, 1818]),
    "shivaji": ("Maratha Empire", [1674, 1680]),
    "panipat": ("Panipat Battles", [1526, 1761]),
    "plassey": ("Battle of Plassey", [1757, 1757]),
    "jallianwala": ("British Raj Period", [1919, 1919]),
    "non-cooperation": ("Independence Movement", [1920, 1922]),
    "gandhi": ("Independence Era", [1869, 1948]),
    "constitution": ("Republic of India", [1949, 1950]),
}


@dataclass
class VisualPayload:
    event: str  # "HISTORICAL_CONTEXT"
    entity: str
    dynasty: str
    era_years: list[int]
    subgraph: dict[str, list[dict[str, Any]]]
    geo_coordinates: list[float]
    citation: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_visual_payload(
    question: str,
    answer: str,
    hits: list[Hit],
    graph_paths: list[GraphHit],
) -> VisualPayload:
    """Builds a rich Visual Stage payload by inspecting the turn's entities,

    hits, and graph traversals.
    """
    text_corpus = (question + " " + answer).lower()

    # Match geo coordinates
    coords = [28.61, 77.21]  # Default to Delhi
    matched_entity = "Indian History"
    for key, c in HISTORICAL_GEO.items():
        if key in text_corpus:
            coords = c
            matched_entity = key.title()
            break

    # Match era and dynasty
    dynasty = "Historical Era"
    era_years = [0, 2026]
    for key, (d_name, years) in HISTORICAL_ERAS.items():
        if key in text_corpus:
            dynasty = d_name
            era_years = years
            break

    # Assemble Subgraph from GraphHits
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    for gh in graph_paths:
        for i in range(0, len(gh.path) - 2, 2):
            source = gh.path[i].get("entity")
            rel = gh.path[i + 1].get("rel", "RELATED_TO")
            target = gh.path[i + 2].get("entity")
            if source and target:
                if source not in seen_nodes:
                    nodes.append({"id": source, "name": source, "group": 1})
                    seen_nodes.add(source)
                if target not in seen_nodes:
                    nodes.append({"id": target, "name": target, "group": 2})
                    seen_nodes.add(target)
                links.append({"source": source, "target": target, "relation": rel})

    if not nodes:
        # Construct fallback 2-node graph for visual clarity
        nodes = [
            {"id": matched_entity, "name": matched_entity, "group": 1},
            {"id": dynasty, "name": dynasty, "group": 2},
        ]
        links = [{"source": matched_entity, "target": dynasty, "relation": "ERA_OF"}]

    # Top citation
    citation = {
        "chunk_id": hits[0].chunk_id if hits else "corpus::overview",
        "article": hits[0].article if hits else "Indian History",
        "section": hits[0].section if hits else "Overview",
        "text": hits[0].text[:300] if hits else "Verified historical records.",
    }

    return VisualPayload(
        event="HISTORICAL_CONTEXT",
        entity=matched_entity,
        dynasty=dynasty,
        era_years=era_years,
        subgraph={"nodes": nodes, "links": links},
        geo_coordinates=coords,
        citation=citation,
    )
