"""Neo4j-backed multi-hop entity graph. Additive only (C11): health() and
query_entity() both degrade to "nothing found" on any failure, never raise —
graph_retrieve.py's caller must always get a valid answer from vector
retrieval alone if this class can't reach the database.
"""

from __future__ import annotations

import logging

from server.config import get_settings
from server.providers.base import GraphHit, GraphProvider

logger = logging.getLogger(__name__)


class Neo4jGraphProvider(GraphProvider):
    name = "neo4j"

    def __init__(self, uri: str | None = None, user: str | None = None,
                password: str | None = None):
        s = get_settings()
        self._uri = uri or s.neo4j_uri
        self._user = user or s.neo4j_user
        self._password = password or s.neo4j_password
        self._driver = None

    @property
    def driver(self):
        """Public on purpose: build_graph.py needs raw write access (MERGE
        entities/relations) that query_entity()'s read-only interface doesn't
        expose."""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    async def health(self) -> bool:
        try:
            driver = self.driver
            async with driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as e:  # noqa: BLE001 - C11: never raise, always degrade
            logger.warning("neo4j unreachable: %r", e)
            return False

    async def query_entity(self, entity_name: str, max_hops: int = 2) -> list[GraphHit]:
        driver = self.driver
        cypher = (
            f"MATCH path = (a:Entity)-[r*1..{int(max_hops)}]-(b:Entity) "
            "WHERE toLower(a.name) = toLower($name) "
            "RETURN path LIMIT 20"
        )
        try:
            hits: list[GraphHit] = []
            async with driver.session() as session:
                result = await session.run(cypher, name=entity_name)
                async for record in result:
                    hits.append(_path_to_hit(record["path"]))
            return hits
        except Exception as e:  # noqa: BLE001 - C11: unreachable graph -> no hits, never a 500
            logger.warning("neo4j query failed for %r: %r", entity_name, e)
            return []


def _path_to_hit(path) -> GraphHit:
    steps: list[dict] = [{"entity": path.nodes[0].get("name")}]
    chunk_ids: set[str] = set()
    confidences: list[float] = []
    for rel, node in zip(path.relationships, path.nodes[1:], strict=True):
        steps.append({"rel": rel.type})
        steps.append({"entity": node.get("name")})
        chunk_ids.update(rel.get("source_chunk_ids") or [])
        confidences.append(rel.get("confidence", 1.0))
    return GraphHit(
        path=steps,
        source_chunk_ids=sorted(chunk_ids),
        confidence=min(confidences) if confidences else 1.0,
    )


class NullGraphProvider(GraphProvider):
    """GRAPH_PROVIDER=none. Always unhealthy, so callers' C11 fallback path
    (vector-only) runs with zero special-casing for "graph disabled"."""

    name = "none"

    async def health(self) -> bool:
        return False

    async def query_entity(self, entity_name: str, max_hops: int = 2) -> list[GraphHit]:
        return []
